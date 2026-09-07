#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shelve
import logging
import networkx as nx
import hashlib

from lib import geometry

from datetime import datetime

from pathlib import Path

from lib.Triangle import Triangle, Deadend

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from datetime import timedelta

from pprint import pformat

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

TRIES_PER_TRI = 10

CAPTUREAP = 500+(125*8)+250+(125*2)
LINKAP = 313
FIELDAP = 1250

cooltime = {
    'none': 5,
    'hs': 4,
    'rhs': 2.5,
    'vrhs': 1.5,
}

logger = logging.getLogger('fieldplan')

combined_graph = None
portal_graph = None
waypoint_graph = None
active_graph = None

cooling = 'rhs'
# None means no AP floor. Must not be a number: the guard in make_workplan()
# only tests 'is not None', so any numeric default makes every plan look
# under-AP and return the bare linkplan before capture routing ever runs.
minap = None
keysperhack = 1.5
coolthreshold = 5
maxmu = False
travelmode = 'walking'
maxtime = None
# ms of guided local search to improve each capture route; 0 = greedy only
capture_search_ms = 200
# Subset (--maxtime) routes are solved on every cache miss, so cap their
# budget by size: measured gains saturate at roughly this many ms per node.
SUBSET_SEARCH_MS_PER_NODE = 3

# Default iteration count when -i is not given. An iteration costs roughly
# ITERATION_MS_AT_10 * (n/10)**ITERATION_MS_EXP milliseconds on one core;
# that curve was fitted to measured runs at n = 10, 15, 20, 30 and 40 and is
# within 2.2% across the range. Spending a fixed slice of single-core work
# gives small portal lists the extra restarts they can afford almost for
# free, while the floor keeps quality for the large ones, where an iteration
# is expensive but few of them are affordable anyway.
ITERATION_MS_AT_10 = 3.5
ITERATION_MS_EXP = 2.75
ITERATION_BUDGET_MS = 100000
ITERATIONS_MIN = 5000
ITERATIONS_MAX = 20000

capture_cache = dict()
dist_matrix = list()
time_matrix = list()
direct_dist_matrix = list()
smallest_triangle = None
largest_triangle = None
seen_subsets = list()

# in metres per minute, only used in the absence of Google Maps API
travel_speed = {
    'walking': 80,
    'bicycling': 300,
    'driving': 1000,
    'transit': 500,
}


def get_cache_dir():
    home = str(Path.home())
    cachedir = os.path.join(home, '.cache', 'ingress-fieldmap')
    Path(cachedir).mkdir(parents=True, exist_ok=True)
    return cachedir


def gen_distance_matrix(gmapskey=None):
    global dist_matrix
    global direct_dist_matrix

    cachedir = get_cache_dir()
    distcachefile = os.path.join(cachedir, 'distcache')
    # Google Maps lookups are non-free, so cache them aggressively
    # TODO: Invalidate these somehow after a period?
    _gmap_cache_db = shelve.open(distcachefile, 'c')

    # Do we have a gmaps key?
    if gmapskey is None:
        # Do we have a cached copy in the cache?
        if 'clientkey' in _gmap_cache_db:
            gmapskey = _gmap_cache_db['clientkey']
    else:
        # save it in the cache db if not present or different
        if 'clientkey' not in _gmap_cache_db or _gmap_cache_db['clientkey'] != gmapskey:
            logger.info('Caching google maps key for future lookups')
            _gmap_cache_db['clientkey'] = gmapskey

    gmaps = None
    if gmapskey:
        import googlemaps
        gmaps = googlemaps.Client(key=gmapskey)
        logger.info('Generating the distance matrix using Google Maps API, may take a moment')
    else:
        logger.info('Generating the distance matrix')

    a = combined_graph.copy()
    n = a.order()
    logger.debug('n=%s', n)

    # We consider any direct distance shorter than 40m as effectively 0,
    # since the agent doesn't need to travel to access both portals.
    for p1 in range(n):
        matrow = list()
        matrow_dur = list()
        direct_matrow = list()
        for p2 in range(n):
            # Do direct distance first
            p1pos = a.nodes[p1]['geo']
            p2pos = a.nodes[p2]['geo']
            dist = int(geometry.sphereDist(p1pos, p2pos)[0])
            duration = int(dist/travel_speed[travelmode])
            direct_matrow.append(dist)
            logger.debug('%s -( %d )-> %s (Direct)', a.nodes[p1]['name'], dist, a.nodes[p2]['name'])

            # If it's over 40 meters and we have a gmaps client key,
            # look up the actual distance using google maps API
            if dist > 40 and gmaps is not None:
                p1pos = a.nodes[p1]['pll']
                p2pos = a.nodes[p2]['pll']
                dkey = '%s,%s,%s' % (p1pos, p2pos, travelmode)
                rkey = '%s,%s,%s' % (p2pos, p1pos, travelmode)
                dkey_dur = '%s_dur' % dkey
                rkey_dur = '%s_dur' % rkey

                if dkey in _gmap_cache_db and dkey_dur in _gmap_cache_db:
                    dist = _gmap_cache_db[dkey]
                    duration = _gmap_cache_db[dkey_dur]
                    logger.debug('%s -( %d )-> %s (Google/%s/cached)', a.nodes[p1]['name'],
                                 dist, a.nodes[p2]['name'], travelmode)
                elif rkey in _gmap_cache_db and rkey_dur in _gmap_cache_db:
                    dist = _gmap_cache_db[rkey]
                    duration = _gmap_cache_db[rkey_dur]
                    logger.debug('%s -( %d )-> %s (Google/%s/cached)', a.nodes[p1]['name'],
                                 dist, a.nodes[p2]['name'], travelmode)
                else:
                    # Perform the lookup
                    now = datetime.now()
                    gdir = gmaps.directions(p1pos, p2pos, mode=travelmode, departure_time=now)
                    dist = gdir[0]['legs'][0]['distance']['value']
                    duration = int(gdir[0]['legs'][0]['duration']['value']/60)
                    _gmap_cache_db[dkey] = dist
                    _gmap_cache_db[dkey_dur] = duration
                    logger.debug('%s -( %d )-> %s (Google/%s/lookup)', a.nodes[p1]['name'],
                                 dist, a.nodes[p2]['name'], travelmode)

            matrow.append(dist)
            matrow_dur.append(duration)

        direct_dist_matrix.append(direct_matrow)
        dist_matrix.append(matrow)
        time_matrix.append(matrow_dur)


# Lookup tables for the current active_graph: (graph, dist, time, blocker, keys)
# dist/time are n x n lists indexed by active-graph node id, so the hot
# loops in get_workplan_stats never go through the 'pos' indirection.
_active_tables = (None, None, None, None, None)


def get_active_tables():
    global _active_tables
    a = active_graph
    if _active_tables[0] is not a:
        n = a.order()
        pos = [a.nodes[i]['pos'] for i in range(n)]
        dist = [[dist_matrix[pos[i]][pos[j]] for j in range(n)] for i in range(n)]
        tim = [[int(time_matrix[pos[i]][pos[j]]) for j in range(n)] for i in range(n)]
        blocker = [a.nodes[i].get('special') == '_w_blocker' for i in range(n)]
        keys = [a.nodes[i].get('keys', 0) for i in range(n)]
        _active_tables = (a, dist, tim, blocker, keys)
    return _active_tables[1:]


def get_portal_distance(p1, p2, direct=False):
    if active_graph is not None:
        p1 = active_graph.nodes[p1]['pos']
        p2 = active_graph.nodes[p2]['pos']
    if direct:
        return direct_dist_matrix[p1][p2]
    return dist_matrix[p1][p2]


def get_portal_time(p1, p2):
    if active_graph is not None:
        p1 = active_graph.nodes[p1]['pos']
        p2 = active_graph.nodes[p2]['pos']
    return int(time_matrix[p1][p2])


def estimate_iteration_ms(n):
    # Single-core cost of one solver iteration at n portals
    if n < 3:
        return ITERATION_MS_AT_10
    return ITERATION_MS_AT_10 * (n / 10.0) ** ITERATION_MS_EXP


def default_iterations(n):
    """
    How many random restarts to run when the user did not say.

    Iterations needed to converge fall as the list grows, because each
    iteration does more work inside improve_workplan. Measured iteration of
    the last improvement: 14,339 at n=10, 8,720 at n=15, 5,684 at n=20,
    3,085 at n=30, 2,132 at n=40. At 5,000 iterations a run lands within
    about 1-2% of a very long run (99.1% at n=10, 98.8% at n=15, 97.7% at
    n=20, 100% at n=30 and n=40), so 5,000 is the floor.

    Small lists are cheap enough to search much harder, so the budget gives
    them more, up to a cap that keeps any run bounded. The result tracks the
    measured need: 20,000 at n=10, 9,500 at n=15, and the floor from n=20 up.
    """
    if n < 3:
        return ITERATIONS_MIN
    raw = ITERATION_BUDGET_MS / estimate_iteration_ms(n)
    raw = int(round(raw / 500.0) * 500)
    return max(ITERATIONS_MIN, min(ITERATIONS_MAX, raw))


def dedupe_portals(portals):
    # Two entries at one location are the same portal in game, so a plan
    # built from both tells you to walk to where you already are and to
    # link a portal to itself. Drop the later entries, keep the first.
    seen = dict()
    unique = list()
    for row in portals:
        pll = row[1].strip()
        if pll in seen:
            logger.warning('Ignoring "%s": same location as "%s" (%s)', row[0], seen[pll], pll)
            continue
        seen[pll] = row[0]
        unique.append(row)
    return unique


def populate_graphs(portals, waypoints):
    global combined_graph
    global portal_graph
    global waypoint_graph
    global active_graph
    a, basis = populate_graph(portals)
    # a graph with just portals
    portal_graph = a
    # Make a master graph that contains both portals and waypoints
    combined_graph = a.copy()
    if waypoints:
        waypoint_graph, _ = populate_graph(waypoints, basis=basis)
        extend_graph_with_waypoints(combined_graph)


def extend_graph_with_waypoints(a):
    if waypoint_graph is None:
        return
    master_num = portal_graph.order()
    num = a.order()
    for i in range(waypoint_graph.order()):
        attrs = waypoint_graph.nodes[i]
        a.add_node(num, **attrs)
        a.nodes[num]['pos'] = master_num
        num += 1
        master_num += 1


def populate_graph(portals, basis=None):
    a = nx.DiGraph()
    locs = []

    for num, row in enumerate(portals):
        a.add_node(num)
        a.nodes[num]['name'] = row[0]
        coord_parts = row[1].split(',')
        a.nodes[num]['pll'] = row[1]
        if len(row) > 2:
            a.nodes[num]['special'] = row[2]
        else:
            a.nodes[num]['special'] = None
        # Keys already in hand for this portal; they save hacking time
        if len(row) > 3 and row[3]:
            a.nodes[num]['keys'] = int(row[3])
        else:
            a.nodes[num]['keys'] = 0
        lat = int(float(coord_parts[0]) * 1.e6)
        lon = int(float(coord_parts[1]) * 1.e6)
        locs.append(np.array([lat, lon], dtype=float))

    n = a.order()
    locs = np.array(locs, dtype=float)

    # Convert coords to radians, then to cartesian, then to
    # gnomonic projection
    locs = geometry.e6LLtoRads(locs)
    xyz = geometry.radstoxyz(locs)
    
    if basis is None:
        basis = xyz.mean(0)
        basis /= np.linalg.norm(basis)

    xy = geometry.gnomonicProj(locs, xyz, basexyz=basis)

    for i in range(n):
        a.nodes[i]['pos'] = i
        a.nodes[i]['geo'] = locs[i]
        a.nodes[i]['xyz'] = xyz[i]
        a.nodes[i]['xy'] = xy[i]

    return a, basis


def make_workplan(a, is_subset=False):
    global active_graph
    global capture_cache

    linkplan = [None] * a.size()

    for p, q in a.edges():
        linkplan[a.edges[p, q]['order']] = (p, q, len(a.edges[p, q]['fields']))

    if minap is not None:
        stats = get_workplan_stats(linkplan)
        if stats['ap'] < minap:
            logger.debug('Plan does not have enough AP, abandon early')
            return linkplan, stats

    # pre-optimize linkplan without the captures first
    linkplan, stats = improve_workplan(linkplan)

    w_start = None
    w_end = None

    for i in range(a.order()):
        # skip non-special nodes
        if 'special' not in a.nodes[i]:
            continue
        # Is it a start enpoint?
        if a.nodes[i]['special'] == '_w_start':
            w_start = i
        elif a.nodes[i]['special'] == '_w_end':
            w_end = i

    if w_start is None:
        # Find the portal that's furthest away from the starting portal
        maxdist = None
        for p in range(a.order()):
            dist = get_portal_distance(linkplan[0][0], p)
            if maxdist is None or dist > maxdist:
                w_start = p
                maxdist = dist

        logger.debug('Furthest from %s is %s', a.nodes[linkplan[0][0]]['name'], a.nodes[w_start]['name'])

    cachekey = [w_start, linkplan[0][0]]
    if is_subset:
        subset_key = list()
        for n in range(a.order()):
            subset_key.append(a.nodes[n]['pos'])
        subset_key.sort()
        cachekey = cachekey + subset_key
    cachekey = tuple(cachekey)

    if cachekey not in capture_cache:
        logger.debug('Capture cache miss, starting ortools calculation')
        dist, _, _, _ = get_active_tables()
        search_ms = capture_search_ms
        if is_subset:
            search_ms = min(search_ms, SUBSET_SEARCH_MS_PER_NODE * a.order())
        dist_ordered = solve_capture_route(dist, w_start, linkplan[0][0], search_ms)
        capture_cache[cachekey] = dist_ordered
        if dist_ordered is None:
            logger.debug('Could not solve for these constraints, ignoring plan')
            return None, None
    else:
        logger.debug('Capture cache hit')
        if capture_cache[cachekey] is None:
            logger.debug('Known unsolvable, ignoring')
            return None, None
        dist_ordered = capture_cache[cachekey]

    logger.debug('dist_ordered=%s', dist_ordered)
    a.captureplan = dist_ordered

    # Make a unified workplan
    workplan = []
    for p in dist_ordered:
        workplan.append((p, None, 0))
    workplan.extend(linkplan)
    if w_end is not None:
        logger.debug('Adding end waypoint to the workplan')
        workplan.append((w_end, None, 0))

    workplan, stats = improve_workplan(workplan)

    return workplan, stats


def solve_capture_route(dist, w_start, w_end, search_ms):
    """
    Order in which to visit every node of an n x n distance matrix, starting
    at w_start and finishing at w_end. Returns None if unsolvable. Pure
    function of its arguments so it can run in a worker pool.
    """
    manager = pywrapcp.RoutingIndexManager(len(dist), 1, [w_start], [w_end])
    routing = pywrapcp.RoutingModel(manager)

    # Hand the solver the whole matrix up front; a Python callback would
    # be invoked tens of thousands of times per solve.
    transit_callback_index = routing.RegisterTransitMatrix(dist)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    )
    if search_ms > 0:
        # Improve on the greedy first solution
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromMilliseconds(search_ms)
    logger.debug('Starting solver')
    assignment = routing.SolveWithParameters(search_parameters)
    logger.debug('Ended solver')

    if not assignment:
        return None

    index = routing.Start(0)
    dist_ordered = list()
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        dist_ordered.append(node)
        index = assignment.Value(routing.NextVar(index))

    return dist_ordered


def _solve_capture_key(job):
    cachekey, dist, w_start, w_end, search_ms = job
    return cachekey, solve_capture_route(dist, w_start, w_end, search_ms)


def precompute_capture_routes(ncpus):
    """
    Solve every capture route a full-graph run can ask for, in parallel,
    before the workers start. Routes depend only on where the linking
    starts (the start waypoint, or failing that the portal furthest from
    the first link), so there are at most one per portal, and every worker
    would otherwise solve the same ones independently.
    """
    global active_graph
    a = combined_graph
    active_graph = a
    dist, _, _, _ = get_active_tables()

    w_start = None
    for i in range(a.order()):
        if a.nodes[i]['special'] == '_w_start':
            w_start = i

    jobs = list()
    for first in range(portal_graph.order()):
        if w_start is None:
            # Same rule as make_workplan: furthest node from the first link
            start = max(range(a.order()), key=lambda p: (get_portal_distance(first, p), -p))
        else:
            start = w_start
        cachekey = (start, first)
        if cachekey not in capture_cache:
            jobs.append((cachekey, dist, start, first, capture_search_ms))

    if not jobs:
        return
    logger.info('Precomputing %s capture routes using %s processes', len(jobs), ncpus)
    import multiprocessing
    with multiprocessing.Pool(min(ncpus, len(jobs))) as pool:
        for cachekey, route in pool.imap_unordered(_solve_capture_key, jobs):
            capture_cache[cachekey] = route
    active_graph = None


def get_portals_perimeter(p1, p2, p3, direct=False):
    s1 = get_portal_distance(p1, p2, direct=direct)
    s2 = get_portal_distance(p2, p3, direct=direct)
    s3 = get_portal_distance(p1, p3, direct=direct)
    perimeter = s1+s2+s3
    logger.debug('Triangle %s-%s-%s, perimeter: %s m', p1, p2, p3, perimeter)

    return perimeter


def get_portals_area(p1, p2, p3):
    s1 = get_portal_distance(p1, p2, direct=True)
    s2 = get_portal_distance(p2, p3, direct=True)
    s3 = get_portal_distance(p1, p3, direct=True)
    s = (s1 + s2 + s3)/2
    try:
        area = int(np.sqrt(s * (s - s1) * (s - s2) * (s - s3)))
    except ValueError:
        # Effectively, 0
        area = 0
    logger.debug('Triangle %s-%s-%s, area: %s m2', p1, p2, p3, area)
    return area


def reverse_edge(p, q):
    logger.debug('Reversing %s->%s for a better plan', p, q)
    attrs = active_graph.edges[p, q]
    active_graph.add_edge(q, p, **attrs)
    active_graph.remove_edge(p, q)


def get_needed_keys(workplan, keys_t=None):
    """
    For every index where the agent arrives at a portal (first entry of a
    run of consecutive actions at the same portal), work out how many keys
    for that portal are needed before we come back to it, and whether this
    is the last visit.

    needkeys: keys for all future links to p if this is the last visit,
              otherwise only the links to p made before the next visit.
    Computed in a single backward pass instead of a lookahead per visit.

    keys_t, if given, is the number of keys already in hand per portal.
    Those are a budget for the whole run, not per visit, so they are spent
    against the earliest visits that need them.
    """
    n = len(workplan)
    links_to = dict()     # portal -> links to it at indices > current
    at_next_visit = dict()  # portal -> links_to at the start of its next visit
    pending = dict()
    needkeys = [0] * n
    lastvisit = [True] * n
    for idx in range(n - 1, -1, -1):
        p, q, f = workplan[idx]
        if idx == n - 1 or workplan[idx + 1][0] != p:
            # last action of a visit: links_to counts everything after it
            total = links_to.get(p, 0)
            if p in at_next_visit:
                pending[p] = (total - at_next_visit[p], False)
            else:
                pending[p] = (total, True)
        if idx == 0 or workplan[idx - 1][0] != p:
            # first action of a visit
            needkeys[idx], lastvisit[idx] = pending[p]
            at_next_visit[p] = links_to.get(p, 0)
        if q is not None:
            links_to[q] = links_to.get(q, 0) + 1

    if keys_t and any(keys_t):
        budget = dict()
        for idx in range(n):
            if not needkeys[idx]:
                continue
            p = workplan[idx][0]
            if p not in budget:
                budget[p] = keys_t[p]
            if not budget[p]:
                continue
            spend = min(budget[p], needkeys[idx])
            needkeys[idx] -= spend
            budget[p] -= spend

    return needkeys, lastvisit


def get_workplan_stats(workplan):
    workplan = remove_useless_captures(workplan)
    dist_t, time_t, blocker, keys_t = get_active_tables()
    needkeys_at, lastvisit_at = get_needed_keys(workplan, keys_t)

    totalap = active_graph.order() * CAPTUREAP
    totaldist = 0
    totaltime = 0
    totalarea = 0
    traveltime = 0
    links = 0
    fields = 0
    hscount = 0
    hs_at = list()
    portal_times = [0] * active_graph.order()

    try:
        totalarea = active_graph.totalarea
        need_area = False
    except AttributeError:
        need_area = True

    prev_p = None
    seen_p = set()
    time_at_portal = 0
    for idx, (p, q, f) in enumerate(workplan):
        # Are we at a different location than the previous portal?
        if p != prev_p:
            # Append previous portal's time_at_portal to total time
            totaltime += time_at_portal
            if prev_p is not None:
                portal_times[prev_p] += time_at_portal

            # We are at a new portal, so add half a minute just because
            # it takes time to get positioned and get to the right
            # screen in the UI
            time_at_portal = 0.5
            # Are we capturing?
            if p not in seen_p:
                # Add half a minute for capturing, unless idkfa
                if cooling != 'idkfa':
                    time_at_portal += 0.5
                seen_p.add(p)

            if prev_p is not None:
                duration = time_t[prev_p][p]
                totaltime += duration
                traveltime += duration
                dist = dist_t[prev_p][p]
                if dist > 40:
                    totaldist += dist

            # Are we at a blocker?
            if blocker[p]:
                # assume it takes 3 minutes to destroy a blocker
                time_at_portal += 3
                prev_p = p
                continue

            # How many keys do we need if/until we come back?
            needkeys = needkeys_at[idx]
            lastvisit = lastvisit_at[idx]

            # IDKFA means you already have all the keys
            if needkeys and cooling != 'idkfa':
                # We assume:
                # - we get roughly 1.5 keys per each hack (override with --keys-per-hack)
                # - we glyph-hack, meaning it takes about 30 seconds per actual hack action
                # - we'll use a Heat Sink only if we'd spend more than 10 min at a portal
                #   (override with --cool-if-longer-than)
                if keysperhack != 1:
                    needed_hacks = int((needkeys/keysperhack) + (needkeys % keysperhack))
                else:
                    needed_hacks = needkeys
                time_at_portal += needed_hacks/2
                wait_time = cooltime['none']*(needed_hacks-1)
                if cooling != 'none' and wait_time >= coolthreshold:
                    # Apply a heat sink and try again
                    hscount += 1
                    hs_at.append(p)
                    # second hack is free regardless of the type of HS
                    wait_time = cooltime[cooling]*(needed_hacks-2)

                time_at_portal += wait_time

            if lastvisit:
                # Add half a minute for putting on shields
                time_at_portal += 0.5

            prev_p = p

        if q is None:
            continue

        # Add 15 seconds per link
        time_at_portal += 0.25
        totalap += LINKAP
        links += 1

        if not f:
            continue

        fields += f
        totalap += FIELDAP*f

        # Total area of a graph doesn't change regardless of the order
        # of linking and fielding, so calculate it only once.
        if need_area:
            for t in active_graph.edges[p, q]['fields']:
                area = get_portals_area(t[0], t[1], t[2])
                totalarea += area

    if need_area:
        active_graph.totalarea = totalarea

    # Add time at the last portal
    totaltime += time_at_portal
    portal_times[p] = time_at_portal

    stats = {
        'time': totaltime,
        'nicetime': str(timedelta(minutes=totaltime)),
        'traveltime': traveltime,
        'nicetraveltime': str(timedelta(minutes=traveltime)),
        'portaltimes': portal_times,
        'hs': hscount,
        'hs_at': hs_at,
        'ap': totalap,
        'dist': totaldist,
        'area': totalarea,
        'links': links,
        'fields': fields,
        'sqmpmin': int(totalarea/totaltime),
        'appmin': int(totalap/totaltime),
    }

    logger.debug('stats: %s', stats)

    return stats


def workplan_is_better(orig_stats, new_stats):
    if maxmu:
        if new_stats['sqmpmin'] > orig_stats['sqmpmin']:
            logger.debug('old best: %s, new best: %s', orig_stats['sqmpmin'], new_stats['sqmpmin'])
            logger.debug('New plan has better coverage')
            return True
        logger.debug('New plan is not better')
        return False

    if new_stats['appmin'] > orig_stats['appmin']:
        logger.debug('old best: %s, new best: %s', orig_stats['appmin'], new_stats['appmin'])
        logger.debug('New plan has better AP score')
        return True
    logger.debug('New plan is not better')
    return False


def triangle_edges(a, t):
    # The three edges of a field, in whichever direction they were built
    out = []
    for i in range(3):
        x, y = t[i-1], t[i-2]
        if not a.has_edge(x, y):
            x, y = y, x
        out.append((x, y))
    return out


def get_link_depends(a):
    """
    edge -> set of edges that have to be made before it.

    markEdgesWithFields() records a field on whichever of its three edges
    came last, so that edge is the one that completes it and the other two
    have to precede it. That is the whole dependency relation, and it is
    what makes a link movable or not. The old code approximated it with the
    field count on the link ("2 fields means frozen, 1 means reversible in
    place only"), which refuses plenty of reorderings that are legal.

    Enforcing it also keeps the bookkeeping honest, since the edge carrying
    a field stays the last of its three.

    Edges are keyed as frozensets: whether a field can be completed depends
    on its three links existing, not on which way round each was thrown, so
    reversing a link leaves the relation untouched.
    """
    depends = dict()
    for p, q in a.edges():
        deps = set()
        for t in a.edges[p, q]['fields']:
            for e in triangle_edges(a, t):
                key = frozenset(e)
                if key != frozenset((p, q)):
                    deps.add(key)
        depends[frozenset((p, q))] = deps
    return depends


def workplan_is_valid(workplan, depends, maxlinks=8):
    """
    A plan is playable when every link is thrown from the portal we are
    standing at, to a portal we have already captured, after the other two
    edges of any field it completes, and without exceeding the outbound
    link limit.
    """
    seen = set()
    made = set()
    outdeg = dict()
    for p, q, f in workplan:
        seen.add(p)
        if q is None:
            continue
        if q not in seen:
            return False
        key = frozenset((p, q))
        for d in depends.get(key, ()):
            if d not in made:
                return False
        outdeg[p] = outdeg.get(p, 0) + 1
        if outdeg[p] > maxlinks:
            return False
        made.add(key)
    return True


def improve_workplan(workplan):
    a = active_graph
    a.orig_workplan = list(workplan)
    a.fixes = list()
    depends = get_link_depends(a)
    rcount = 0
    current_stats = get_workplan_stats(workplan)
    fielders_moved = False
    while True:
        rcount += 1
        logger.debug('Starting improve_workplan round %s', rcount)
        # logger.debug('Current workplan:\n%s', pformat(workplan))
        m = len(workplan)-1
        visited_origins = [workplan[0][0]]
        reordered = False
        improved = False
        for i in range(m):
            logger.debug('Workplan is at %s: %s', i, workplan[i])
            p, q, f = workplan[i]
            if p not in visited_origins:
                visited_origins.append(p)
            # we moved to a new origin
            # Find all actions involving this origin
            # and any of the visited origins where the number
            # of fields is fewer than 2
            for j in range(i+1, m):
                jp, jq, jf = workplan[j]
                if jp != p and jq != p:
                    continue
                if jp not in visited_origins or jq not in visited_origins:
                    continue
                # If previous origin and next origin are same, then we don't
                # need to do anything
                if m-j > 2 and workplan[j-1][0] == jp and workplan[j+1][0] == jp:
                    continue

                logger.debug('Improvement candidate: %s', workplan[j])
                # Any link may move, in either direction, as long as the plan
                # stays playable. workplan_is_valid() is what decides that
                # now; the field count on the link no longer gates it.
                nwp = list(workplan)
                del(nwp[j])
                if q is None:
                    del(nwp[i])
                    newpos = i
                else:
                    newpos = i+1

                for reversed_move in (False, True):
                    if not reversed_move:
                        if p != jp:
                            continue
                        moved = (jp, jq, jf)
                    else:
                        if p != jq or a.out_degree(jq) >= 8:
                            continue
                        moved = (jq, jp, jf)

                    cwp = list(nwp)
                    cwp.insert(newpos, moved)
                    if not workplan_is_valid(cwp, depends):
                        continue
                    new_stats = get_workplan_stats(cwp)
                    if not workplan_is_better(current_stats, new_stats):
                        continue
                    if reversed_move:
                        a.fixes.append('R%s: Reversed and moved %s to %s' % (rcount, workplan[j], newpos))
                        reverse_edge(jp, jq)
                    else:
                        a.fixes.append('R%s: Moved %s to %s' % (rcount, workplan[j], newpos))
                    logger.debug(a.fixes[-1])
                    workplan = cwp
                    current_stats = new_stats
                    improved = reordered = True
                    break

                if reordered:
                    break

            if reordered:
                break

            # Reversing in place needs no relocation, so it is worth trying
            # for any link, not just the ones completing a single field
            elif q is not None and a.out_degree(q) < 8:
                nwp = list(workplan)
                nwp[i] = (q, p, f)
                if workplan_is_valid(nwp, depends):
                    new_stats = get_workplan_stats(nwp)
                    if workplan_is_better(current_stats, new_stats):
                        a.fixes.append('R%s: In-place reversed %s at %s' % (rcount, workplan[i], i))
                        logger.debug(a.fixes[-1])
                        reverse_edge(p, q)
                        workplan = nwp
                        current_stats = new_stats
                        fielders_moved = True
                        improved = True

        if reordered:
            logger.debug('Plan was reordered, restart the loop')
            continue

        logger.debug('Reached the end of the workplan')
        if not improved:
            logger.debug('No further improvements found')
            break

        # Run it again, Stan!
        logger.debug('Plan was improved, going for another loop')

    workplan = remove_useless_captures(workplan)

    logger.debug('Renumbering links')
    # Record the new order of edges
    fc = 0
    for i in range(len(workplan)):
        p, q, f = workplan[i]
        if q is None:
            continue
        a.edges[p, q]['order'] = fc
        fc += 1
        if fielders_moved:
            a.edges[p, q]['fields'] = list()

    if fielders_moved:
        logger.debug('Recalculating fields')
        for t in a.triangulation:
            t.markEdgesWithFields()

    # Stick linkplan into a for debugging purposes
    a.workplan = workplan
    # logger.debug('Final workplan:\n%s', pformat(workplan))
    stats = get_workplan_stats(workplan)
    # logger.debug('Final stats:\n%s', pformat(stats))

    return workplan, stats


def remove_useless_captures(workplan):
    # A capture is useless if we visit that portal again before anything
    # links to it. One backward pass tracking the next event per portal.
    next_event = dict()
    keep = [True] * len(workplan)
    for idx in range(len(workplan) - 1, -1, -1):
        p, q, f = workplan[idx]
        if q is None and next_event.get(p) == 'visit':
            logger.debug('Removing useless capture at pos %s: %s', idx, workplan[idx])
            keep[idx] = False
        next_event[p] = 'visit'
        if q is not None:
            next_event[q] = 'link'
    return [w for w, k in zip(workplan, keep) if k]


def remove_since(a, m, t):
    # Remove all but the first m edges from a (and .edge_stck)
    # Remove all but the first t Triangules from a.triangulation
    for i in range(len(a.edgeStack) - m):
        p, q = a.edgeStack.pop()
        a.remove_edge(p, q)
        logger.debug('removing, p=%s, q=%s', p, q)
        logger.debug('edgeStack follows')
        logger.debug(a.edgeStack)
    while len(a.triangulation) > t:
        a.triangulation.pop()


def triangulate(a, perim):
    """
    Recursively tries every triangulation in search a feasible one
        Each layer
            makes a Triangle out of three perimeter portals
            for every feasible way of max-fielding that Triangle
                try triangulating the two perimeter-polygons to the sides of the Triangle

    Returns True if a feasible triangulation has been made in graph a
    """
    pn = len(perim)
    if pn < 3:
        return True

    try:
        start_stack_len = len(a.edgeStack)
    except AttributeError:
        start_stack_len = 0
        a.edgeStack = []
    try:
        start_tri_len = len(a.triangulation)
    except AttributeError:
        start_tri_len = 0
        a.triangulation = []

    # Try all triangles using perim[0:2] and another perim node
    for i in np.random.permutation(range(2, pn)):

        for j in range(TRIES_PER_TRI):
            t0 = Triangle(perim[[0, 1, i]], a, True)
            t0.findContents()
            t0.randSplit()
            try:
                t0.buildGraph()
            except Deadend:
                # remove the links formed since beginning of loop
                remove_since(a, start_stack_len, start_tri_len)
            else:
                # This build was successful. Break from the loop
                break
        else:
            # The loop ended "normally" so this triangle failed
            continue

        if not triangulate(a, perim[range(1, i+1)]):
            # remove the links formed since beginning of loop
            remove_since(a, start_stack_len, start_tri_len)
            continue

        if not triangulate(a, perim[range(0, i-pn-1, -1)]):
            # remove the links formed since beginning of loop
            remove_since(a, start_stack_len, start_tri_len)
            continue

        # This will be a list of the first generation triangles
        a.triangulation.append(t0)

        # This triangle and the ones to its sides succeeded
        logger.debug('Succeeded with perim=%s', perim)
        return True

    # Could not find a solution
    logger.debug('Failed with perim=%s', perim)
    return False


def make_subset(minportals, random_start=False):
    global active_graph
    global smallest_triangle
    global largest_triangle

    if smallest_triangle is None:
        # for smallest, we look for a triangle with the shortest perimeter
        # for largest, we look for a triangle with the largest area
        sperim = np.inf
        larea = 0
        active_graph = None
        for p1 in range(portal_graph.order()):
            for p2 in range(p1+1, portal_graph.order()):
                for p3 in range(p2+1, portal_graph.order()):
                    area = get_portals_area(p1, p2, p3)
                    perim = get_portals_perimeter(p1, p2, p3)
                    if area > larea:
                        larea = area
                        largest_triangle = (p1, p2, p3)
                    if perim < sperim:
                        sperim = perim
                        smallest_triangle = (p1, p2, p3)

    if random_start:
        allp = list(range(portal_graph.order()))
        # Ensure we have at least 3 portals
        if len(allp) >= 3:
            subset = list(np.random.choice(allp, 3, replace=False))
            # Convert numpy types to native python ints
            subset = [int(x) for x in subset]
        else:
            subset = list(allp)
    elif maxmu:
        subset = list(largest_triangle)
    else:
        subset = list(smallest_triangle)
    # Add portals until we get to minportals
    while len(subset) < minportals:
        add_subset_portal(subset)
    return subset


def add_subset_portal(subset):
    global seen_subsets
    allp = list(range(portal_graph.order()))
    missing = [x for x in allp if x not in subset]
    if not missing:
        return
    if maxmu:
        maxtry = 0
        while True:
            candidate = np.random.choice(missing)
            subset.append(candidate)
            if maxtry > 10 or subset not in seen_subsets:
                seen_subsets.append(list(subset))
                break
            subset.pop()
            maxtry += 1
        return

    candidate = None
    slen = None
    for i in missing:
        mylen = 0
        for p in subset:
            mylen += get_portal_distance(p, i)
        if slen is None or mylen < slen:
            candidate = i
            slen = mylen
    if candidate is not None:
        subset.append(candidate)


def make_subset_graph(subset):
    subset.sort()
    b = nx.DiGraph()
    ct = 0
    for num in subset:
        attrs = portal_graph.nodes[num]
        b.add_node(ct, **attrs)
        ct += 1
    return b


def max_fields(a):
    n = a.order()
    # Generate a distance matrix for all portals
    pts = np.array([a.nodes[i]['xy'] for i in range(n)])
    perim = np.array(geometry.getPerim(pts))

    if not triangulate(a, perim):
        logger.debug('Could not triangulate')
        return False

    return True


def gen_cache_key():
    plls = list()
    a = combined_graph
    for m in range(a.order()):
        plls.append(a.nodes[m]['pll'])
    h = hashlib.sha1()
    for pll in plls:
        h.update(pll.encode('utf-8'))
    phash = h.hexdigest()
    cachekey = travelmode
    if maxmu:
        cachekey += '+maxmu'
    if cooling != 'rhs':
        cachekey += '+%s' % cooling
    if maxtime:
        cachekey += '+timelimit-%s' % maxtime
    cachekey += '-%s' % phash

    return cachekey


def save_cache(bestgraph, bestplan):
    # let's cache processing results for the same portals, just so
    # we can "add more cycles" to existing best plans
    # We use portal pll coordinates to generate the cache file key
    # and dump a in there.
    cachekey = gen_cache_key()
    cachedir = get_cache_dir()
    plancachedir = os.path.join(cachedir, 'plans')
    Path(plancachedir).mkdir(parents=True, exist_ok=True)
    cachefile = os.path.join(plancachedir, cachekey)
    wc = shelve.open(cachefile, 'c')
    wc['bestplan'] = bestplan
    wc['bestgraph'] = bestgraph
    logger.info('Saved plan cache in %s', cachefile)
    wc.close()


def load_cache():
    global active_graph

    cachekey = gen_cache_key()
    cachedir = get_cache_dir()
    plancachedir = os.path.join(cachedir, 'plans')
    cachefile = os.path.join(plancachedir, cachekey)
    bestgraph = None
    bestplan = None
    try:
        wc = shelve.open(cachefile, 'r')
        logger.info('Loading cache data from cache %s', cachefile)
        bestgraph = wc['bestgraph']
        bestplan = wc['bestplan']
        wc.close()
    except:
        pass

    active_graph = bestgraph
    return bestgraph, bestplan
