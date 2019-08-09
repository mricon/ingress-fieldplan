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

_capture_cache = dict()
_dist_matrix = list()
_time_matrix = list()
_direct_dist_matrix = list()
_smallest_triangle = None
_largest_triangle = None
_seen_subsets = list()

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


def gen_distance_matrix(gmapskey=None, gmapsmode='walking'):
    global _dist_matrix
    global _direct_dist_matrix

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
            p1pos = a.node[p1]['geo']
            p2pos = a.node[p2]['geo']
            dist = int(geometry.sphereDist(p1pos, p2pos)[0])
            duration = int(dist/travel_speed[gmapsmode])
            direct_matrow.append(dist)
            logger.debug('%s -( %d )-> %s (Direct)', a.node[p1]['name'], dist, a.node[p2]['name'])

            # If it's over 40 meters and we have a gmaps client key,
            # look up the actual distance using google maps API
            if dist > 40 and gmaps is not None:
                p1pos = a.node[p1]['pll']
                p2pos = a.node[p2]['pll']
                dkey = '%s,%s,%s' % (p1pos, p2pos, gmapsmode)
                rkey = '%s,%s,%s' % (p2pos, p1pos, gmapsmode)
                dkey_dur = '%s_dur' % dkey
                rkey_dur = '%s_dur' % rkey

                if dkey in _gmap_cache_db and dkey_dur in _gmap_cache_db:
                    dist = _gmap_cache_db[dkey]
                    duration = _gmap_cache_db[dkey_dur]
                    logger.debug('%s -( %d )-> %s (Google/%s/cached)', a.node[p1]['name'],
                                 dist, a.node[p2]['name'], gmapsmode)
                elif rkey in _gmap_cache_db and rkey_dur in _gmap_cache_db:
                    dist = _gmap_cache_db[rkey]
                    duration = _gmap_cache_db[rkey_dur]
                    logger.debug('%s -( %d )-> %s (Google/%s/cached)', a.node[p1]['name'],
                                 dist, a.node[p2]['name'], gmapsmode)
                else:
                    # Perform the lookup
                    now = datetime.now()
                    gdir = gmaps.directions(p1pos, p2pos, mode=gmapsmode, departure_time=now)
                    dist = gdir[0]['legs'][0]['distance']['value']
                    duration = int(gdir[0]['legs'][0]['duration']['value']/60)
                    _gmap_cache_db[dkey] = dist
                    _gmap_cache_db[dkey_dur] = duration
                    logger.debug('%s -( %d )-> %s (Google/%s/lookup)', a.node[p1]['name'],
                                 dist, a.node[p2]['name'], gmapsmode)

            matrow.append(dist)
            matrow_dur.append(duration)

        _direct_dist_matrix.append(direct_matrow)
        _dist_matrix.append(matrow)
        _time_matrix.append(matrow_dur)


def get_portal_distance(p1, p2, direct=False):
    if active_graph is not None:
        p1 = active_graph.node[p1]['pos']
        p2 = active_graph.node[p2]['pos']
    if direct:
        return _direct_dist_matrix[p1][p2]
    return _dist_matrix[p1][p2]


def get_portal_time(p1, p2):
    if active_graph is not None:
        p1 = active_graph.node[p1]['pos']
        p2 = active_graph.node[p2]['pos']
    return int(_time_matrix[p1][p2])


def populate_graphs(portals, waypoints):
    global combined_graph
    global portal_graph
    global waypoint_graph
    global active_graph
    a = populate_graph(portals)
    # a graph with just portals
    portal_graph = a
    # Make a master graph that contains both portals and waypoints
    combined_graph = a.copy()
    if waypoints:
        waypoint_graph = populate_graph(waypoints)
        extend_graph_with_waypoints(combined_graph)


def extend_graph_with_waypoints(a):
    if waypoint_graph is None:
        return
    master_num = portal_graph.order()
    num = a.order()
    for i in range(waypoint_graph.order()):
        attrs = waypoint_graph.node[i]
        a.add_node(num, **attrs)
        a.node[num]['pos'] = master_num
        num += 1
        master_num += 1


def populate_graph(portals):
    a = nx.DiGraph()
    locs = []

    for num, row in enumerate(portals):
        a.add_node(num)
        a.node[num]['name'] = row[0]
        coord_parts = row[1].split(',')
        a.node[num]['pll'] = row[1]
        if len(row) > 2:
            a.node[num]['special'] = row[2]
        else:
            a.node[num]['special'] = None
        lat = int(float(coord_parts[0]) * 1.e6)
        lon = int(float(coord_parts[1]) * 1.e6)
        locs.append(np.array([lat, lon], dtype=float))

    n = a.order()
    locs = np.array(locs, dtype=float)

    # Convert coords to radians, then to cartesian, then to
    # gnomonic projection
    locs = geometry.e6LLtoRads(locs)
    xyz = geometry.radstoxyz(locs)
    xy = geometry.gnomonicProj(locs, xyz)

    for i in range(n):
        a.node[i]['pos'] = i
        a.node[i]['geo'] = locs[i]
        a.node[i]['xyz'] = xyz[i]
        a.node[i]['xy'] = xy[i]

    return a


def make_workplan(a, cooling, maxmu, minap, is_subset=False):
    global active_graph
    global _capture_cache

    linkplan = [None] * a.size()

    for p, q in a.edges():
        linkplan[a.edges[p, q]['order']] = (p, q, len(a.edges[p, q]['fields']))

    if minap:
        stats = get_workplan_stats(linkplan, cooling)
        if stats['ap'] < minap:
            logger.debug('Plan does not have enough AP, abandon early')
            return linkplan, stats

    # pre-optimize linkplan without the captures first
    linkplan, stats = improve_workplan(a, linkplan, cooling, maxmu)

    # Find the portals we need to capture
    seen_portals = list()
    captures_needed = list()
    for p, q, f in linkplan:
        if q not in seen_portals and q not in captures_needed:
            captures_needed.append(q)
        seen_portals.append(p)

    w_start = None
    w_end = None

    for i in range(a.order()):
        # skip non-special nodes
        if 'special' not in a.node[i]:
            continue
        # Is it a start enpoint?
        if a.node[i]['special'] == '_w_start':
            w_start = i
        elif a.node[i]['special'] == '_w_end':
            w_end = i

    if w_start is None:
        # Find the portal that's furthest away from the starting portal
        maxdist = None
        for p in captures_needed:
            dist = get_portal_distance(linkplan[0][0], p)
            if maxdist is None or dist > maxdist:
                w_start = p
                maxdist = dist

        logger.debug('Furthest from %s is %s', a.node[linkplan[0][0]]['name'], a.node[w_start]['name'])

    # Move w_start and w_end to start and end
    if w_start in captures_needed:
        captures_needed.remove(w_start)
    captures_needed.insert(0, w_start)
    if linkplan[0][0] in captures_needed:
        captures_needed.remove(linkplan[0][0])
    captures_needed.append(linkplan[0][0])

    logger.debug('captures_needed: %s', captures_needed)

    cachekey = list(captures_needed)
    if is_subset:
        subset_key = list()
        for n in range(a.order()):
            subset_key.append(a.node[n]['pos'])
        subset_key.sort()
        cachekey = cachekey + subset_key
    cachekey = tuple(cachekey)

    if cachekey not in _capture_cache:
        logger.debug('Capture cache miss, starting ortools calculation')
        mapping = list()
        or_dist_matrix = list()
        for pp in captures_needed:
            mapping.append(pp)
            or_dist_matrix.append(list())
            for pq in captures_needed:
                pdist = get_portal_distance(pp, pq)
                or_dist_matrix[len(mapping)-1].append(pdist)
        logger.debug('or_dist_matrix:\n%s', pformat(or_dist_matrix))

        manager = pywrapcp.RoutingIndexManager(len(captures_needed), 1, [0], [len(captures_needed)-1])
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return or_dist_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        )
        logger.debug('Starting solver')
        assignment = routing.SolveWithParameters(search_parameters)
        logger.debug('Ended solver')

        if not assignment:
            logger.debug('Could not solve for these constraints, ignoring plan')
            _capture_cache[cachekey] = None
            return None

        index = routing.Start(0)
        dist_ordered = list()
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            dist_ordered.append(mapping[node])
            index = assignment.Value(routing.NextVar(index))

        _capture_cache[cachekey] = dist_ordered
    else:
        logger.debug('Capture cache hit')
        if _capture_cache[cachekey] is None:
            logger.debug('Known unsolvable, ignoring')
            return None
        dist_ordered = _capture_cache[cachekey]

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

    workplan, stats = improve_workplan(a, workplan, cooling, maxmu)

    return workplan, stats


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


def get_workplan_stats(workplan, cooling='rhs'):
    workplan = remove_useless_captures(workplan)
    totalap = active_graph.order() * CAPTUREAP
    totaldist = 0
    totaltime = 0
    totalarea = 0
    traveltime = 0
    links = 0
    fields = 0

    try:
        totalarea = active_graph.totalarea
        need_area = False
    except AttributeError:
        need_area = True

    prev_p = None
    plan_at = 0
    for p, q, f in workplan:
        mp = combined_graph.node[p]['pos']
        plan_at += 1

        # Are we at a different location than the previous portal?
        if p != prev_p:
            # We are at a new portal, so add 1 minute just because
            # it takes time to get positioned and get to the right
            # screen in the UI
            totaltime += 1
            # How many keys do we need if/until we come back?
            ensurekeys = 0
            totalkeys = 0
            # Track when we leave this portal
            lastvisit = True
            same_p = True
            for fp, fq, ff in workplan[plan_at:]:
                if fp == p:
                    # Are we still at the same portal?
                    if same_p:
                        continue
                    if lastvisit:
                        lastvisit = False
                        ensurekeys = totalkeys
                else:
                    # we're at a different portal
                    same_p = False
                if fq == p:
                    # Future link to this portal
                    totalkeys += 1

            if prev_p is not None:
                duration = get_portal_time(prev_p, p)
                totaltime += duration
                traveltime += duration
                dist = get_portal_distance(prev_p, p)
                if dist > 40:
                    totaldist += dist

            # Are we at a blocker?
            if 'special' in combined_graph.node[mp] and combined_graph.node[mp]['special'] == '_w_blocker':
                # assume it takes 3 minutes to destroy a blocker
                totaltime += 3
                prev_p = p
                continue

            needkeys = 0
            if totalkeys:
                if lastvisit:
                    needkeys = totalkeys
                elif ensurekeys:
                    needkeys = ensurekeys

            # IDKFA means you already have all the keys
            if needkeys and cooling != 'idkfa':
                # We assume:
                # - we get roughly 1.5 keys per each hack
                # - we glyph-hack, meaning it takes about a minute per actual hack action
                needed_hacks = int((needkeys/1.5) + (needkeys % 1.5))
                # Hacking time
                totaltime += needed_hacks
                if cooling == 'none':
                    # uh-oh, no cooling?
                    totaltime += cooltime['none']*(needed_hacks-1)
                elif needed_hacks > 2:
                    # second hack is free regardless of the type of HS
                    totaltime += cooltime[cooling]*(needed_hacks-2)

            if lastvisit:
                # Add half a minute for putting on shields
                totaltime += 0.5

            prev_p = p

        if not q:
            continue

        # Add 15 seconds per link
        totaltime += 0.25
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

    stats = {
        'time': totaltime,
        'nicetime': str(timedelta(minutes=totaltime)),
        'traveltime': traveltime,
        'nicetraveltime': str(timedelta(minutes=traveltime)),
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


def workplan_is_better(orig_stats, new_stats, maxmu):
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


def improve_workplan(a, workplan, cooling, maxmu):
    a.orig_workplan = list(workplan)
    a.fixes = list()
    rcount = 0
    current_stats = get_workplan_stats(workplan, cooling)
    fielders_moved = False
    while True:
        rcount += 1
        logger.debug('Starting improve_workplan round %s', rcount)
        logger.debug('Current workplan:\n%s', pformat(workplan))
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
                # We don't touch links that create 2 fields,
                # because we cannot move or reverse them.
                if jf > 1:
                    continue
                if jp != p and jq != p:
                    continue
                if jp not in visited_origins or jq not in visited_origins:
                    continue
                # If previous origin and next origin are same, then we don't
                # need to do anything
                if m-j > 2 and workplan[j-1][0] == jp and workplan[j+1][0] == jp:
                    continue

                logger.debug('Improvement candidate: %s', workplan[j])
                # Move non-fielding edges to happen at capture stage, if that's better
                if jf == 0:
                    if p == jp:
                        # See if moving this edge will be better
                        nwp = list(workplan)
                        del(nwp[j])
                        if q is None:
                            del(nwp[i])
                            newpos = i
                        else:
                            newpos = i+1
                        nwp.insert(newpos, (jp, jq, jf))
                        new_stats = get_workplan_stats(nwp, cooling)
                        if workplan_is_better(current_stats, new_stats, maxmu):
                            # Replace current capture with this edge
                            a.fixes.append('R%s: Moved %s to %s' % (rcount, workplan[j], newpos))
                            logger.debug(a.fixes[-1])
                            workplan = nwp
                            current_stats = new_stats
                            improved = reordered = True
                            break
                    if p == jq and a.out_degree(jq) < 8:
                        # Reverse and move this edge to see if it's better
                        nwp = list(workplan)
                        del(nwp[j])
                        if q is None:
                            del(nwp[i])
                            newpos = i
                        else:
                            newpos = i+1
                        nwp.insert(newpos, (jq, jp, jf))
                        new_stats = get_workplan_stats(nwp, cooling)
                        if workplan_is_better(current_stats, new_stats, maxmu):
                            a.fixes.append('R%s: Reversed and moved %s to %s' % (rcount, workplan[j], newpos))
                            logger.debug(a.fixes[-1])
                            workplan = nwp
                            current_stats = new_stats
                            reverse_edge(jp, jq)
                            improved = reordered = True
                            break

                # This action creates one field, so we can't move it in the workplan
                elif jf == 1 and a.out_degree(jq) < 8:
                    # Try reversing this link in place to see if we get a better plan
                    nwp = list(workplan)
                    nwp[j] = (jq, jp, jf)
                    new_stats = get_workplan_stats(nwp, cooling)
                    if workplan_is_better(current_stats, new_stats, maxmu):
                        a.fixes.append('R%s: In-place reversed %s at %s' % (rcount, workplan[j], j))
                        logger.debug(a.fixes[-1])
                        reverse_edge(jp, jq)
                        workplan = nwp
                        current_stats = new_stats
                        fielders_moved = True
                        improved = True

            if reordered:
                break

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
    for i in range(a.size()):
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
    logger.debug('Final workplan:\n%s', pformat(workplan))
    stats = get_workplan_stats(workplan, cooling)
    logger.debug('Final stats:\n%s', pformat(stats))

    return workplan, stats


def remove_useless_captures(workplan):
    final = list()
    pos = 0
    for p, q, f in workplan:
        if q is None:
            useless = False
            # Are we going to visit this portal again before we link to it?
            for fp, fq, ff in workplan[pos+1:]:
                if fp == p:
                    # Yes, it's useless
                    logger.debug('Removing useless capture at pos %s: %s', pos, workplan[pos])
                    useless = True
                    break
                if fq == p:
                    # Yes, we'll link to it before we come back
                    break
            if useless:
                pos += 1
                continue
        final.append(workplan[pos])
        pos += 1
    return final


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


def make_subset(minportals, maxmu=False):
    global active_graph
    global _smallest_triangle
    global _largest_triangle

    if _smallest_triangle is None:
        # for smallest, we look for a triangle with the shortest perimeter
        # for largest, we look for a triangle with the largest area
        sperim = None
        larea = None
        active_graph = None
        for p1 in range(portal_graph.order()):
            for p2 in range(p1+1, portal_graph.order()):
                for p3 in range(p2+1, portal_graph.order()):
                    area = get_portals_area(p1, p2, p3)
                    perim = get_portals_perimeter(p1, p2, p3)
                    if larea is None or area > larea:
                        _largest_triangle = (p1, p2, p3)
                    if sperim is None or perim < sperim:
                        _smallest_triangle = (p1, p2, p3)

    if maxmu:
        subset = list(_largest_triangle)
    else:
        subset = list(_smallest_triangle)
    # Add portals until we get to minportals
    while len(subset) < minportals:
        add_subset_portal(subset, maxmu)
    return subset


def add_subset_portal(subset, maxmu=False):
    global _seen_subsets
    allp = list(range(portal_graph.order()))
    missing = [x for x in allp if x not in subset]
    if not missing:
        return
    if maxmu:
        maxtry = 0
        while True:
            candidate = np.random.choice(missing)
            subset.append(candidate)
            if maxtry > 10 or subset not in _seen_subsets:
                _seen_subsets.append(list(subset))
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
        attrs = portal_graph.node[num]
        b.add_node(ct, **attrs)
        ct += 1
    return b


def max_fields(a):
    n = a.order()
    # Generate a distance matrix for all portals
    pts = np.array([a.node[i]['xy'] for i in range(n)])
    perim = np.array(geometry.getPerim(pts))

    if not triangulate(a, perim):
        logger.debug('Could not triangulate')
        return False

    return True


def gen_cache_key(mode, maxmu, cooling, timelimit):
    plls = list()
    a = combined_graph
    for m in range(a.order()):
        plls.append(a.node[m]['pll'])
    h = hashlib.sha1()
    for pll in plls:
        h.update(pll.encode('utf-8'))
    phash = h.hexdigest()
    cachekey = mode
    if maxmu:
        cachekey += '+maxmu'
    if cooling != 'rhs':
        cachekey += '+%s' % cooling
    if timelimit:
        cachekey += '+timelimit-%s' % timelimit
    cachekey += '-%s' % phash

    return cachekey


def save_cache(bestgraph, bestplan, mode, maxmu, cooling, timelimit):
    # let's cache processing results for the same portals, just so
    # we can "add more cycles" to existing best plans
    # We use portal pll coordinates to generate the cache file key
    # and dump a in there.
    cachekey = gen_cache_key(mode, maxmu, cooling, timelimit)
    cachedir = get_cache_dir()
    plancachedir = os.path.join(cachedir, 'plans')
    Path(plancachedir).mkdir(parents=True, exist_ok=True)
    cachefile = os.path.join(plancachedir, cachekey)
    wc = shelve.open(cachefile, 'c')
    wc['bestplan'] = bestplan
    wc['bestgraph'] = bestgraph
    logger.info('Saved plan cache in %s', cachefile)
    wc.close()


def load_cache(mode, maxmu, cooling, timelimit):
    global active_graph

    cachekey = gen_cache_key(mode, maxmu, cooling, timelimit)
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
