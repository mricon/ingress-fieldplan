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

from random import shuffle
from datetime import timedelta

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

_master_graph = None
_current_graph = None

_capture_cache = dict()
_dist_matrix = list()
_time_matrix = list()
_direct_dist_matrix = list()
_area_cache = dict()

# in metres per minute, only used in the absence of Google Maps API
travel_speed = {
    'walking': 80,
    'bicycling': 300,
    'driving': 1000,
    'transit': 500,
}


def getCacheDir():
    home = str(Path.home())
    cachedir = os.path.join(home, '.cache', 'ingress-fieldmap')
    Path(cachedir).mkdir(parents=True, exist_ok=True)
    return cachedir


def genDistanceMatrix(a, ab, gmapskey=None, gmapsmode='walking'):
    global _dist_matrix
    global _direct_dist_matrix

    cachedir = getCacheDir()
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

    if ab is not None:
        num = a.order()
        logger.debug('Adding %s blockers to the matrix', ab.order())
        for i in range(ab.order()):
            attrs = ab.node[i]
            a.add_node(num, **attrs)
            a.node[num]['blocker'] = True
            num += 1

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


def getPortalDistance(p1, p2, direct=False):
    mp1 = _current_graph.node[p1]['pos']
    mp2 = _current_graph.node[p2]['pos']
    if direct:
        logger.debug('%s->%s=%s (direct)', _master_graph.node[mp1]['name'],
                     _master_graph.node[mp2]['name'], _direct_dist_matrix[mp1][mp2])
        return _direct_dist_matrix[mp1][mp2]
    logger.debug('%s->%s=%s (gmap)', _master_graph.node[mp1]['name'],
                 _master_graph.node[mp2]['name'], _dist_matrix[mp1][mp2])
    return _dist_matrix[mp1][mp2]


def getPortalTime(p1, p2):
    mp1 = _current_graph.node[p1]['pos']
    mp2 = _current_graph.node[p2]['pos']
    logger.debug('%s->%s=%s seconds (gmap)', _master_graph.node[mp1]['name'],
                 _master_graph.node[mp2]['name'], _time_matrix[mp1][mp2])
    return int(_time_matrix[mp1][mp2])


def populateGraph(portals):
    a = nx.DiGraph()
    locs = []

    for num, portal in enumerate(portals):
        a.add_node(num)
        a.node[num]['name'] = portal[0]
        coords = (portal[1].split('pll='))
        coord_parts = coords[1].split(',')
        a.node[num]['pll'] = '%s,%s' % (coord_parts[0], coord_parts[1])
        lat = int(float(coord_parts[0]) * 1.e6)
        lon = int(float(coord_parts[1]) * 1.e6)
        locs.append(np.array([lat, lon], dtype=float))

    n = a.order()
    locs = np.array(locs, dtype=float)

    # Convert coords to radians, then to cartesian, then to
    # gnomonic projection
    locs = geometry.e6LLtoRads(locs)
    xyz = geometry.radstoxyz(locs)
    xy = geometry.gnomonicProj(locs,xyz)

    for i in range(n):
        a.node[i]['pos'] = i
        a.node[i]['geo'] = locs[i]
        a.node[i]['xyz'] = xyz[i]
        a.node[i]['xy'] = xy[i]

    # Set it for distance/area calculation purposes
    global _master_graph
    global _current_graph
    _master_graph = a.copy()
    _current_graph = _master_graph

    return a


def makeLinkPlan(a):
    linkplan = [None] * a.size()

    for p, q in a.edges():
        linkplan[a.edges[p, q]['order']] = (p, q, len(a.edges[p, q]['fields']))

    linkplan = fixPingPong(a, linkplan)
    return linkplan


def makeWorkPlan(linkplan, a, ab=None, roundtrip=False, beginfirst=False, is_subset=False):
    global _capture_cache

    # make a linkplan first
    # Add blockers we need to destroy
    all_p = list(range(a.order()))
    if ab is not None:
        num = a.order()
        logger.debug('Adding %s blockers to the plan', ab.order())
        for i in range(ab.order()):
            attrs = ab.node[i]
            a.add_node(num, **attrs)
            a.node[num]['blocker'] = True
            all_p.append(num)
            num += 1

    logger.debug('roundtrip=%s', roundtrip)
    logger.debug('beginfirst=%s', beginfirst)

    startp = linkplan[0][0]
    cachekey = list()
    if is_subset:
        for n in range(a.order()):
            cachekey.append(a.node[n]['pos'])
    if beginfirst:
        endp = 0
        cachekey = tuple(cachekey + [a.node[startp]['pos']])
    elif roundtrip:
        endp = linkplan[-1][0]
        cachekey = tuple(cachekey + [a.node[startp]['pos'], a.node[endp]['pos']])
    else:
        cachekey = tuple(cachekey + [a.node[startp]['pos']])
        endp = startp

    if cachekey not in _capture_cache:
        if not (roundtrip or beginfirst):
            # Find the portal that's furthest away from the starting portal
            maxdist = getPortalDistance(startp, endp)
            for i in all_p:
                if i in (startp, endp):
                    continue
                dist = getPortalDistance(startp, i)
                if dist > maxdist:
                    endp = i
                    maxdist = dist

            logger.debug('Furthest from %s is %s', a.node[startp]['name'], a.node[endp]['name'])

        routing = pywrapcp.RoutingModel(len(all_p), 1, [endp], [startp])

        routing.SetArcCostEvaluatorOfAllVehicles(getPortalTime)
        search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        assignment = routing.SolveWithParameters(search_parameters)

        index = routing.Start(0)
        dist_ordered = list()
        while not routing.IsEnd(index):
            node = routing.IndexToNode(index)
            dist_ordered.append(node)
            index = assignment.Value(routing.NextVar(index))

        dist_ordered.append(routing.IndexToNode(index))
        dist_ordered.remove(startp)
        # Find clusters that are within 5 min travel of each-other
        # We shuffle them later in hopes that it gives us a more efficient
        # capture/link sequence.
        clusters = list()
        cluster = [dist_ordered[0]]
        for p in dist_ordered[1:]:
            if getPortalTime(cluster[0], p) <= 5:
                cluster.append(p)
                continue
            clusters.append(cluster)
            cluster = [p]

        clusters.append(cluster)
        _capture_cache[cachekey] = clusters
    else:
        clusters = _capture_cache[cachekey]

    a.clusters = clusters
    dist_ordered = list()
    for cluster in clusters:
        if len(cluster) > 1:
            shuffle(cluster)
        dist_ordered.extend(cluster)

    a.captureplan = dist_ordered

    # Make a unified workplan
    workplan = []
    p_captured = []
    for p in dist_ordered:
        req_capture = True
        for lp, lq, lf in linkplan:
            # if we see p show up in lp before it shows up in lq,
            # then it's a useless capture
            if lq == p:
                # We're making a link to it before we visit it, so
                # keep it in the capture plan
                break
            if lp == p:
                # We're coming back to it before linking to it, so don't
                # capture it separately
                a.fixes.append('rpost: removed useless capture of %s before (%s, %s, %s)' % (
                    a.node[p]['name'], lp, lq, lf))
                req_capture = False
                break

        if req_capture:
            # Go through those already captured and
            # see if we can move any non-field-making
            # links in the linkplan to this position
            links_moved = False
            for cp in p_captured:
                if (p, cp, 0) not in linkplan:
                    continue
                ploc = linkplan.index((p, cp, 0))
                # don't move if we make any fields during our visit
                # to that portal, to keep linking operations bunched
                # up together.
                fields_made = False
                for nloc in range(ploc+1, len(linkplan)):
                    if linkplan[nloc][0] != p:
                        # moved to a different origin
                        break
                    if linkplan[nloc][2] > 0:
                        # making a field, don't move this link
                        fields_made = True
                        break

                if not fields_made:
                    # Yes, found a link we can make early
                    a.fixes.append('rpost: moved (%s, %s, 0) into capture plan' % (p, cp))
                    workplan.append((p, cp, 0))
                    linkplan.remove((p, cp, 0))
                    links_moved = True

            if not links_moved:
                # Just capturing, then
                workplan.append((p, None, 0))

            p_captured.append(p)

        elif beginfirst and not len(workplan):
            # Force capture of first portal anyway when beginfirst
            workplan.append((p, None, 0))
            p_captured.append(p)

    workplan.extend(linkplan)
    workplan = fixPingPong(a, workplan)
    if beginfirst and roundtrip:
        if workplan[-1][0] != workplan[0][0]:
            logger.debug('Appending start portal to the end of plan for beginfirst+roundtrip')
            workplan.append((workplan[0][0], None, 0))
    return workplan


def getWorkplanStats(a, workplan, cooling='rhs'):
    totalap = a.order() * CAPTUREAP
    totaldist = 0
    totalarea = 0
    totaltime = 0
    traveltime = 0
    links = 0
    fields = 0

    prev_p = None
    plan_at = 0
    for p, q, f in workplan:
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
                duration = getPortalTime(prev_p, p)
                totaltime += duration
                traveltime += duration
                dist = getPortalDistance(prev_p, p)
                if dist > 40:
                    totaldist += dist

            # Are we at a blocker?
            if 'blocker' in a.node[p]:
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
                # - we glyph-hack, meaning it takes about half minute per actual hack action
                needed_hacks = int((needkeys/1.5) + (needkeys % 1.5))
                # Hacking time
                totaltime += needed_hacks/2
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
        for t in a.edges[p, q]['fields']:
            s1 = getPortalDistance(t[0], t[1], direct=True)
            s2 = getPortalDistance(t[1], t[2], direct=True)
            s3 = getPortalDistance(t[0], t[2], direct=True)
            # Hero's formula for triangle area
            s = (s1 + s2 + s3)/2
            try:
                area = int(np.sqrt(s * (s - s1) * (s - s2) * (s - s3)))
            except ValueError:
                # Effectively, 0
                area = 0
            logger.debug('Field %s-%s-%s, area: %s sqm', t[0], t[1], t[2], area)
            totalarea += area

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
    }

    return stats


def fixPingPong(a, workplan):
    # avoid this stupid single-portal pingpong:
    #   at portal_a
    #   portal_b -> portal_a
    #   at portal_x
    # This should be optimized into:
    #   at portal_a
    #   portal_a -> portal_b
    #   at portal_x
    #
    # Similarly:
    #   at portal_x
    #   portal_a -> portal_b
    #   at portal_b
    # should become:
    #   at portal_x
    #   portal_b -> portal_a
    #   at portal_b
    rcount = 0
    while True:
        improved = False
        rcount += 1
        seen = [workplan[0][0]]

        for i in range(1, len(workplan)-1):
            p, q, f = workplan[i]

            prev_origin = workplan[i-1][0]
            if prev_origin != p:
                if prev_origin not in seen:
                    seen.append(prev_origin)

                if p not in seen:
                    # Don't consider portals we're still capturing
                    continue

                # we moved to a new origin
                # skip if next step makes no links
                if workplan[i+1][1] is None:
                    continue

                next_origin = workplan[i+1][0]
                reverse_edge = False
                if (next_origin == q or prev_origin == q) and a.out_degree(q) < 8:
                    reverse_edge = True
                    a.fixes.append('r%d: fixed ping-pong %s->%s->%s' % (rcount, prev_origin, p, next_origin))

                # if prev_origin == q and next_origin == prev_origin:
                #     reverse_edge = True
                #     a.fixes.append('r%d: fixed exact ping-pong %s->%s->%s' % (rcount, prev_origin, p, next_origin))
                # Turn off distance-based pingpong fixes for now
                # we are already catching them by the blunt logic above
                # else:
                #    dist_to_prev = getPortalDistance(prev_origin, p)
                #    dist_to_next = getPortalDistance(p, next_origin)
                #    dist_prev_to_next = getPortalDistance(prev_origin, next_origin)
                #    if next_origin == q and (dist_to_prev+dist_to_next)/2 > dist_prev_to_next:
                #        reverse_edge = True
                #        a.fixes.append('r%d: fixed inefficient ping-pong %s->%s->%s' % (rcount, prev_origin, p, next_origin))

                if reverse_edge:
                    improved = True
                    # reverse this link
                    attrs = a.edges[p, q]
                    a.add_edge(q, p, **attrs)
                    a.remove_edge(p, q)
                    workplan[i] = (q, p, f)

        if not improved:
            logger.debug('No further pingpong improvements found.')
            break

    return workplan


def improveEdgeOrder(a):
    m = a.size()
    linkplan = [-1] * m

    for p, q in a.edges():
        linkplan[a.edges[p, q]['order']] = (p, q, len(a.edges[p, q]['fields']))
    # Stick original plan into a for debug purposes
    a.orig_linkplan = list(linkplan)
    a.fixes = list()

    prev_origin = None
    fielders_moved = False
    seen_origins = list()
    prev_origin_created_fields = False
    z = None
    # This moves non-fielding origins closer to other portals
    for i in range(m):
        p, q, f = linkplan[i]
        # If both the origin and the target are in seen_origins, then
        # flipping the link will probably give us a more efficient
        # fielding plan
        reverse_edge = False
        if p in seen_origins and q in seen_origins and f < 2:
            reverse_edge = True

        # If we haven't visited this target yet, but will in the future,
        # we're better off reversing the link
        if q not in seen_origins and m - i > 1 and f < 2:
            for j in range(i+1, m):
                if linkplan[j][0] == q:
                    reverse_edge = True
                    break

        if reverse_edge and a.out_degree(q) < 8:
            attrs = a.edges[p, q]
            a.add_edge(q, p, **attrs)
            a.remove_edge(p, q)
            linkplan[i] = (q, p, f)
            a.fixes.append('reversed %s->%s:' % (p, q))
            if f:
                fielders_moved = True
            # Send us for another loop on this
            i -= 1
            continue

        if prev_origin != p:
            # we moved to a new origin
            if p not in seen_origins:
                seen_origins.append(p)

            # If the target is in seen_origins, then we flip the link
            # around and move it to happen
            if z and not prev_origin_created_fields:
                # previous origin didn't create any fields, so move it
                # to happen right before the same (or closest) portal
                # that we've already been to before
                closest_node_pos = 0
                shortest_hop = None
                for j in range(0, z):
                    if linkplan[j][0] == prev_origin:
                        # Found exact match
                        closest_node_pos = j
                        break

                    dist = getPortalDistance(linkplan[j][0], prev_origin)
                    if shortest_hop is None or dist <= shortest_hop:
                        shortest_hop = dist
                        closest_node_pos = j

                if closest_node_pos < 0:
                    closest_node_pos = 0

                a.fixes.append('moved above %s:' % str(linkplan[closest_node_pos]))
                for row in linkplan[z:i]:
                    a.fixes.append('     %s' % str(row))
                linkplan = (linkplan[:closest_node_pos] +
                            linkplan[z:i] +
                            linkplan[closest_node_pos:z] +
                            linkplan[i:])

            prev_origin = p
            prev_origin_created_fields = False
            z = i

        # Only move those that don't complete fields
        if f:
            prev_origin_created_fields = True

    # Record the new order of edges
    for i in range(m):
        p, q, f = linkplan[i]
        a.edges[p, q]['order'] = i
        if fielders_moved:
            a.edges[p, q]['fields'] = list()

    if fielders_moved:
        # Recalculate fields
        for t in a.triangulation:
            t.markEdgesWithFields()

    # Stick linkplan into a for debugging purposes
    a.linkplan = linkplan


def removeSince(a, m, t):
    # Remove all but the first m edges from a (and .edge_stck)
    # Remove all but the first t Triangules from a.triangulation
    for i in range(len(a.edgeStack) - m):
        p,q = a.edgeStack.pop()
        a.remove_edge(p,q)
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
        startStackLen = len(a.edgeStack)
    except AttributeError:
        startStackLen = 0
        a.edgeStack = []
    try:
        startTriLen = len(a.triangulation)
    except AttributeError:
        startTriLen = 0
        a.triangulation = []

    # Try all triangles using perim[0:2] and another perim node
    for i in np.random.permutation(range(2, pn)):

        for j in range(TRIES_PER_TRI):
            t0 = Triangle(perim[[0,1,i]], a, True)
            t0.findContents()
            t0.randSplit()
            try:
                t0.buildGraph()
            except Deadend as d:
                # remove the links formed since beginning of loop
                removeSince(a,startStackLen, startTriLen)
            else:
                # This build was successful. Break from the loop
                break
        else:
            # The loop ended "normally" so this triangle failed
            continue

        if not triangulate(a,perim[range(1,i   +1   )]): # 1 through i
            # remove the links formed since beginning of loop
            removeSince(a,startStackLen, startTriLen)
            continue

        if not triangulate(a,perim[range(0,i-pn-1,-1)]): # i through 0
           # remove the links formed since beginning of loop
           removeSince(a,startStackLen,startTriLen)
           continue

        # This will be a list of the first generation triangles
        a.triangulation.append(t0)

        # This triangle and the ones to its sides succeeded
        logger.debug('Succeeded with perim=%s', perim)
        return True

    # Could not find a solution
    logger.debug('Failed with perim=%s', perim)
    return False


def maxFields(a):
    n = a.order()
    # Generate a distance matrix for all portals
    pts = np.array([a.node[i]['xy'] for i in range(n)])
    perim = np.array(geometry.getPerim(pts))

    if not triangulate(a, perim):
        logger.debug('Could not triangulate')
        return False

    return True


def genCacheKey(a, ab, mode, beginfirst, roundtrip, maxmu, timelimit):
    plls = list()
    for m in range(a.order()):
        plls.append(a.node[m]['pll'])
    if ab is not None:
        for m in range(ab.order()):
            if ab.node[m]['pll'] not in plls:
                plls.append(ab.node[m]['pll'])
    h = hashlib.sha1()
    for pll in plls:
        h.update(pll.encode('utf-8'))
    phash = h.hexdigest()
    cachekey = mode
    if beginfirst:
        cachekey += '+beginfirst'
    if roundtrip:
        cachekey += '+roundtrip'
    if maxmu:
        cachekey += '+maxmu'
    if timelimit:
        cachekey += '+timelimit-%s' % timelimit
    cachekey += '-%s' % phash

    return cachekey


def saveCache(a, ab, bestplan, mode, beginfirst, roundtrip, maxmu, timelimit):
    # let's cache processing results for the same portals, just so
    # we can "add more cycles" to existing best plans
    # We use portal pll coordinates to generate the cache file key
    # and dump a in there.
    cachekey = genCacheKey(a, ab, mode, beginfirst, roundtrip, maxmu, timelimit)
    cachedir = getCacheDir()
    plancachedir = os.path.join(cachedir, 'plans')
    Path(plancachedir).mkdir(parents=True, exist_ok=True)
    cachefile = os.path.join(plancachedir, cachekey)
    wc = shelve.open(cachefile, 'c')
    wc['bestplan'] = bestplan
    wc['bestgraph'] = a
    logger.info('Saved plan cache in %s', cachefile)
    wc.close()


def loadCache(a, ab, mode, beginfirst, roundtrip, maxmu, timelimit):
    cachekey = genCacheKey(a, ab, mode, beginfirst, roundtrip, maxmu, timelimit)
    cachedir = getCacheDir()
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

    return bestgraph, bestplan


