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

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

TRIES_PER_TRI = 10

logger = logging.getLogger('fieldplan')

_capture_cache = dict()
_dist_matrix = list()


def getCacheDir():
    home = str(Path.home())
    cachedir = os.path.join(home, '.cache', 'ingress-fieldmap')
    Path(cachedir).mkdir(parents=True, exist_ok=True)
    return cachedir


def genDistanceMatrix(a, ab, gmapskey=None, gmapsmode='walking'):
    global _dist_matrix

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
        for p2 in range(n):
            # Do direct distance first
            p1pos = a.node[p1]['geo']
            p2pos = a.node[p2]['geo']
            dist = int(geometry.sphereDist(p1pos, p2pos)[0])

            # If it's over 40 meters and we have a gmaps client key,
            # look up the actual distance using google maps API
            if dist > 40 and gmaps is not None:
                p1pos = a.node[p1]['pll']
                p2pos = a.node[p2]['pll']
                dkey = '%s,%s,%s' % (p1pos, p2pos, gmapsmode)
                rkey = '%s,%s,%s' % (p2pos, p1pos, gmapsmode)

                if dkey in _gmap_cache_db:
                    dist = _gmap_cache_db[dkey]
                    logger.debug('%s -( %d )-> %s (Google/%s/cached)', a.node[p1]['name'], dist, a.node[p2]['name'], gmapsmode)
                elif rkey in _gmap_cache_db:
                    dist = _gmap_cache_db[rkey]
                    logger.debug('%s -( %d )-> %s (Google/%s/cached)', a.node[p1]['name'], dist, a.node[p2]['name'], gmapsmode)
                else:
                    # Perform the lookup
                    now = datetime.now()
                    gdir = gmaps.directions(p1pos, p2pos, mode=gmapsmode, departure_time=now)
                    dist = gdir[0]['legs'][0]['distance']['value']
                    _gmap_cache_db[dkey] = dist
                    logger.debug('%s -( %d )-> %s (Google/%s/lookup)', a.node[p1]['name'], dist, a.node[p2]['name'], gmapsmode)
            else:
                logger.debug('%s -( %d )-> %s (Direct)', a.node[p1]['name'], dist, a.node[p2]['name'])

            matrow.append(dist)

        _dist_matrix.append(matrow)


def getPortalDistance(p1, p2):
    logger.debug('p1=%s, p2=%s', p1, p2)
    return _dist_matrix[p1][p2]


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
        a.node[i]['geo'] = locs[i]
        a.node[i]['xyz'] = xyz[i]
        a.node[i]['xy' ] = xy[i]

    return a


def makeWorkPlan(a, ab=None, roundtrip=False, beginfirst=False):
    global _capture_cache

    # make a linkplan first
    linkplan = [None] * a.size()

    for p, q in a.edges():
        linkplan[a.edges[p, q]['order']] = (p, q, len(a.edges[p, q]['fields']))

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
    if beginfirst:
        endp = 0
        cachekey = startp
    elif roundtrip:
        linkplan = fixPingPong(a, linkplan)
        endp = linkplan[-1][0]
        cachekey = (startp, endp)
    else:
        cachekey = startp
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

        routing.SetArcCostEvaluatorOfAllVehicles(getPortalDistance)
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
        # Find clusters that are within 160m of each-other
        # We shuffle them later in hopes that it gives us a more efficient
        # capture/link sequence.
        clusters = list()
        cluster = [dist_ordered[0]]
        for p in dist_ordered[1:]:
            if getPortalDistance(cluster[0], p) <= 160:
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


def getWorkplanDist(a, workplan):
    prev_p = None
    totaldist = 0
    for p, q, f in workplan:
        # Are we at a different location than the previous portal?
        if p != prev_p:
            if prev_p is not None:
                dist = getPortalDistance(prev_p, p)
                if dist > 40:
                    totaldist += dist
            prev_p = p
    return totaldist


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
    prev_origin_created_fields = False
    z = None
    # This moves non-fielding origins closer to other portals
    for i in range(m):
        p, q, f = linkplan[i]
        if prev_origin != p:
            # we moved to a new origin
            if z and not prev_origin_created_fields:
                # previous origin didn't create any fields, so move it
                # to happen right before the same (or closest) portal
                # that we've already been to before
                closest_node_pos = 0
                shortest_hop = None
                for j in range(z-1, -1, -1):
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

    # Stick linkplan into a for debugging purposes
    a.linkplan = linkplan

    # Record the new order of edges
    for i in range(m):
        p, q, f = linkplan[i]
        a.edges[p, q]['order'] = i


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


def genCacheKey(a, ab, mode, beginfirst, roundtrip):
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
    cachekey += '-%s' % phash

    return cachekey


def saveCache(a, ab, bestplan, bestdist, mode, beginfirst, roundtrip):
    # let's cache processing results for the same portals, just so
    # we can "add more cycles" to existing best plans
    # We use portal pll coordinates to generate the cache file key
    # and dump a in there.
    cachekey = genCacheKey(a, ab, mode, beginfirst, roundtrip)
    cachedir = getCacheDir()
    plancachedir = os.path.join(cachedir, 'plans')
    Path(plancachedir).mkdir(parents=True, exist_ok=True)
    cachefile = os.path.join(plancachedir, cachekey)
    wc = shelve.open(cachefile, 'c')
    wc['bestplan'] = bestplan
    wc['bestgraph'] = a
    wc['dist_matrix'] = _dist_matrix
    wc['bestdist'] = bestdist
    logger.info('Saved plan cache in %s', cachefile)
    wc.close()


def loadCache(a, ab, mode, beginfirst, roundtrip):
    cachekey = genCacheKey(a, ab, mode, beginfirst, roundtrip)
    cachedir = getCacheDir()
    plancachedir = os.path.join(cachedir, 'plans')
    cachefile = os.path.join(plancachedir, cachekey)
    bestgraph = None
    bestplan = None
    bestdist = np.inf
    try:
        global _dist_matrix
        wc = shelve.open(cachefile, 'r')
        logger.info('Loading cache data from cache %s', cachefile)
        _dist_matrix = wc['dist_matrix']
        bestgraph = wc['bestgraph']
        bestplan = wc['bestplan']
        bestdist = wc['bestdist']
        wc.close()
    except:
        pass
    return (bestgraph, bestplan, bestdist)


