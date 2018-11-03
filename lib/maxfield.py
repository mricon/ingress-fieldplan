#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shelve

from lib import geometry
import logging
import networkx as nx

from datetime import datetime

from pathlib import Path

from lib.Triangle import Triangle, Deadend

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

TRIES_PER_TRI = 10

logger = logging.getLogger('maxfield3')

# Use it to cache distances between coordinates
# so we don't continuously re-calculate them.
# Especially useful when using Google Maps for
# true distances
_gmap_cache_db = None

_capture_cache = dict()
_dist_matrix = list()

# Stick a gmaps client here if we have a key
gmapsclient = None
gmapsmode = 'walking'


def genDistanceMatrix(a):
    global _dist_matrix
    global _gmap_cache_db

    n = a.order()
    if gmapsclient:
        logger.info('Generating the distance matrix using Google Maps API, may take a moment')
    else:
        logger.info('Generating the distance matrix')


    # We consider any direct distance shorter than 80m as effectively 0,
    # since the agent doesn't need to travel to access both portals.
    for p1 in range(n):
        matrow = list()
        for p2 in range(n):
            # Do direct distance first
            p1pos = a.node[p1]['geo']
            p2pos = a.node[p2]['geo']
            dist = int(geometry.sphereDist(p1pos, p2pos)[0])

            # If it's over 80 meters and we have a gmaps client key,
            # look up the actual distance using google maps API
            if dist > 80 and gmapsclient is not None:
                if _gmap_cache_db is None:
                    home = str(Path.home())
                    # Google Maps lookups are non-free, so cache them aggressively
                    # TODO: Invalidate these somehow after a period?
                    cacheloc = os.path.join(home, '.cache', 'ingress-fieldmap')
                    _gmap_cache_db = shelve.open(cacheloc, 'c')

                p1pos = a.node[p1]['pll']
                p2pos = a.node[p2]['pll']
                dkey = '%s,%s' % (p1pos, p2pos)
                rkey = '%s,%s' % (p2pos, p1pos)

                if dkey in _gmap_cache_db:
                    dist = _gmap_cache_db[dkey]
                    logger.debug('%s -( %d )-> %s (Google/%s/cached)', a.node[p1]['name'], dist, a.node[p2]['name'], gmapsmode)
                elif rkey in _gmap_cache_db:
                    dist = _gmap_cache_db[rkey]
                    logger.debug('%s -( %d )-> %s (Google/%s/cached)', a.node[p1]['name'], dist, a.node[p2]['name'], gmapsmode)
                else:
                    # Perform the lookup
                    now = datetime.now()
                    gdir = gmapsclient.directions(p1pos, p2pos, mode=gmapsmode, departure_time=now)
                    dist = gdir[0]['legs'][0]['distance']['value']
                    _gmap_cache_db[dkey] = dist
                    logger.debug('%s -( %d )-> %s (Google/%s/lookup)', a.node[p1]['name'], dist, a.node[p2]['name'], gmapsmode)
            else:
                logger.debug('%s -( %d )-> %s (Direct)', a.node[p1]['name'], dist, a.node[p2]['name'])

            matrow.append(dist)

        _dist_matrix.append(matrow)


def getPortalDistance(p1, p2):
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


def makeWorkPlan(a, ab=None):
    global _capture_cache

    # make a linkplan first
    linkplan = [None] * a.size()

    all_p = []
    for p, q in a.edges():
        linkplan[a.edges[p, q]['order']] = (p, q, len(a.edges[p, q]['fields']) > 0)
        if p not in all_p:
            all_p.append(p)
        if q not in all_p:
            all_p.append(q)

    # Add blockers we need to destroy
    if ab is not None:
        num = a.order()
        logger.debug('Adding %s blockers to the plan', ab.order())
        for i in range(ab.order()):
            num += 1
            attrs = ab.node[i]
            a.add_node(num, **attrs)
            a.node[num]['blocker'] = True
            all_p.append(num)

    startp = linkplan[0][0]
    if startp not in _capture_cache:
        # Find the portal that's furthest away from the starting portal
        maxdist = 0
        endp = startp
        for i in range(a.order()):
            dist = getPortalDistance(startp, i)
            if dist > maxdist:
                endp = i
                maxdist = dist

        routing = pywrapcp.RoutingModel(len(all_p), 1, [endp], [startp])

        search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
        routing.SetArcCostEvaluatorOfAllVehicles(getPortalDistance)
        assignment = routing.SolveWithParameters(search_parameters)
        dist_ordered = list()
        index = routing.Start(0)
        while not routing.IsEnd(index):
            dist_ordered.append(index)
            index = assignment.Value(routing.NextVar(index))

        dist_ordered.pop(-1)
        _capture_cache[startp] = dist_ordered
    else:
        dist_ordered = _capture_cache[startp]

    # This is the naive "find next closest, which often gives wonky
    # results. This is the classic "Traveling Salesman Problem" so
    # there are no perfect solutions. OR-Tools seems to offer better
    # planning than the naive implementation below.

    # Starting with the first portal in the linkplan, make a chain
    # of closest portals not yet visited for the capture/keyhack plan
    #if startp not in _capture_cache:
    #    dist_ordered = [startp]
    #    while True:
    #        if not len(all_p):
    #            break
    #        shortest_hop = None
    #        next_node = None
    #        for x in all_p:
    #            # calculate distance from current portal to next portal
    #            dist = getPortalDistance(dist_ordered[-1], x)
    #            if shortest_hop is None or dist < shortest_hop:
    #                shortest_hop = dist
    #                next_node = x
    #        dist_ordered.append(next_node)
    #        all_p.remove(next_node)
    #    dist_ordered.pop(0)
    #    dist_ordered.reverse()
    #    _capture_cache[startp] = dist_ordered
    #else:
    #    dist_ordered = _capture_cache[startp]

    a.captureplan = dist_ordered

    # Make a unified workplan
    workplan = []
    p_captured = []
    for p in dist_ordered:
        # Go through those already captured and
        # see if we can move any non-field-making
        # links in the linkplan to this position
        links_moved = False
        for cp in p_captured:
            if (p, cp, False) in linkplan:
                # Yes, found a link we can make early
                a.fixes.append('rpost: moved (%s, %s, False) into capture plan' % (p, cp))
                workplan.append((p, cp, False))
                linkplan.remove((p, cp, False))
                links_moved = True
        if not links_moved:
            # Just capturing, then
            workplan.append((p, None, False))

        p_captured.append(p)

    #p_captured.append(startp)

    #for p, q, f in linkplan:
    #    # Quick sanity check
    #    if q not in p_captured:
    #        logger.critical('Awooga, linking to a non-captured portal: %s->%s', p, q)
    #        sys.exit(1)

    workplan.extend(linkplan)
    return workplan


def getWorkplanDist(a, workplan):
    prev_p = None
    totaldist = 0
    for p, q, f in workplan:
        # Are we at a different location than the previous portal?
        if p != prev_p:
            if prev_p is not None:
                dist = getPortalDistance(prev_p, p)
                if dist > 80:
                    totaldist += dist
            prev_p = p
    return totaldist


def improveEdgeOrder(a):
    m = a.size()
    linkplan = [-1] * m

    for p, q in a.edges():
        linkplan[a.edges[p, q]['order']] = (p, q, len(a.edges[p, q]['fields']) > 0)
    # Stick original plan into a for debug purposes
    a.orig_linkplan = list(linkplan)
    a.fixes = list()
    rcount = 0

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

                a.fixes.append('r%d: moved above %s:' % (rcount, linkplan[closest_node_pos]))
                for row in linkplan[z:i]:
                    a.fixes.append('r%d:     %s' % (rcount, str(row)))
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

    # avoid this stupid single-portal pingpong:
    #   portal_a -> foo
    #   portal_b -> portal_a (or much closer than portal_b)
    #   portal_a -> bar
    # This should be optimized into:
    #   portal_a -> foo
    #   portal_a -> portal_b
    #   portal_a -> bar
    while True:
        improved = False
        rcount += 1

        for i in range(1, m-1):
            p, q, f = linkplan[i]
            prev_origin = linkplan[i-1][0]
            if prev_origin != p:
                # we moved to a new origin
                next_origin = linkplan[i+1][0]
                reverse_edge = False
                if prev_origin == q and next_origin == prev_origin:
                    reverse_edge = True
                    a.fixes.append('r%d: fixed exact ping-pong %s->%s->%s' % (rcount, prev_origin, p, next_origin))
                else:
                    dist_to_prev = getPortalDistance(prev_origin, p)
                    dist_to_next = getPortalDistance(p, next_origin)
                    dist_prev_to_next = getPortalDistance(prev_origin, next_origin)
                    if next_origin == q and (dist_to_prev+dist_to_next)/2 > dist_prev_to_next:
                        reverse_edge = True
                        a.fixes.append('r%d: fixed inefficient ping-pong %s->%s->%s' % (rcount, prev_origin, p, next_origin))

                if reverse_edge:
                    improved = True
                    # reverse this link
                    attrs = a.edges[p, q]
                    a.add_edge(q, p, **attrs)
                    a.remove_edge(p, q)
                    linkplan[i] = (q, p, f)

        if not improved:
            logger.debug('No further pingpong improvements found.')
            break

    # Stick linkplan into a for debug purposes
    a.linkplan = list(linkplan)

    # Record the new order of edges
    for i in range(m):
        p, q, f = linkplan[i]
        a.edges[p, q]['order'] = i


def removeSince(a,m,t):
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

