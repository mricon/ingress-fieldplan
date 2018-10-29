#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from lib import geometry
import logging
import networkx as nx

from lib.Triangle import Triangle, Deadend

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

TRIES_PER_TRI = 10

logger = logging.getLogger('maxfield3')

# Use it to cache distances between portals
# so we don't continuously re-calculate them
_dist_cache = {}
_capture_cache = {}


def getPortalDistance(a, p1, p2):
    global _dist_cache
    if (p1, p2) in _dist_cache:
        return _dist_cache[(p1, p2)]
    # the reverse distance is the same
    if (p2, p1) in _dist_cache:
        return _dist_cache[(p2, p1)]

    p1pos = a.node[p1]['geo']
    p2pos = a.node[p2]['geo']
    dist = int(geometry.sphereDist(p1pos, p2pos)[0])
    _dist_cache[(p1, p2)] = dist
    return _dist_cache[(p1, p2)]


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

    # Starting with the first portal in the linkplan, make a chain
    # of closest portals not yet visited for the capture/keyhack plan
    startp = all_p.pop(0)
    if startp not in _capture_cache:
        dist_ordered = [startp]
        while True:
            if not len(all_p):
                break
            shortest_hop = None
            next_node = None
            for x in all_p:
                # calculate distance from current portal to next portal
                dist = getPortalDistance(a, dist_ordered[-1], x)
                if shortest_hop is None or dist < shortest_hop:
                    shortest_hop = dist
                    next_node = x

            dist_ordered.append(next_node)
            all_p.remove(next_node)

        dist_ordered.pop(0)
        dist_ordered.reverse()
        _capture_cache[startp] = dist_ordered
    else:
        dist_ordered = _capture_cache[startp]

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
                workplan.append((p, cp, False))
                linkplan.remove((p, cp, False))
                links_moved = True
        if not links_moved:
            # Just capturing, then
            workplan.append((p, None, False))

        p_captured.append(p)

    workplan.extend(linkplan)
    return workplan


def getWorkplanDist(a, workplan):
    prev_p = None
    totaldist = 0
    for p, q, f in workplan:
        # Are we at a different location than the previous portal?
        if p != prev_p:
            if prev_p is not None:
                dist = getPortalDistance(a, prev_p, p)
                if dist > 80:
                    totaldist += dist
            prev_p = p
    return totaldist


def improveEdgeOrder(a):
    m = a.size()
    # If link i is e then orderedEdges[i]=e
    orderedEdges = [-1] * m

    for p, q in a.edges():
        orderedEdges[a.edges[p, q]['order']] = (p, q, len(a.edges[p, q]['fields']) > 0)

    logger.debug('orderedEdges before improvement: %s', orderedEdges)
    for j in range(1, m):
        origin, q, f = orderedEdges[j]
        # Only move those that don't complete fields
        if f:
            continue

        # The first time this portal is used as an origin
        i = 0
        while orderedEdges[i][0] != origin:
            i += 1

        if i < j:
            logger.debug('moving %s before %s',  orderedEdges[j], orderedEdges[i])
            # Move link j to be just before link i
            orderedEdges =  (orderedEdges[   :i] +
                            [orderedEdges[  j  ]]+
                             orderedEdges[i  :j] +
                             orderedEdges[j+1: ])

    prev_origin = None
    prev_origin_created_fields = False
    o_starts = []
    z = None
    for i in range(m):
        p, q, f = orderedEdges[i]
        if prev_origin != p:
            # we moved to a new origin
            if z and not prev_origin_created_fields:
                # previous origin didn't create any fields, so move it
                # to happen right before the closest located portal
                # that we've already been to before
                closest_node_pos = 0
                if len(o_starts) > 2:
                    shortest_hop = None
                    curpos = a.node[prev_origin]['geo']
                    for o_seen, o_pos in o_starts[:-1]:
                        # calculate distance to this portal
                        dist = getPortalDistance(a, prev_origin, o_seen)
                        if shortest_hop is None or dist < shortest_hop:
                            shortest_hop = dist
                            closest_node_pos = o_pos

                logger.debug('moving %s before %s',  z, closest_node_pos)
                orderedEdges = (orderedEdges[:closest_node_pos] +
                                orderedEdges[z:i] +
                                orderedEdges[closest_node_pos:z] +
                                orderedEdges[i:])

            prev_origin = p
            o_starts.append((p,i))
            prev_origin_created_fields = False
            z = i

        # Only move those that don't complete fields
        if f:
            prev_origin_created_fields = True

    # avoid this stupid single-portal pingpong:
    #   portal_a -> foo
    #   portal_b -> portal_a
    #   portal_a -> bar
    # This should be optimized into:
    #   portal_a -> foo
    #   portal_a -> portal_b
    #   portal_a -> bar
    for i in range(1, m-1):
        p, q, f = orderedEdges[i]
        prev_origin = orderedEdges[i-1][0]
        if prev_origin != p:
            # we moved to a new origin
            next_origin = orderedEdges[i+1][0]
            if prev_origin == q and next_origin == prev_origin:
                logger.debug('fixed ping-pong %s->%s->%s', prev_origin, p, next_origin)
                # reverse this link
                attrs = a.edges[p, q]
                a.add_edge(q, p, **attrs)
                a.remove_edge(p, q)
                orderedEdges[i] = (q, p, f)

    logger.debug('orderedEdges after improvement: %s', orderedEdges)
    #    print
    for i in range(1, m):
        p = orderedEdges[i][0]
        prev_p = orderedEdges[i-1][0]
        if prev_p != p:
            dist = getPortalDistance(a, prev_p, p)
            logger.debug('Moved %s -> %s : %s', a.node[prev_p]['name'],
                         a.node[p]['name'], dist)


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
    # TODO: Move iterations here?
    n = a.order()

    pts = np.array([a.node[i]['xy'] for i in range(n)])
    perim = np.array(geometry.getPerim(pts))

    if not triangulate(a, perim):
        logger.debug('Could not triangulate')
        return False

    return True

