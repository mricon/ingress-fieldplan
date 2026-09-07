#!/usr/env python3
# -*- coding: utf-8 -*-

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import logging
logger = logging.getLogger('fieldplan')

class Deadend(Exception):
    def __init__(self, s):
        logger.debug('Hit a deadend: %s', s)
        self.explain = s


def try_ordered_edge(a,p,q,reversible):
    if a.has_edge(p,q) or a.has_edge(q,p):
        return

    if a.out_degree(p) >= 8:
        if not reversible:
            raise(Deadend('%s already has 8 outgoing' % p))
        if a.out_degree(q) >= 8:
            raise(Deadend('%s and %s already have 8 outgoing'%(p,q)))
        p, q = q, p

    try:
        stack = a.edgeStack
    except AttributeError:
        stack = a.edgeStack = []

    # Every edge in the graph goes through here and remove_since() pops the
    # stack in lockstep with remove_edge(), so len(stack) is the edge count.
    # (networkx's a.size() recomputes it by summing all degrees each call.)
    a.add_edge(p, q, order=len(stack), reversible=reversible, fields=[])
    stack.append((p, q))
    #logger.debug('adding p=%s, q=%s', p, q)
    #logger.debug('edgeStack follows')
    #logger.debug(a.edgeStack)


def graph_xyz(a):
    """n x 3 array of every node's xyz, built lazily and cached on the graph."""
    xyz = getattr(a, '_xyz', None)
    if xyz is None or len(xyz) != a.order():
        xyz = a._xyz = np.array([a.nodes[i]['xyz'] for i in range(a.order())])
    return xyz


class Triangle:
    def __init__(self, verts, a, exterior=False):
        """
        verts should be a 3-list of Portals
        verts[0] should be the final one used in linking
        exterior should be set to true if this triangle has no triangle parent
            the orientation of the outer edges of exterior Triangles do not matter
        """
        # If this portal is exterior, the final vertex doesn't matter
        self.verts = list(verts)
        self.a = a
        self.exterior = exterior

        if exterior:
            # Randomizing should help prevent perimeter nodes from getting too many links
            final = np.random.randint(3)
            tmp = self.verts[final]
            self.verts[final] = self.verts[0]
            self.verts[0] = tmp

        self.pts = np.array([a.nodes[p]['xyz'] for p in verts])
        # Normals of the three planes through the origin and each side.
        # They don't depend on the point being tested, so compute them once
        # here instead of on every containment check (see findContents).
        A = self.pts[[1, 2, 0]]
        B = self.pts[[2, 0, 1]]
        self.crosses = np.stack([A[:, 1]*B[:, 2] - A[:, 2]*B[:, 1],
                                 A[:, 2]*B[:, 0] - A[:, 0]*B[:, 2],
                                 A[:, 0]*B[:, 1] - A[:, 1]*B[:, 0]], axis=1)
        # Which side of each plane the opposite vertex lies on
        self.psign = np.sum(self.crosses * self.pts, 1)
        self.children = []
        self.contents = []
        self.center = None


    def findContents(self, candidates=None):
        if candidates is None:
            candidates = range(self.a.order())

        verts = self.verts
        cands = [p for p in candidates if p not in verts]
        if not cands:
            return

        # Same test as geometry.sphereTriContains, vectorized over all
        # candidates: a point is inside iff it is on the same side of all
        # three planes as the opposite vertex.
        xyz = graph_xyz(self.a)[cands]
        inside = np.all((xyz @ self.crosses.T) * self.psign > 0, axis=1)
        self.contents = [p for p, ok in zip(cands, inside) if ok]


    def randSplit(self):
        if not len(self.contents):
            return
        
        p = self.contents[np.random.randint(len(self.contents))]
        
        self.splitOn(p)

        for child in self.children:
            child.randSplit()


    def nearSplit(self):
        # Split on the node closest to final
        if len(self.contents) == 0:
            return

        contentPts = np.array([self.a.nodes[p]['pos'] for p in self.contents])
        displaces = contentPts - self.a.nodes[self.verts[0]]['pos']
        dists = np.sum(displaces**2,1)
        closest = np.argmin(dists)

        self.splitOn(self.contents[closest])

        for child in self.children:
            child.nearSplit()


    def splitOn(self,p):
        # 'opposite' is the child that does not share the final vertex
        # Because of the build order, it's safe for this triangle to believe it is exterior
        opposite  =  Triangle([self.verts[1], p,
                               self.verts[2]], self.a, True)
        # The other two children must also use my final as their final
        adjacents = [
                     Triangle([self.verts[0],
                               self.verts[2],p],self.a),
                     Triangle([self.verts[0],
                               self.verts[1],p],self.a)
                    ]
        
        self.children = [opposite]+adjacents
        self.center = p

        for child in self.children:
            child.findContents(self.contents)


    def tostr(self):
        # Just a string representation of the triangle
        return str([self.a.nodes[self.verts[i]]['name'] for i in range(3)])


    def buildFinal(self):
#        print 'building final',self.tostr()
        if self.exterior:
            # Avoid making the final the link origin when possible
#            print self.tostr(),'is exterior'
            try_ordered_edge(self.a,self.verts[1],
                               self.verts[0],self.exterior)
            try_ordered_edge(self.a,self.verts[2],
                               self.verts[0],self.exterior)
        else:
#            print self.tostr(),'is NOT exterior'
            try_ordered_edge(self.a,self.verts[0],
                               self.verts[1],self.exterior)
            try_ordered_edge(self.a,self.verts[0],
                               self.verts[2],self.exterior)

        if len(self.children) > 0:
            for i in [1,2]:
                self.children[i].buildFinal()


    def buildExceptFinal(self):
#        print 'building EXCEPT final',self.tostr()
        if len(self.children) == 0:
#            print 'no children'
            p,q = self.verts[2] , self.verts[1]
            try_ordered_edge(self.a,p,q,True)
            return

        # Child 0 is guaranteed to be the one opposite final
        self.children[0].buildGraph()

        for child in self.children[1:3]:
            child.buildExceptFinal()


    def buildGraph(self):
#        print 'building',self.tostr()
        # A first generation triangle could have its final vertex's edges already completed by neighbors. This will cause the first generation to be completed when the opposite edge is added which complicates  completing inside descendents. This could be solved by choosing a new final vertex (or carefully choosing the order of completion of first generation triangles).
        if (
                self.a.has_edge(self.verts[0],self.verts[1]) or
                self.a.has_edge(self.verts[1],self.verts[0])
               ) and (
                self.a.has_edge(self.verts[0],self.verts[2]) or
                self.a.has_edge(self.verts[2],self.verts[0])
               ):
            raise Deadend('Final vertex completed by neighbors')
        self.buildExceptFinal()
        self.buildFinal()


    # def contains(self,pt):
    #    return np.all(np.sum(self.orths*(pt-self.pts),1) < 0)


    # Attach to each edge a list of fields that it completes
    def markEdgesWithFields(self):
        edges = [(0,0)] * 3
        for i in range(3):
            p = self.verts[i-1]
            q = self.verts[i-2]
            if not self.a.has_edge(p,q):
                p,q = q,p
            # The graph should have been completed by now, so the edge p,q exists
            edges[i] = (p,q)
            logger.debug('adding edge %d: %s', i, (p, q))
            if not self.a.has_edge(p,q):
                logger.debug('a does NOT have edge p=%s, q=%s', p, q)
                logger.debug('there is a programming error')
                logger.debug('a only has the edges:')
                for p, q in self.a.edges_iter():
                    logger.debug(p, q)
                logger.debug('a has %s 1st gen triangles:', len(self.a.triangulation))
                for t in self.a.triangulation:
                    logger.debug(t.verts)

        edgeOrders = [self.a.edges[p, q]['order'] for p, q in edges]
        logger.debug('edgeOrders=%s', edgeOrders)

        lastInd = np.argmax(edgeOrders)
        # The edge that completes this triangle
        p, q = edges[lastInd]

        if self.verts not in self.a.edges[p, q]['fields']:
            self.a.edges[p, q]['fields'].append(self.verts)

        logger.debug('edge %s, fields: %s', (p, q), self.a.edges[p, q]['fields'])

        for child in self.children:
            child.markEdgesWithFields()


    def edgesByDepth(self,depth):
        # Return list of edges of triangles at given depth
        # 0 means edges of this very triangle
        # 1 means edges splitting this triangle
        # 2 means edges splitting this triangle's children 
        # etc.
        if depth == 0:
            return [ (self.verts[i],self.verts[i-1]) for i in range(3) ]
        if depth == 1:
            if self.center == None:
                return []
            return [ (self.verts[i],self.center) for i in range(3) ]
        return [e for child in self.children\
                  for e in child.edgesByDepth(depth-1)]

