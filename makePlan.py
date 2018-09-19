#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ingress Maxfield - makePlan.py

usage: makePlan.py [-h] [-v] [-g] [-n NUM_AGENTS] [-s SAMPLES] [-d OUTPUT_DIR]
                   [-f OUTPUT_FILE]
                   input_file

Ingress Maxfield - Maximize the number of links and fields, and thus AP, for a
collection of portals in the game Ingress.

positional arguments:
  input_file            Input semi-colon delimited portal file

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -g, --google          Make maps with google maps API. Default: False
  -a, --api_key         Google API key for google maps. Default: None
  -n NUM_AGENTS, --num_agents NUM_AGENTS
                        Number of agents. Default: 1
  -s SAMPLES, --samples SAMPLES
                        Number of iterations to perform. More iterations may
                        improve results, but will take longer to process.
                        Default: 50
  -d OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Directory for results. Default: this directory
  -f OUTPUT_FILE, --output_file OUTPUT_FILE
                        Filename for pickle object. Default: plan.pkl

Original version by jpeterbaker
22 July 2014 - tvw updates csv file format
15 August 2014 - tvw updates with google API, adds -s,
                 switchted to ; delimited file
29 Sept 2014 - tvw V2.0 major update to new version
21 April 2015 - tvw V2.1 force data read to be string
"""

import sys
import os
import argparse
import networkx as nx
import numpy as np
import pandas as pd
from lib import maxfield,PlanPrinterMap,geometry,agentOrder
import pickle

import matplotlib.pyplot as plt

# version number
_V_ = '2.0.2'
# max portals allowed
_MAX_PORTALS_ = 50

def main():
    description=("Ingress Maxfield - Maximize the number of links "
                 "and fields, and thus AP, for a collection of "
                 "portals in the game Ingress.")
    parser = argparse.ArgumentParser(description=description,
                                     prog="makePlan.py")
    parser.add_argument('-v','--version',action='version',
                        version="Ingress Maxfield v{0}".format(_V_))
    parser.add_argument('-g','--google',action='store_true',
                        help='Make maps with google maps API. Default: False')
    parser.add_argument('-a','--api_key',default=None,
                        help='Google API key for Google maps. Default: None')
    parser.add_argument('-n','--num_agents',type=int,default='1',
                        help='Number of agents. Default: 1')
    parser.add_argument('-s','--samples',type=int,default=50,
                        help="Number of iterations to "
                        "perform. More iterations may improve "
                        "results, but will take longer to process. "
                        "Default: 50")
    parser.add_argument('input_file',
                        help="Input semi-colon delimited portal file")
    parser.add_argument('-d','--output_dir',default='',
                        help="Directory for results. Default: "
                        "this directory")
    parser.add_argument('-f','--output_file',default='plan.pkl',
                        help="Filename for pickle object. Default: "
                        "plan.pkl")
    parser.add_argument('-p','--plots',action='store_true',
                        default=False,
                        help="Generate graphs and plots. Default: %default")
    args = vars(parser.parse_args())

    # Number of iterations to complete since last improvement
    EXTRA_SAMPLES = args["samples"]

    GREEN = '#3BF256' # Actual faction text colors in the app
    BLUE  = '#2ABBFF'
    # Use google?
    useGoogle = args['google']
    api_key = args['api_key']

    output_directory = args["output_dir"]
    # add ending separator
    if output_directory[-1] != os.sep:
        output_directory += os.sep
    # create directory if doesn't exist
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    output_file = args["output_file"]
    if output_file[-4:] != '.pkl':
        output_file += ".pkl"

    nagents = args["num_agents"]
    if nagents < 0:
        sys.exit("Number of agents should be positive")

    EXTRA_SAMPLES = args["samples"]
    if EXTRA_SAMPLES < 0:
        sys.exit("Number of extra samples should be positive")
    elif EXTRA_SAMPLES > 10000:
        sys.exit("Extra samples may not be more than 10000")

    input_file = args['input_file']

    if input_file[-3:] != 'pkl':
        # If the input file is a portal list, let's set things up
        a = nx.DiGraph() # network tool
        locs = [] # portal coordinates
        # each line should be name;intel_link;keys
        portals = pd.read_table(input_file,sep=';',
                                comment='#',index_col=False,
                                names=['name','link','keys'],dtype=str)
        portals = np.array(portals)
        portals = np.array([portal for portal in portals if (isinstance(portal[0], basestring) and isinstance(portal[1], basestring))])
        print "Found {0} portals in portal list.".format(len(portals))
        if len(portals) < 3:
            sys.exit("Error: Must have more than 2 portals!")
        if len(portals) > _MAX_PORTALS_:
            sys.exit("Error: Portal limit is {0}".\
                     format(_MAX_PORTALS_))
        for num,portal in enumerate(portals):
            if len(portal) < 3:
                print "Error! Portal ",portal[0]," has a formatting problem."
                sys.exit()
            a.add_node(num)
            a.node[num]['name'] = portal[0]
            coords = (portal[1].split('pll='))
            if len(coords) < 2:
                print "Error! Portal ",portal[0]," has a formatting problem."
                sys.exit()
            coord_parts = coords[1].split(',')
            lat = int(float(coord_parts[0]) * 1.e6)
            lon = int(float(coord_parts[1]) * 1.e6)
            locs.append(np.array([lat,lon],dtype=float))
            try:
                keys = int(portal[2])
                a.node[num]['keys'] = keys
            except ValueError:
                a.node[num]['keys'] = 0

        n = a.order() # number of nodes
        locs = np.array(locs,dtype=float)

        # Convert coords to radians, then to cartesian, then to
        # gnomonic projection
        locs = geometry.e6LLtoRads(locs)
        xyz  = geometry.radstoxyz(locs)
        xy   = geometry.gnomonicProj(locs,xyz)

        for i in xrange(n):
            a.node[i]['geo'] = locs[i]
            a.node[i]['xyz'] = xyz[i]
            a.node[i]['xy' ] = xy[i]

        # EXTRA_SAMPLES attempts to get graph with few missing keys
        # Try to minimuze TK + 2*MK where
        # TK is the total number of missing keys
        # MK is the maximum number of missing keys for any single
        # portal
        bestgraph = None
        bestdist = np.inf
        sinceImprove = 0

        while sinceImprove<EXTRA_SAMPLES:
            b = a.copy()

            sinceImprove += 1

            if not maxfield.maxFields(b):
                print 'Randomization failure\nThe program may work if you try again. It is more likely to work if you remove some portals.'
                continue

            m = b.size()  # number of links
            agentOrder.improveEdgeOrder(b)
            orderedEdges = [None]*m
            for e in b.edges_iter():
                orderedEdges[b.edge[e[0]][e[1]]['order']] = e

            movements = agentOrder.getAgentOrder(b,nagents,orderedEdges)
            totaldist = 0
            for i in xrange(nagents):
                movie = movements[i]
                # first portal in first link
                curpos = b.node[orderedEdges[movie[0]][0]]['geo']
                for e in movie[1:]:
                    p,q = orderedEdges[e]
                    newpos = b.node[p]['geo']
                    dist = geometry.sphereDist(curpos,newpos)
                    totaldist += int(dist[0])
                    if totaldist > bestdist:
                        # no need to continue
                        break
                    curpos = newpos

            if totaldist < bestdist:
                sinceImprove = 0
                bestgraph = b
                bestdist  = totaldist
                bestkm = bestdist/float(1000)

            sys.stdout.write('\r(%0.2f km best): %s/%s      ' % (bestkm, sinceImprove, EXTRA_SAMPLES))
            sys.stdout.flush()

        print
        if bestgraph == None:
            print 'EXITING RANDOMIZATION LOOP WITHOUT SOLUTION!'
            print ''
            exit()

        a = bestgraph

        # Attach to each edge a list of fields that it completes
        # catch no triangulation (bad portal file?)
        try:
            for t in a.triangulation:
                t.markEdgesWithFields()
        except AttributeError:
            print "Error: problem with bestgraph... no triangulation...?"

        agentOrder.improveEdgeOrder(a)

        with open(output_directory+output_file,'w') as fout:
            pickle.dump(a,fout)
    else:
        with open(input_file,'r') as fin:
            a = pickle.load(fin)
    #    agentOrder.improveEdgeOrder(a)
    #    with open(output_directory+output_file,'w') as fout:
    #        pickle.dump(a,fout)

    PP = PlanPrinterMap.PlanPrinter(a,output_directory,nagents,useGoogle=useGoogle,
                                    api_key=api_key)
    PP.keyPrep()
    PP.agentKeys()
    PP.agentLinks()
    PP.makeODS()

    # These make step-by-step instructional images
    if args['plots']:
        PP.planMap(useGoogle=useGoogle)
        PP.animate(useGoogle=useGoogle)
        PP.split3instruct(useGoogle=useGoogle)

    print "Number of portals: {0}".format(PP.num_portals)
    print "Number of links: {0}".format(PP.num_links)
    print "Number of fields: {0}".format(PP.num_fields)
    portal_ap = (125*8 + 500 + 250)*PP.num_portals
    link_ap = 313 * PP.num_links
    field_ap = 1250 * PP.num_fields
    #print "AP from portals capture: {0}".format(portal_ap)
    print "AP from link creation: {0}".format(link_ap)
    print "AP from field creation: {0}".format(field_ap)
    print "Total AP: {0}".format(portal_ap+link_ap+field_ap)

if __name__ == "__main__":
    main()
