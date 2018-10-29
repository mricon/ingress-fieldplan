#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Konstantin Ryabitsev <icon@mricon.com>'

import sys
import os
import argparse
import networkx as nx
import numpy as np
# import pickle

from lib import gsheets, geometry, maxfield # PlanPrinterMap, geometry, agentOrder

import logging
logger = logging.getLogger(__name__)

# version number
_V_ = '3.0.0'
# max portals allowed
_MAX_PORTALS_ = 25


# noinspection PyUnresolvedReferences
def main():
    description = ('Ingress Maxfield - Maximize the number of links '
                   'and fields, and thus AP, for a collection of '
                   'portals in the game Ingress.')

    parser = argparse.ArgumentParser(description=description, prog='makePlan.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--google_token', default='token.json',
                        help='Path to the google token for manipulating Google Spreadsheets')
    parser.add_argument('-s', '--sheetid', default=None, required=True,
                        help='The Google Spreadsheet ID with portal definitions.')
    parser.add_argument('-i', '--iterations', type=int, default=10000,
                        help='Number of iterations to perform. More iterations may improve '
                        'results, but will take longer to process.')
    parser.add_argument('-k', '--maxkeys', type=int, default='6',
                        help='Maximum lacking keys per portal')
    #parser.add_argument('-f','--output_file',default='plan.pkl',
    #                    help="Filename for pickle object. Default: "
    #                    "plan.pkl")
    #parser.add_argument('-p', '--plots', action='store_true', default=False,
    #                    help='Generate graphs and plots')
    parser.add_argument('-m', '--travelmode', default='walking',
                        help='Travel mode (walking, bicycling, driving, transit).')
    parser.add_argument('-l', '--log', default=None,
                        help='Log file where to log processing info')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='Add debug information to the logfile')
    parser.add_argument('-q', '--quiet', action='store_true', default=False,
                        help='Only output errors to the stdout')
    args = parser.parse_args()

    #output_file = args["output_file"]
    #if output_file[-4:] != '.pkl':
    #    output_file += ".pkl"

    if args.iterations < 0:
        parser.error('Number of extra samples should be positive')

    global logger
    logger = logging.getLogger('maxfield3')
    logger.setLevel(logging.DEBUG)

    if args.log:
        ch = logging.FileHandler(args.log)
        formatter = logging.Formatter(
            '{%(module)s:%(funcName)s:%(lineno)s} %(message)s')
        ch.setFormatter(formatter)

        if args.debug:
            ch.setLevel(logging.DEBUG)
        else:
            ch.setLevel(logging.INFO)

        logger.addHandler(ch)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)

    if args.quiet:
        ch.setLevel(logging.CRITICAL)
    else:
        ch.setLevel(logging.INFO)

    logger.addHandler(ch)

    gs = gsheets.setup(args.google_token)

    portals, blockers = gsheets.get_portals_from_sheet(gs, args.sheetid)
    logger.info('Considering %d portals', len(portals))

    if len(portals) < 3:
        logger.critical('Must have more than 2 portals!')
        sys.exit(1)

    if len(portals) > _MAX_PORTALS_:
        logger.critical('Portal limit is %d', _MAX_PORTALS_)

    a = maxfield.populateGraph(portals)

    bestplan = None
    bestdist = np.inf
    bestkm = None
    counter = 0

    ab = None
    if blockers:
        ab = maxfield.populateGraph(blockers)

    logger.info('Finding the shortest plan with max %s lacking keys', args.maxkeys)
    logger.info('Ctrl-C to exit early')

    try:
        while counter < args.iterations:
            b = a.copy()
            counter += 1

            if not args.quiet:
                if bestkm is not None:
                    sys.stdout.write('\r(%0.2f km best): %s/%s      ' % (
                        bestkm, counter, args.iterations))
                    sys.stdout.flush()

            if not maxfield.maxFields(b):
                logger.debug('Could not find a triangulation')
                continue

            for t in b.triangulation:
                t.markEdgesWithFields()

            maxfield.improveEdgeOrder(b)

            # do any of the portals require more than maxkeys
            sane_key_reqs = True
            for i in range(len(b.node)):
                if b.in_degree(i) > args.maxkeys:
                    sane_key_reqs = False
                    break

            if not sane_key_reqs:
                logger.debug('Too many keys required, ignoring plan')
                continue

            sane_out_links = True
            for i in range(len(b.node)):
                if b.out_degree(i) > 8:
                    sane_out_links = False
                    break

            if not sane_out_links:
                logger.debug('Too many outgoing links, ignoring plan')
                continue

            workplan = maxfield.makeWorkPlan(b, ab)
            totaldist = maxfield.getWorkplanDist(b, workplan)

            if totaldist < bestdist:
                counter = 0
                bestplan = workplan
                bestdist = totaldist
                bestkm = bestdist/float(1000)

    except KeyboardInterrupt:
        print()
        print('Exiting loop')
        pass

    if not args.quiet:
        print()

    if bestplan is None:
        logger.critical('Could not find a solution for this list of portals.')
        sys.exit(1)

    gsheets.write_workplan(gs, args.sheetid, a, bestplan, args.travelmode)

if __name__ == "__main__":
    main()
