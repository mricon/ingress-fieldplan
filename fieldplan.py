#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse

from lib import gsheets, maxfield, animate

import logging
import multiprocessing as mp
import time

import networkx as nx
import random

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

logger = logging.getLogger('fieldplan')

# version number
_V_ = '3.2.0'


def queue_job(args, ready_queue):
    # When limiting the number of actions, we use this
    # counter to see how many portals out of the total
    # number provided we need to consider.
    maxportals = maxfield.portal_graph.order()
    p_considered = maxportals
    is_subset = False
    # We just crank out plans until we are terminated
    while True:
        if ready_queue.qsize() > 10:
            # Queue is getting full, so do an idle loop
            time.sleep(0.1)
            continue
        if p_considered == maxportals and args.minportals is None:
            b = maxfield.portal_graph.copy()
        else:
            b = nx.DiGraph()
            ct = 0
            is_subset = True
            if args.minportals is not None:
                # Play with the number of portals to consider to arrive at the best
                # MU per distance travelled number. Randomly select a number of portals
                # between 3 and p_considered
                subset = random.sample(range(maxportals),
                                       random.randint(args.minportals, p_considered))
            else:
                subset = random.sample(range(maxportals), p_considered)

            subset.sort()
            if args.beginfirst:
                subset[0] = 0
            for num in subset:
                attrs = maxfield.portal_graph.node[num]
                b.add_node(ct, **attrs)
                ct += 1

        success = maxfield.maxFields(b)
        if success and args.maxkeys:
            # do any of the portals require more than maxkeys
            sane_key_reqs = True
            for i in range(b.order()):
                if b.in_degree(i) > args.maxkeys:
                    sane_key_reqs = False
                    break

            if not sane_key_reqs:
                success = False
        if not success:
            ready_queue.put((False, None, None, None))
            continue

        for t in b.triangulation:
            t.markEdgesWithFields()

        maxfield.extendGraphWithWaypoints(b)
        maxfield.active_graph = b

        maxfield.improveEdgeOrder(b)
        linkplan = maxfield.makeLinkPlan(b)
        workplan = maxfield.makeWorkPlan(b, linkplan, is_subset)

        if workplan is None:
            ready_queue.put((False, None, None, None))
            continue

        stats = maxfield.getWorkplanStats(b, workplan, cooling=args.cooling)

        if args.maxtime is not None:
            if stats['time'] > args.maxtime:
                p_considered -= 1
                if p_considered < 3:
                    p_considered = 3
                ready_queue.put((False, None, None, None))
                continue
            elif p_considered < maxportals:
                p_considered += 1
            if args.mintime is not None and stats['time'] < args.mintime:
                ready_queue.put((False, None, None, None))
                continue

        ready_queue.put((success, b, workplan, stats))


# noinspection PyUnresolvedReferences
def main():
    description = ('Ingress FieldPlan - Maximize the number of links '
                   'and fields, and thus AP, for a collection of '
                   'portals in the game Ingress and create a convenient plan '
                   'in Google Spreadsheets. Spin-off from Maxfield.')

    parser = argparse.ArgumentParser(description=description, prog='makePlan.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--iterations', type=int, default=10000,
                        help='Number of iterations to perform. More iterations may improve '
                        'results, but will take longer to process.')
    parser.add_argument('-k', '--maxkeys', type=int, default=None,
                        help='Limit number of keys required per portal '
                        '(may result in less efficient plans).')
    parser.add_argument('-m', '--travelmode', default='walking',
                        help='Travel mode (walking, bicycling, driving, transit).')
    parser.add_argument('-s', '--sheetid', default=None, required=True,
                        help='The Google Spreadsheet ID with portal definitions.')
    parser.add_argument('-n', '--nosave', action='store_true', default=False,
                        help='Do not attempt to save the spreadsheet, just calculate the plan.')
    parser.add_argument('-p', '--plots', default=None,
                        help='Save step-by-step PNGs of the workplan into this directory.')
    parser.add_argument('--plotdpi', default=96, type=int,
                        help='DPI to use for generating plots (try 144 for high-dpi screens)')
    parser.add_argument('-g', '--gmapskey', default=None,
                        help='Google Maps API key (for better distances)')
    parser.add_argument('-f', '--faction', default='enl',
                        help='Set to "res" to use resistance colours')
    parser.add_argument('-c', '--cooling', default='rhs',
                        help='What kind of heatsinks to assume (hs, rhs, vrhs, none, idkfa)')
    parser.add_argument('-u', '--maxmu', action='store_true', default=False,
                        help='Find a plan with highest MU coverage instead of best AP')
    parser.add_argument('-t', '--maxtime', default=None, type=int,
                        help='Ignore plans that would take longer than this (in minutes)')
    parser.add_argument('--mintime', default=None, type=int,
                        help='Ignore plans that would take less time than this (in minutes)')
    parser.add_argument('--maxcpus', default=mp.cpu_count(), type=int,
                        help='Maximum number of cpus to use')
    parser.add_argument('-l', '--log', default=None,
                        help='Log file where to log processing info')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='Add debug information to the logfile')
    parser.add_argument('-q', '--quiet', action='store_true', default=False,
                        help='Only output errors to the stdout')
    # Obsolete options
    parser.add_argument('-b', '--beginfirst', action='store_true', default=False,
                        help='(Obsolete, use waypoints instead)')
    parser.add_argument('-r', '--roundtrip', action='store_true', default=False,
                        help='(Obsolete, use waypoints instead)')
    args = parser.parse_args()

    if args.beginfirst or args.roundtrip:
        parser.error('Options -b and -r are obsolete. Use waypoints instead (see README).')

    if args.iterations < 0:
        parser.error('Number of extra samples should be positive')

    if args.plotdpi < 1:
        parser.error('%s is not a valid screen dpi' % args.plotdpi)

    if args.faction not in ('enl', 'res'):
        parser.error('Sorry, I do not know about faction "%s".' % args.faction)

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

    gs = gsheets.setup()

    portals, waypoints = gsheets.get_portals_from_sheet(gs, args.sheetid)
    logger.info('Considering %d portals and %s waypoints', len(portals), len(waypoints))

    if len(portals) < 3:
        logger.critical('Must have more than 2 portals!')
        sys.exit(1)

    maxfield.populateGraphs(portals, waypoints)

    maxfield.genDistanceMatrix(args.gmapskey, args.travelmode)

    (bestgraph, bestplan) = maxfield.loadCache(args.travelmode, args.maxmu, args.maxtime)

    if bestgraph is not None:
        beststats = maxfield.getWorkplanStats(bestgraph, bestplan)
        bestdist = beststats['dist']
        bestarea = beststats['area']
        bestkm = bestdist/float(1000)
        bestsqkm = bestarea/float(1000000)
        nicetime = beststats['nicetime']
        bestmutime = bestarea/beststats['time']
        bestap = beststats['ap']
        bestaptime = bestap/beststats['time']
        logger.info('Best distance of the plan loaded from cache: %0.2f km', bestkm)
        logger.info('Best coverage of the plan loaded from cache: %0.2f km2', bestsqkm)
        logger.info('Best AP of the plan loaded from cache: %s', bestap)

    else:
        bestkm = None
        bestsqkm = None
        beststats = None
        bestmutime = 0
        bestap = 0
        nicetime = '0:00'
        bestaptime = 0

    counter = 0

    if args.maxkeys:
        logger.info('Finding an efficient plan with max %s keys', args.maxkeys)
    elif args.maxmu:
        logger.info('Finding an efficient plan that maximizes MU coverage')
    else:
        logger.info('Finding an efficient plan that maximizes AP')

    failcount = 0
    seenplans = list()

    # set up multiprocessing
    ready_queue = mp.Queue()
    processes = list()
    for i in range(args.maxcpus):
        logger.debug('Starting process %s', i)
        p = mp.Process(target=queue_job, args=(args, ready_queue))
        processes.append(p)
        p.start()
    logger.info('Started %s worker processes', len(processes))

    logger.info('Ctrl-C to exit and use the latest best plan')

    try:
        while counter < args.iterations:
            if failcount > 1000:
                logger.info('Too many consecutive failures, exiting early.')
                break

            success, b, workplan, stats = ready_queue.get()

            if not success:
                failcount += 1
                continue
            if workplan in seenplans:
                # Already seen this plan, so don't consider it again. This is done
                # to avoid pingpoings for best plan.
                continue

            counter += 1

            if not args.quiet:
                if bestkm is not None:
                    sys.stdout.write('\r(Best: %0.2f km, %0.2f km2, %s portals, %s AP, %s): %s/%s     ' % (
                        bestkm, bestsqkm, bestgraph.order(), bestap, nicetime, counter, args.iterations))
                    sys.stdout.flush()

            failcount = 0

            mutime = int(stats['area']/stats['time'])
            aptime = int(stats['ap']/stats['time'])

            newbest = False
            if args.maxmu:
                # choose a plan that gives us most MU captured per distance of travel
                if mutime > bestmutime:
                    newbest = True
            else:
                # We want most AP per distance of travel
                if aptime > bestaptime:
                    newbest = True

            if newbest:
                if bestplan:
                    sys.stdout.write('\r(     \n')
                beststats = stats
                counter = 0
                bestplan = workplan
                seenplans.append(workplan)
                bestgraph = b
                bestdist = stats['dist']
                bestarea = stats['area']
                bestkm = bestdist/float(1000)
                bestsqkm = bestarea/float(1000000)
                nicetime = stats['nicetime']
                bestmutime = mutime
                bestap = stats['ap']
                bestaptime = aptime

    except KeyboardInterrupt:
        if not args.quiet:
            print()
            print('Exiting loop')
    finally:
        for p in processes:
            p.terminate()

    if not args.quiet:
        print()

    if bestplan is None:
        logger.critical('Could not find a solution for this list of portals.')
        sys.exit(1)

    maxfield.active_graph = bestgraph

    maxfield.saveCache(bestgraph, bestplan, args.travelmode, args.maxmu, args.maxtime)

    if args.plots:
        animate.make_png_steps(bestgraph, bestplan, args.plots, args.faction, args.plotdpi)

    gsheets.write_workplan(gs, args.sheetid, bestgraph, bestplan, beststats, args.faction, args.travelmode, args.nosave)


if __name__ == "__main__":
    main()
