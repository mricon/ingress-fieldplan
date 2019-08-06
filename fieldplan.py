#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse

from lib import gsheets, maxfield, animate

import logging
import multiprocessing as mp
import time
import queue

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

logger = logging.getLogger('fieldplan')

# version number
_V_ = '3.2.0'

_maxfield_names = {'combined_graph', 'portal_graph', 'waypoint_graph', 'active_graph', 'capture_cache', 'dist_matrix',
                   'time_matrix', 'direct_dist_matrix'}


def push_maxfield_data(args):
    if sys.platform != 'win32':
        return
    source = vars(maxfield)
    dest = vars(args)
    for name in _maxfield_names:
        dest[name] = source[name]


def pop_maxfield_data(args):
    if sys.platform != 'win32':
        return
    source = vars(args)
    dest = vars(maxfield)
    for name in _maxfield_names:
        dest[name] = source[name]


def queue_job(args, best, counter, ready_queue):
    nogood = 0
    # A bit of a magical number used for subsets
    # (basically, 10% of iterations per subprocess)
    nogood_max = int(args.iterations/10/args.maxcpus)
    pop_maxfield_data(args)
    if args.maxtime:
        is_subset = True
        subset = maxfield.makeSubset(3, args.maxmu)
        mygraph = maxfield.makeSubsetGraph(subset)
    else:
        is_subset = False
        subset = None
        mygraph = maxfield.portal_graph

    failed = (False, None, None, None)
    bestmode = False
    mycounter = 0

    while True:
        mycounter += 1
        # Increment global counter by 10s to reduce lock contention
        if mycounter >= 10:
            with counter.get_lock():
                counter.value += mycounter
            mycounter = 0

        b = mygraph.copy()

        success = maxfield.maxFields(b)
        if not success:
            ready_queue.put(failed)
            continue

        for t in b.triangulation:
            t.markEdgesWithFields()

        maxfield.extendGraphWithWaypoints(b)
        maxfield.active_graph = b

        maxfield.improveEdgeOrder(b)
        linkplan = maxfield.makeLinkPlan(b)
        workplan = maxfield.makeWorkPlan(b, linkplan, is_subset)

        if workplan is None:
            ready_queue.put(failed)
            continue

        stats = maxfield.getWorkplanStats(b, workplan, cooling=args.cooling)

        if args.maxmu:
            mybest = stats['sqmpmin']
        else:
            mybest = stats['appmin']

        if args.maxtime:
            if bestmode:
                # In best mode, we just run the same graph in hopes to improve it
                if nogood > nogood_max:
                    # Couldn't improve on the best
                    bestmode = False
            elif args.minap is not None and stats['ap'] < args.minap:
                # Add moar portals
                maxfield.active_graph = None
                maxfield.addSubsetPortal(subset, args.maxmu)
                mygraph = maxfield.makeSubsetGraph(subset)
                continue
            elif stats['time'] > args.maxtime:
                nogood += 1
                if nogood > nogood_max:
                    # start from a brand new random triangle
                    subset = maxfield.makeSubset(3, args.maxmu)
                    mygraph = maxfield.makeSubsetGraph(subset)
                    nogood = 0
                continue

            maxfield.active_graph = None
            maxfield.addSubsetPortal(subset, args.maxmu)
            mygraph = maxfield.makeSubsetGraph(subset)

        if mybest <= best.value:
            nogood += 1
            continue

        with best.get_lock():
            best.value = mybest
        with counter.get_lock():
            counter.value = 0
        ready_queue.put((success, b, workplan, stats))
        nogood = 0
        mycounter = 0
        bestmode = True


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
    parser.add_argument('--minap', default=None, type=int,
                        help='Ignore plans that result in less AP than specified (used with --maxtime)')
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
    parser.add_argument('-k', '--maxkeys', type=int, default=None,
                        help='(Obsolete, use --cooling options instead)')
    args = parser.parse_args()

    if args.beginfirst or args.roundtrip:
        parser.error('Options -b and -r are obsolete. Use waypoints instead (see README).')
    if args.maxkeys:
        parser.error('Option -k is obsolete. Use --cooling instead.')

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

    if args.maxtime:
        bestgraph = None
        bestplan = None
    else:
        (bestgraph, bestplan) = maxfield.loadCache(args.travelmode, args.maxmu, args.maxtime)

    if args.maxmu:
        beststr = 'm2/min'
    else:
        beststr = 'AP/min'

    beststats = None
    bestkm = '-.--'
    bestsqkm = '-.--'
    bestap = '-----'
    nicetime = '-:--'
    bestportals = '-'
    best = 0

    if bestgraph is not None:
        beststats = maxfield.getWorkplanStats(bestgraph, bestplan)
        if args.maxmu:
            best = beststats['sqmpmin']
        else:
            best = beststats['appmin']

    logger.info('Finding an efficient plan that maximizes %s', beststr)

    failcount = 0
    seenplans = list()

    s_best = mp.Value('I', best)
    s_counter = mp.Value('I', 0)

    push_maxfield_data(args)

    ready_queue = mp.Queue(maxsize=10)
    processes = list()
    for i in range(args.maxcpus):
        logger.debug('Starting process %s', i)
        p = mp.Process(target=queue_job, args=(args, s_best, s_counter, ready_queue))
        processes.append(p)
        p.start()
    logger.info('Started %s worker processes', len(processes))

    logger.info('Ctrl-C to exit and use the latest best plan')

    try:
        while True:
            if failcount > 1000:
                logger.info('Too many consecutive failures, exiting early.')
                break

            if beststats is not None:
                bestdist = beststats['dist']
                bestarea = beststats['area']
                bestap = beststats['ap']
                bestkm = '%0.2f' % (bestdist/float(1000))
                bestsqkm = '%0.2f' % (bestarea/float(1000000))
                nicetime = beststats['nicetime']
                if args.maxmu:
                    best = beststats['sqmpmin']
                else:
                    best = beststats['appmin']
                bestportals = bestgraph.order()
                if maxfield.waypoint_graph is not None:
                    bestportals -= maxfield.waypoint_graph.order()

            if not args.quiet:
                sys.stdout.write('\r(Best: %s km, %s km2, %s portals, %s AP, %s %s, %s): %s/%s     ' % (
                    bestkm, bestsqkm, bestportals, bestap, best, beststr, nicetime, s_counter.value,
                    args.iterations))
                sys.stdout.flush()

            if s_counter.value >= args.iterations:
                break

            try:
                success, b, workplan, stats = ready_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.1)
                continue

            if not success:
                failcount += 1
                continue
            if workplan in seenplans:
                # Already seen this plan, so don't consider it again. This is done
                # to avoid pingpoings for best plan.
                continue

            if not args.quiet:
                if bestkm is not None:
                    sys.stdout.write('\r(      %s km, %s km2, %s portals, %s AP, %s %s, %s)            \n' % (
                        bestkm, bestsqkm, bestportals, bestap, best, beststr, nicetime))

            beststats = stats
            bestgraph = b
            bestplan = workplan

            if args.maxmu:
                best = stats['sqmpmin']
            else:
                best = stats['appmin']

            failcount = 0

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
