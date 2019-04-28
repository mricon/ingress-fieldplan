#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse

from lib import gsheets, maxfield, animate

import logging
import multiprocessing as mp
import queue
import time

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

logger = logging.getLogger('fieldplan')

# version number
_V_ = '3.0.0'
# max portals allowed
_MAX_PORTALS_ = 25


def queue_job(a, ready_queue):
    # We just crank out plans until we are terminated
    while True:
        b = a.copy()
        success = maxfield.maxFields(b)
        if success:
            for t in b.triangulation:
                t.markEdgesWithFields()

            maxfield.improveEdgeOrder(b)

        ready_queue.put((success, b))


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
    parser.add_argument('-r', '--roundtrip', action='store_true', default=False,
                        help='Make sure the plan starts and ends at the same portal (may be less efficient).')
    parser.add_argument('-b', '--beginfirst', action='store_true', default=False,
                        help='Begin capture with the first portal in the spreadsheet (may be less efficient).')
    parser.add_argument('-p', '--plots', default=None,
                        help='Save step-by-step PNGs of the workplan into this directory.')
    parser.add_argument('--plotdpi', default=96, type=int,
                        help='DPI to use for generating plots (try 144 for high-dpi screens)')
    parser.add_argument('-g', '--gmapskey', default=None,
                        help='Google Maps API key (for better distances)')
    parser.add_argument('-f', '--faction', default='enl',
                        help='Set to "res" to use resistance colours')
    parser.add_argument('-u', '--maxmu', action='store_true', default=False,
                        help='Find a plan with best MU per distance travelled ratio')
    parser.add_argument('--maxcpus', default=mp.cpu_count(), type=int,
                        help='Maximum number of cpus to use')
    parser.add_argument('-l', '--log', default=None,
                        help='Log file where to log processing info')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='Add debug information to the logfile')
    parser.add_argument('-q', '--quiet', action='store_true', default=False,
                        help='Only output errors to the stdout')
    args = parser.parse_args()

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

    portals, blockers = gsheets.get_portals_from_sheet(gs, args.sheetid)
    logger.info('Considering %d portals', len(portals))

    if len(portals) < 3:
        logger.critical('Must have more than 2 portals!')
        sys.exit(1)

    if len(portals) > _MAX_PORTALS_:
        logger.critical('Portal limit is %d', _MAX_PORTALS_)

    a = maxfield.populateGraph(portals)

    ab = None
    if blockers:
        ab = maxfield.populateGraph(blockers)

    # Use a copy, because we concat ab to a for blockers distances
    maxfield.genDistanceMatrix(a.copy(), ab, args.gmapskey, args.travelmode)

    (bestgraph, bestplan) = maxfield.loadCache(a, ab, args.travelmode,
                                               args.beginfirst, args.roundtrip, args.maxmu)
    if bestgraph is not None:
        bestdist = maxfield.getWorkplanDist(bestgraph, bestplan)
        bestarea = maxfield.getWorkplanArea(bestgraph, bestplan)
        bestmudist = int(bestarea/bestdist)
        bestkm = bestdist/float(1000)
        bestsqkm = bestarea/float(1000000)
        logger.info('Best distance of the plan loaded from cache: %0.2f km', bestkm)
        logger.info('Best coverage of the plan loaded from cache: %0.2f sqkm', bestsqkm)

    else:
        bestkm = None
        bestsqkm = None
        bestdist = np.inf
        bestarea = 0
        bestmudist = 0

    counter = 0

    if args.maxkeys:
        logger.info('Finding an efficient plan with max %s keys', args.maxkeys)
    else:
        logger.info('Finding an efficient plan')

    failcount = 0
    seenplans = list()

    # set up multiprocessing
    ready_queue = mp.Queue()
    processes = list()
    for i in range(args.maxcpus):
        logger.debug('Starting process %s', i)
        p = mp.Process(target=queue_job, args=(a, ready_queue))
        processes.append(p)
        p.start()
    logger.info('Started %s worker processes', len(processes))

    logger.info('Ctrl-C to exit and use the latest best plan')

    try:
        while counter < args.iterations:
            if failcount >= 100:
                logger.info('Too many consecutive failures, exiting early.')
                break

            success, b = ready_queue.get()
            counter += 1

            if not args.quiet:
                if bestkm is not None:
                    sys.stdout.write('\r(Best: %0.2f km, %0.2f sqkm, %s sqm/m, %s actions): %s/%s      ' % (
                        bestkm, bestsqkm, bestmudist, len(bestplan), counter, args.iterations))
                    sys.stdout.flush()

            if not success:
                failcount += 1
                continue

            workplan = maxfield.makeWorkPlan(b, ab, args.roundtrip, args.beginfirst)

            if args.maxkeys:
                # do any of the portals require more than maxkeys
                sane_key_reqs = True
                for i in range(len(b.node)):
                    if b.in_degree(i) > args.maxkeys:
                        sane_key_reqs = False
                        break

                if not sane_key_reqs:
                    failcount += 1
                    logger.debug('Too many keys required, ignoring plan')
                    continue

            sane_out_links = True
            for i in range(len(b.node)):
                if b.out_degree(i) > 8:
                    sane_out_links = False
                    break

            if not sane_out_links:
                failcount += 1
                logger.debug('Too many outgoing links, ignoring plan')
                continue

            failcount = 0

            totaldist = maxfield.getWorkplanDist(b, workplan)
            totalarea = maxfield.getWorkplanArea(b, workplan)
            mudist = int(totalarea/totaldist)

            newbest = False
            if args.maxmu:
                # choose a plan that gives us most MU captured per distance of travel
                if mudist > bestmudist:
                    newbest = True
            else:
                # We want:
                # - the shorter plan, or
                # - the plan with a similar length that requires fewer actions, or
                # - the plan with a similar length that has higher mu per distance of travel
                # - we have not yet considered this plan
                if ((bestdist-totaldist > 80 or
                     (len(workplan) < len(bestplan) and totaldist-bestdist <= 80) or
                     (mudist > bestmudist and totaldist-bestdist <= 80))
                        and workplan not in seenplans):
                    newbest = True

            if newbest:
                counter = 0
                bestplan = workplan
                seenplans.append(workplan)
                bestgraph = b
                bestdist = totaldist
                bestarea = totalarea
                bestkm = bestdist/float(1000)
                bestsqkm = bestarea/float(1000000)
                bestmudist = mudist

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

    maxfield.saveCache(bestgraph, ab, bestplan, args.travelmode,
                       args.beginfirst, args.roundtrip, args.maxmu)

    if args.plots:
        animate.make_png_steps(bestgraph, bestplan, args.plots, args.faction, args.plotdpi)

    gsheets.write_workplan(gs, args.sheetid, bestgraph, bestplan, args.faction, args.travelmode, args.nosave,
                           args.roundtrip, args.maxmu)


if __name__ == "__main__":
    main()
