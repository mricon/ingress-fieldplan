# -*- coding: utf-8 -*-

import sys
import os
import datetime

from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file

from lib import maxfield

from pathlib import Path
from urllib.parse import urlparse, parse_qs

import logging
logger = logging.getLogger('fieldplan')


def setup():
    home = str(Path.home())
    cachedir = os.path.join(home, '.cache', 'ingress-fieldmap')
    tokenfile = os.path.join(cachedir, 'token.json')
    if not os.path.isfile(tokenfile):
        logger.critical('Did not find token.json. Run obtainGSToken.py first.')
        sys.exit(1)
    store = file.Storage(tokenfile)
    creds = store.get()
    if not creds or creds.invalid:
        logger.critical('Invaid token file in %s. Delete and rerun obtainGSToken.py.', tokenfile)
        sys.exit(1)

    return build('sheets', 'v4', http=creds.authorize(Http()))


def _get_qp_from_url(url, qp='pll'):
    p_url = urlparse(url)
    q_parts = parse_qs(p_url.query)
    if qp not in q_parts:
        logger.debug('link=%s', url)
        logger.info('Portal link does not look sane, ignoring')
        return None

    return q_parts[qp][0]


def get_portals_from_sheet(service, spid):
    # Does the sheet ID contain slashes? If so, it's the full URL.
    if spid.find('/') > 0:
        chunks = spid.split('/')
        spid = chunks[5]
    # We only consider first 100 lines
    srange = 'A1:B100'
    res = service.spreadsheets().values().get(
        spreadsheetId=spid,
        range=srange
    ).execute()
    rows = res.get('values', [])
    portals = []
    waypoints = []
    logger.info('Grabbing the spreadsheet')
    at_row = 0
    startpoint_loc = None
    endpoint_loc = None
    for row in rows:
        logger.debug('at_row=%s', at_row)
        if not len(row) or not len(row[0].strip()):
            at_row += 1
            continue
        if row[0][0] == '#':
            # Is the next one a bang?
            if len(row[0]) > 3 and row[0][1] == '!':
                # Is it a waypoint?
                if row[0][2] == 's' and row[1].find('ll='):
                    # Starting waypoint
                    if startpoint_loc is not None:
                        logger.critical('Multiple start waypoints found!')
                    name = row[0][3:].lstrip()
                    coords = _get_qp_from_url(row[1], qp='ll')
                    waypoints.append((name, coords, '_w_start'))
                    logger.info('Adding start waypoint: %s', name)
                    at_row += 1
                    continue
                if row[0][2] == 'e' and row[1].find('ll='):
                    # Ending waypoint
                    if endpoint_loc is not None:
                        logger.critical('Multiple end waypoints found!')
                    name = row[0][3:].lstrip()
                    coords = _get_qp_from_url(row[1], qp='ll')
                    waypoints.append((name, coords, '_w_end'))
                    logger.info('Adding end waypoint: %s', name)
                    endpoint_loc = len(waypoints)-1
                    at_row += 1
                    continue
                if row[0][2] == 'b' and row[1].find('pll='):
                    # Blocker waypoint
                    name = row[0][3:].lstrip()
                    coords = _get_qp_from_url(row[1])
                    waypoints.append((name, coords, '_w_blocker'))
                    logger.info('Adding blocker waypoint: %s', name)
                    at_row += 1
                    continue
            # Comment ignored
            at_row += 1
            continue
        if row[1].find('pll=') < 0:
            logger.debug('link=%s', row[1])
            logger.info('Portal link does not look sane, ignoring')
            at_row += 1
            continue

        logger.info('Adding portal: %s', row[0])
        portals.append((row[0], _get_qp_from_url(row[1])))
        at_row += 1

    # make sure end waypoint is always last in the waypoint list
    if endpoint_loc is not None and endpoint_loc != len(waypoints)-1:
        _ep = waypoints.pop(endpoint_loc)
        waypoints.append(_ep)

    return portals, waypoints


def write_workplan(service, spid, a, workplan, stats, faction, travelmode='walking', nosave=False):
    from pprint import pformat

    if spid.find('/') > 0:
        chunks = spid.split('/')
        spid = chunks[5]
    # Use for spreadsheet rows
    planrows = []
    travelmoji = {
        'walking': u"\U0001F6B6",
        'bicycling': u"\U0001F6B2",
        'transit': u"\U0001F68D",
        'driving': u"\U0001F697",
    }

    n = a.order()
    logger.info('portals:')
    for p in range(n):
        logger.info('    %d: %s', p, a.nodes[p]['name'])
    logger.info('orig_workplan:\n%s', pformat(a.orig_workplan))
    logger.info('fixes:\n%s', pformat(a.fixes))
    logger.info('workplan:\n%s', pformat(workplan))
    logger.info('stats:\n%s', pformat(stats))

    # Track which portals we've already captured
    # (easier than going through the list backwards)
    prev_p = None
    plan_at = 0
    n_waypoints = 0

    for p, q, f in workplan:
        plan_at += 1

        # Are we at a different location than the previous portal?
        if p != prev_p:
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

            mapurl = 'https://www.google.com/maps/dir/?api=1&destination=%s&travelmode=%s' % (
                a.nodes[p]['pll'], travelmode
            )
            if 'special' in a.nodes[p]:
                special = a.nodes[p]['special']
            else:
                special = None

            if prev_p is not None:
                dist = maxfield.get_portal_distance(prev_p, p)
                duration = maxfield.get_portal_time(prev_p, p)

                if dist > 40:
                    if dist >= 500:
                        nicedist = '%0.1f km (%s min)' % ((dist/float(1000)), duration)
                    else:
                        nicedist = '%d m' % dist
                    hyperlink = '=HYPERLINK("%s"; "%s")' % (mapurl, nicedist)
                    planrows.append((u'▼',))
                    planrows.append((travelmoji[travelmode], hyperlink))
                    logger.info('-->Move to %s [%s]', a.nodes[p]['name'], nicedist)
                else:
                    planrows.append((u'▼',))
            else:
                hyperlink = '=HYPERLINK("%s"; "map")' % mapurl
                planrows.append((travelmoji[travelmode], hyperlink))
                logger.info('-->Start at %s', a.nodes[p]['name'])

            logger.info('--|At %s', a.nodes[p]['name'])
            # Are we at a waypoint?
            if special in ('_w_start', '_w_end'):
                planrows.append(('W', a.nodes[p]['name']))
                # Nothing else here
                prev_p = p
                n_waypoints += 1
                continue

            planrows.append(('P', a.nodes[p]['name']))

            # Are we at a blocker?
            if special == '_w_blocker':
                planrows.append(('X', 'destroy blocker'))
                logger.info('--|X: destroy blocker')
                # Nothing else here
                prev_p = p
                n_waypoints += 1
                continue

            if totalkeys:
                if lastvisit:
                    logger.info('--|H: ensure %d keys', totalkeys)
                    planrows.append(('H', 'ensure %d keys' % totalkeys))
                elif ensurekeys:
                    logger.info('--|H: ensure %d keys (%d max)', ensurekeys, totalkeys)
                    planrows.append(('H', 'ensure %d keys (%d max)' % (ensurekeys, totalkeys)))
                else:
                    logger.info('--|H: %d max keys needed', totalkeys)
                    planrows.append(('H', '%d max keys needed' % totalkeys))

            if lastvisit:
                totallinks = a.out_degree(p) + a.in_degree(p)
                planrows.append(('S', 'shields on (%d links)' % totallinks))
                logger.info('--|S: shields on (%d out, %d in)', a.out_degree(p), a.in_degree(p))

            prev_p = p

        if q is not None:
            # Add links/fields
            if f > 1:
                action = 'D'
            elif f == 1:
                action = 'F'
            else:
                action = 'L'

            planrows.append((action, u'▶%s' % a.nodes[q]['name']))
            logger.info('  \\%s--> %s', action, a.nodes[q]['name'])

    totalkm = stats['dist']/float(1000)
    logger.info('Total workplan distance: %0.2f km', totalkm)
    logger.info('Total workplan play time: %s (%s %s)',
                stats['nicetime'], stats['nicetraveltime'], travelmode)
    logger.info('Total AP: %s (%s without capturing)', stats['ap'], stats['ap'] - (a.order()*maxfield.CAPTUREAP))
    if stats['hs']:
        logger.info('Total %s needed: %s', maxfield.cooling.upper(), stats['hs'])
    if nosave:
        logger.info('Not saving spreadsheet per request.')
        return

    # title = 'Ingress: around %s (%s AP)' % (a.node[0]['name'], '{:,}'.format(totalap))
    # logger.info('Setting spreadsheet title: %s', title)

    requests = list()
    # requests.append({
    #    'updateSpreadsheetProperties': {
    #        'properties': {
    #            'title': title,
    #            'locale': 'en_US',
    #        },
    #        'fields': 'title',
    #    }
    # })

    dtitle = '%s %s (%0.2fkm/%dP/%sAP)' % (travelmoji[travelmode], stats['nicetime'], totalkm,
                                           a.order()-n_waypoints, '{:,}'.format(stats['ap']))
    logger.info('Adding "%s" sheet with %d actions', dtitle, len(workplan))
    
     # Check if a sheet with the same name already exists
    sheet_exists = False
    sheets = service.spreadsheets().get(spreadsheetId=spid, fields='sheets(properties(title))').execute().get('sheets', [])
    for sheet in sheets:
        if sheet['properties']['title'] == dtitle:
            sheet_exists = True
            break

    # Append a timestamp to the sheet name if it already exists
    if sheet_exists:
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        dtitle += ' ' + timestamp
        logger.info('Sheet "%s" already exists, using "%s" instead', dtitle, dtitle)
        
        
    requests.append({
        'addSheet': {
            'properties': {
                'title': dtitle,
            }
        }
    })

    body = {
        'requests': requests,
    }
    res = service.spreadsheets().batchUpdate(spreadsheetId=spid, body=body).execute()
    sheet_ids = []
    for blurb in res['replies']:
        if 'addSheet' in blurb:
            sheet_ids.append(blurb['addSheet']['properties']['sheetId'])

    # Now we generate a values update request
    updates = list()

    updates.append({
        'range': '%s!A1:B%d' % (dtitle, len(planrows)),
        'majorDimension': 'ROWS',
        'values': planrows,
    })

    body = {
        'valueInputOption': 'USER_ENTERED',
        'data': updates,
    }

    service.spreadsheets().values().batchUpdate(spreadsheetId=spid, body=body).execute()

    logger.info('Resizing columns and adding colours')
    # now auto-resize all columns
    requests = []
    colors = [
        ('H', 1.0, 0.6, 0.4),  # Hack
        ('S', 0.9, 0.7, 0.9),  # Shield
        ('T', 0.9, 0.9, 0.9),  # Travel
        ('P', 0.6, 0.6, 0.6),  # Portal
        ('W', 0.6, 0.6, 0.6),  # Waypoint
        ('X', 1.0, 0.6, 0.6),  # Blocker
    ]
    if faction == 'res':
        colors += [
            ('L', 0.6, 0.8, 1.0),  # Link
            ('F', 0.3, 0.5, 1.0),  # Field
            ('D', 0.2, 0.3, 0.8),  # Double Field
        ]
    else:
        colors += [
            ('L', 0.8, 1.0, 0.8),  # Link
            ('F', 0.5, 1.0, 0.5),  # Field
            ('D', 0.3, 0.8, 0.3),  # Double Field
        ]

    for sid in sheet_ids:
        requests.append({
            'autoResizeDimensions': {
                'dimensions': {
                    'sheetId': sid,
                    'dimension': 'COLUMNS',
                    'startIndex': 0,
                    'endIndex': 3,
                }
            }
        })

        # set conditional formatting on the Workplan sheet
        my_range = {
            'sheetId': sid,
            'startRowIndex': 0,
            'endRowIndex': len(planrows),
            'startColumnIndex': 0,
            'endColumnIndex': 1,
        }
        for text, red, green, blue in colors:
            requests.append({
                'addConditionalFormatRule': {
                    'rule': {
                        'ranges': [my_range],
                        'booleanRule': {
                            'condition': {
                                'type': 'TEXT_EQ',
                                'values': [{'userEnteredValue': text}]
                            },
                            'format': {
                                'backgroundColor': {'red': red, 'green': green, 'blue': blue}
                            }
                        }
                    },
                    'index': 0
                }
            })

    body = {
        'requests': requests,
    }
    service.spreadsheets().batchUpdate(spreadsheetId=spid, body=body).execute()
    logger.info('Spreadsheet generation done')
