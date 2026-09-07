# -*- coding: utf-8 -*-

import sys
import os

from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file

from lib import maxfield, plansteps

from pathlib import Path
from urllib.parse import urlparse, parse_qs

import logging
logger = logging.getLogger('fieldplan')

# Sheets rejects titles longer than this, and rejects duplicates outright
SHEET_TITLE_MAXLEN = 100


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


def a1_quote(title):
    # A1 notation needs the sheet name single-quoted once it contains
    # anything but letters, digits and underscores, and a literal
    # single quote inside the name is escaped by doubling it
    return "'%s'" % title.replace("'", "''")


def unique_sheet_title(title, existing):
    # Sheets refuses to add a second sheet with an existing title, so a
    # re-run that produces the same plan used to die with an HttpError
    # (issues #26/#27). Suffix until free, trimming the base to stay
    # inside the length limit. Compared case-insensitively, since it is
    # cheaper to add a needless suffix than to guess wrong and crash.
    taken = {t.strip().lower() for t in existing}
    title = title.strip()[:SHEET_TITLE_MAXLEN].strip()
    if title.lower() not in taken:
        return title
    num = 2
    while True:
        suffix = ' #%d' % num
        candidate = title[:SHEET_TITLE_MAXLEN - len(suffix)].strip() + suffix
        if candidate.lower() not in taken:
            return candidate
        num += 1


def get_sheet_titles(service, spid):
    res = service.spreadsheets().get(
        spreadsheetId=spid,
        fields='sheets.properties.title'
    ).execute()
    return [sheet['properties']['title'] for sheet in res.get('sheets', [])]


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

    stops = plansteps.build_stops(a, workplan, travelmode)
    n_waypoints = 0

    for stop in stops:
        travel = stop['travel']
        mapurl = stop['mapurl']

        if travel is None:
            planrows.append((travelmoji[travelmode], '=HYPERLINK("%s"; "map")' % mapurl))
            logger.info('-->Start at %s', stop['name'])
        elif travel['moved']:
            # The sheet shows the walk time next to a distance in km, where
            # it is long enough to be worth planning around
            if travel['dist'] >= 500:
                nicedist = '%0.1f km (%s min)' % (travel['dist']/float(1000), travel['time'])
            else:
                nicedist = travel['nicedist']
            planrows.append((u'\u25bc',))
            planrows.append((travelmoji[travelmode], '=HYPERLINK("%s"; "%s")' % (mapurl, nicedist)))
            logger.info('-->Move to %s [%s]', stop['name'], nicedist)
        else:
            planrows.append((u'\u25bc',))

        logger.info('--|At %s', stop['name'])
        if stop['is_waypoint']:
            planrows.append(('W', stop['name']))
            n_waypoints += 1
            continue

        planrows.append(('P', stop['name']))

        if stop['is_blocker']:
            planrows.append(('X', 'destroy blocker'))
            logger.info('--|X: destroy blocker')
            n_waypoints += 1
            continue

        keys = stop['keys']
        if keys:
            if keys['lastvisit']:
                logger.info('--|H: ensure %d keys', keys['total'])
                planrows.append(('H', 'ensure %d keys' % keys['total']))
            elif keys['ensure']:
                logger.info('--|H: ensure %d keys (%d max)', keys['ensure'], keys['total'])
                planrows.append(('H', 'ensure %d keys (%d max)' % (keys['ensure'], keys['total'])))
            else:
                logger.info('--|H: %d max keys needed', keys['total'])
                planrows.append(('H', '%d max keys needed' % keys['total']))

        if stop['shields']:
            planrows.append(('S', 'shields on (%d links)' % stop['shields']['links']))
            logger.info('--|S: shields on (%d out, %d in)',
                        a.out_degree(stop['node']), a.in_degree(stop['node']))

        for link in stop['links']:
            planrows.append((link['action'], u'\u25b6%s' % link['name']))
            logger.info('  \\%s--> %s', link['action'], link['name'])

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
    dtitle = unique_sheet_title(dtitle, get_sheet_titles(service, spid))
    logger.info('Adding "%s" sheet with %d actions', dtitle, len(workplan))
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
            # Use the title the API actually assigned, not the one we asked for
            dtitle = blurb['addSheet']['properties']['title']

    # Now we generate a values update request
    updates = list()

    updates.append({
        'range': '%s!A1:B%d' % (a1_quote(dtitle), len(planrows)),
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
