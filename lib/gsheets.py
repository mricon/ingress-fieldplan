# -*- coding: utf-8 -*-

import sys
import os

from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file

from lib import maxfield

from pathlib import Path

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


def _get_portal_info_from_row(row, hyperlinks=None):
    try:
        if not len(row) or not len(row[0].strip()):
            # Row starting with an empty cell ignored
            return None
        if row[0][0] == '#':
            # Comment ignored
            return None
        if row[1].find('pll=') < 0:
            logger.debug('link=%s', row[1])
            logger.info('Portal link does not look sane, ignoring')
            return None

        return row

    except IndexError:
        logger.info('Bad portal row: %s', row)
        return None


def get_portals_from_sheet(service, spid):
    # Does the sheet ID contain slashes? If so, it's the full URL.
    if spid.find('/') > 0:
        chunks = spid.split('/')
        spid = chunks[5]
    # We only consider first 100 lines
    srange = 'Portals!A1:B100'
    res = service.spreadsheets().values().get(
        spreadsheetId=spid,
        range=srange
    ).execute()
    rows = res.get('values', [])
    portals = []
    logger.info('Grabbing the spreadsheet')
    hyperlinks = None
    # if the first row/column says #!iitc, then we parse the sheet
    # as iitc portal listing copypaste
    if rows[0][0] == '#!iitc':
        # Grab hyperlink data
        res = service.spreadsheets().get(
            spreadsheetId=spid,
            ranges=['Portals!B1:B100'],
            fields="sheets/data/rowData/values/hyperlink",
        ).execute()
        hyperlinks = res['sheets'][0]['data'][0]['rowData']

    at_row = 0
    for row in rows:
        if not len(row) or not len(row[0].strip()):
            at_row += 1
            continue
        if row[0][0] == '#':
            # Comment ignored
            at_row += 1
            continue
        if row[1].find('pll=') < 0:
            if hyperlinks is not None:
                # Grab portal name and location from iitc paste
                name = row[1]
                url = hyperlinks[at_row]['values'][0]['hyperlink']
                if url.find('pll=') < 0:
                    logger.debug('hyperlink=%s', hyperlinks[at_row])
                    logger.info('IITC row link does not look sane, ignoring')
                    at_row += 1
                    continue
                logger.info('Adding portal from IITC paste: %s', name)
                at_row += 1
                portals.append([name, url])
                continue
            else:
                logger.debug('link=%s', row[1])
                logger.info('Portal link does not look sane, ignoring')
                at_row += 1
                continue
        else:
            logger.info('Adding portal: %s', row[0])
            portals.append(row)
            at_row += 1

    # Now do blockers
    srange = 'Blockers!A1:B50'
    blockers = []
    # There may not be any
    try:
        res = service.spreadsheets().values().get(
            spreadsheetId=spid, range=srange).execute()
        rows = res.get('values', [])
        for row in rows:
            pinfo = _get_portal_info_from_row(row)
            if pinfo is not None:
                logger.info('Adding blocker: %s', pinfo[0])
                blockers.append(pinfo)
    except:
        pass

    return portals, blockers


def write_workplan(service, spid, a, workplan, stats, faction, travelmode='walking', nosave=False):

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
    logger.debug('portals:')
    for p in range(n):
        logger.debug('    %d: %s', p, a.node[p]['name'])
    logger.debug('orig_linkplan:')
    for line in a.orig_linkplan:
        logger.debug('    %s: %s -> %s', line, a.node[line[0]]['name'], a.node[line[1]]['name'])
    logger.debug('    len: %s', len(a.orig_linkplan))
    logger.debug('fixes:')
    for line in a.fixes:
        logger.debug('    %s', line)
    logger.debug('fixed linkplan:')
    for line in a.linkplan:
        logger.debug('    %s: %s -> %s', line, a.node[line[0]]['name'], a.node[line[1]]['name'])
    logger.debug('    len: %s', len(a.linkplan))
    logger.debug('clusters:')
    for line in a.clusters:
        logger.debug('    %s', line)
    logger.debug('captureplan:')
    for line in a.captureplan:
        logger.debug('    %s: %s', line, a.node[line]['name'])
    logger.debug('workplan:')
    for line in workplan:
        logger.debug('    %s', line)

    logger.debug('stats: %s', stats)

    # Track which portals we've already captured
    # (easier than going through the list backwards)
    prev_p = None
    plan_at = 0

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
                a.node[p]['pll'], travelmode
            )
            if prev_p is not None:
                dist = maxfield.getPortalDistance(prev_p, p)
                duration = maxfield.getPortalTime(prev_p, p)

                if dist > 40:
                    if dist >= 500:
                        nicedist = '%0.1f km (%s min)' % ((dist/float(1000)), duration)
                    else:
                        nicedist = '%d m' % dist
                    hyperlink = '=HYPERLINK("%s"; "%s")' % (mapurl, nicedist)
                    planrows.append((u'▼',))
                    planrows.append((travelmoji[travelmode], hyperlink))
                    logger.info('-->Move to %s [%s]', a.node[p]['name'], nicedist)
                else:
                    planrows.append((u'▼',))
            else:
                hyperlink = '=HYPERLINK("%s"; "map")' % mapurl
                planrows.append((travelmoji[travelmode], hyperlink))
                logger.info('-->Start at %s', a.node[p]['name'])

            logger.info('--|At %s', a.node[p]['name'])
            planrows.append(('P', a.node[p]['name']))

            # Are we at a blocker?
            if 'blocker' in a.node[p]:
                planrows.append(('X', 'destroy blocker'))
                logger.info('--|X: destroy blocker')
                # Nothing else here
                prev_p = p
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
            if f:
                action = 'F'
            else:
                action = 'L'

            planrows.append((action, u'▶%s' % a.node[q]['name']))
            logger.info('  \%s--> %s', action, a.node[q]['name'])

    totalkm = stats['dist']/float(1000)
    logger.info('Total workplan distance: %0.2f km', totalkm)
    logger.info('Total workplan play time: %s (%s %s)',
                stats['nicetime'], stats['nicetraveltime'], travelmode)
    logger.info('Total AP: %s (%s without capturing)', stats['ap'], stats['ap'] - (a.order()*maxfield.CAPTUREAP))
    if nosave:
        logger.info('Not saving spreadsheet per request.')
        return

    #title = 'Ingress: around %s (%s AP)' % (a.node[0]['name'], '{:,}'.format(totalap))
    #logger.info('Setting spreadsheet title: %s', title)

    requests = list()
    #requests.append({
    #    'updateSpreadsheetProperties': {
    #        'properties': {
    #            'title': title,
    #            'locale': 'en_US',
    #        },
    #        'fields': 'title',
    #    }
    #})

    dtitle = '%s %s (%0.2fkm/%dP/%sAP)' % (travelmoji[travelmode], stats['nicetime'], totalkm,
                                           a.order(), '{:,}'.format(stats['ap']))
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
        ('X', 1.0, 0.6, 0.6),  # Blocker
    ]
    if faction == 'res':
        colors += [
            ('L', 0.6, 0.8, 1.0),  # Link
            ('F', 0.3, 0.5, 1.0),  # Field
        ]
    else:
        colors += [
            ('L', 0.8, 1.0, 0.8),  # Link
            ('F', 0.5, 1.0, 0.5),  # Field
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
