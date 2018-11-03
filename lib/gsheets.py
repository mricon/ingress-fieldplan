# -*- coding: utf-8 -*-

from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file

from lib import maxfield

import logging
logger = logging.getLogger('maxfield3')


def setup(tokenfile):
    store = file.Storage(tokenfile)
    creds = store.get()
    return build('sheets', 'v4', http=creds.authorize(Http()))


def _get_portal_info_from_row(row):
    try:
        if row[0][0] == '#':
            # Comment ignored
            return None
        if not len(row[0][0].strip()):
            # Row starting with an empty cell ignored
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
    # We only consider first 50 lines
    srange = 'Portals!A1:B50'
    res = service.spreadsheets().values().get(
        spreadsheetId=spid, range=srange).execute()
    rows = res.get('values', [])
    portals = []
    logger.info('Grabbing the spreadsheet')
    for row in rows:
        pinfo = _get_portal_info_from_row(row)
        if pinfo is not None:
            logger.info('Adding portal: %s', pinfo[0])
            portals.append(pinfo)

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


def write_workplan(service, spid, a, workplan, travelmode='walking'):
    # Use for spreadsheet rows
    planrows = []
    travelmoji = {
        'walking': u"\U0001F6B6",
        'bicycling': u"\U0001F6B2",
        'transit': u"\U0001F68D",
        'driving': u"\U0001F697",
    }

    logger.debug('orig_linkplan:')
    for line in a.orig_linkplan:
        logger.debug('    %s', line)
    logger.debug('    len: %s', len(a.orig_linkplan))
    logger.debug('fixes:')
    for line in a.fixes:
        logger.debug('    %s', line)
    logger.debug('fixed linkplan:')
    for line in a.linkplan:
        logger.debug('    %s', line)
    logger.debug('    len: %s', len(a.linkplan))
    logger.debug('captureplan:')
    for line in a.captureplan:
        logger.debug('    %s', line)
    logger.debug('workplan:')
    for line in workplan:
        logger.debug('    %s', line)

    # Track which portals we've already captured
    # (easier than going through the list backwards)
    prev_p = None
    plan_at = 0
    totaldist = 0
    links = 0
    fields = 0
    for p, q, f in workplan:
        plan_at += 1

        # Are we at a different location than the previous portal?
        if p != prev_p:
            # How many keys do we need if/until we come back?
            needkeys = 0
            # Track when we leave this portal
            lastvisit = True
            same_p = True
            for fp, fq, ff in workplan[plan_at:]:
                if fp == p:
                    # Are we still at the same portal?
                    if same_p:
                        continue
                    lastvisit = False
                    break
                else:
                    # we're at a different portal
                    same_p = False
                if fq == p:
                    # Future link to this portal
                    needkeys += 1

            mapurl = 'https://www.google.com/maps/dir/?api=1&destination=%s&travelmode=%s' % (
                a.node[p]['pll'], travelmode
            )
            if prev_p is not None:
                if links:
                    logger.info('    total links: %d', links)
                if fields:
                    logger.info('    total fields: %d', fields)

                dist = maxfield.getPortalDistance(prev_p, p)
                if dist > 80:
                    totaldist += dist
                    if dist >= 500:
                        nicedist = '%0.1f km' % (dist/float(1000))
                    else:
                        nicedist = '%d m' % dist
                    hyperlink = '=HYPERLINK("%s", "%s")' % (mapurl, nicedist)
                    planrows.append((u'▼',))
                    planrows.append((travelmoji[travelmode], hyperlink))
                    logger.info('-->Move to %s (%s)', a.node[p]['name'], nicedist)
                else:
                    planrows.append((u'▼',))
            else:
                hyperlink = '=HYPERLINK("%s", "map")' % mapurl
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

            if needkeys:
                planrows.append(('H', '%d total keys' % needkeys))
                logger.info('--|H: need %d total keys', needkeys)

            if lastvisit:
                # How many total links to and from this portal?
                planrows.append(('S', '%d out, %d in' % (a.out_degree(p), a.in_degree(p))))
                logger.info('--|S: %d out, %d in', a.out_degree(p), a.in_degree(p))

            prev_p = p

        if q is not None:
            # Add links/fields
            action = 'L'
            links += 1
            if f:
                fields += 1
                action = 'F'

            planrows.append((action, u'▶%s' % a.node[q]['name']))
            logger.info('  \%s--> %s', action, a.node[q]['name'])

    if links:
        logger.info('    total links: %d', links)
    if fields:
        logger.info('    total fields: %d', fields)

    totalkm = totaldist/float(1000)
    logger.info('Total workplan distance: %0.2f km', totalkm)
    title = 'Ingress: around %s (%0.2f km)' % (a.node[0]['name'], totalkm)
    logger.info('Setting spreadsheet title: %s', title)

    requests = list()
    requests.append({
        'updateSpreadsheetProperties': {
            'properties': {
                'title': title,
            },
            'fields': 'title',
        }
    })

    logger.info('Adding "Workplan" sheet with %d rows', len(workplan))
    requests.append({
        'addSheet': {
            'properties': {
                'title': 'Workplan',
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
        'range': 'Workplan!A1:B%d' % len(planrows),
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
    colors = (
        ('H', 1.0, 0.6, 0.4),  # Hack
        ('S', 0.6, 0.8, 1.0),  # Shield
        ('L', 0.8, 1.0, 0.8),  # Link
        ('F', 0.5, 1.0, 0.5),  # Field
        ('T', 0.9, 0.9, 0.9),  # Travel
        ('P', 0.6, 0.6, 0.6),  # Portal
        ('X', 1.0, 0.6, 0.6),  # Blocker
    )
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
