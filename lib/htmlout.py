# -*- coding: utf-8 -*-
"""
A self-contained HTML plan you can open on a phone and follow on foot.

The point of this format is to need nothing: no Google account, no server,
no network once the file is on the device. Everything -- markup, styling,
behaviour, and the plan itself -- goes into one file, and progress lives in
the browser's localStorage keyed by a hash of the plan, so re-opening the
same file resumes where you stopped and a different plan starts clean.

The UI is a horizontal deck of cards, one stop per card, with the next card
peeking in at the edge so you can always see what is coming. Everything a
writer needs comes from plansteps.build_stops(); this module only decides
how to say it.
"""

import hashlib
import json
import os

from lib import plansteps
from lib.planpage import TEMPLATE

import logging
logger = logging.getLogger('fieldplan')

FACTION_TINT = {
    'enl': {'accent': '#31d158', 'glow': 'rgba(49,209,88,.22)'},
    'res': {'accent': '#3aa8e8', 'glow': 'rgba(58,168,232,.22)'},
}

TRAVELNAME = {
    'walking': 'walk',
    'bicycling': 'ride',
    'transit': 'transit',
    'driving': 'drive',
}


def _key_action(keys):
    """What to do about keys at this stop. 'hack' is what the clock paid for."""
    want = keys['want']
    hack = keys['hack']

    if not want:
        # Every link into this portal happens on a later visit
        return {'k': 'keys', 'note': True,
                'txt': 'No keys needed yet',
                'sub': '%s wanted here on a later visit.' % plansteps.plural(keys['total'], 'key')}

    if not hack:
        # Deliberately not quoting the number carried: the run spends that
        # stock across visits, so the figure from the portal list is stale
        # by the second visit
        return {'k': 'keys', 'note': True,
                'txt': 'Keys already in hand',
                'sub': ('The one key needed here is already on you.' if want == 1
                        else 'All %d keys needed here are ones you carry.' % want)}

    if keys['lastvisit']:
        sub = 'Last time here, so get them all now.'
    elif keys['ensure'] < keys['total']:
        sub = '%d needed here in all; you come back for the other %d.' % (
            keys['total'], keys['total'] - keys['ensure'])
    else:
        sub = 'You pass through again, but these are all the keys it needs.'
    if hack < want:
        sub += ' The other %s you already carry.' % plansteps.plural(want - hack, 'key')

    return {'k': 'keys', 'txt': 'Hack for %s' % plansteps.plural(hack, 'key'), 'sub': sub}


def _actions(stop):
    """
    The stop as an ordered checklist, in the order you actually play it.

    Note this differs from the spreadsheet, which lists shields before the
    links. You put shields on when you are done with a portal, and the
    links leave from it, so links come first here.
    """
    acts = []
    if stop['is_waypoint']:
        acts.append({'k': 'arrive', 'txt': 'Arrive', 'sub': 'Waypoint, nothing to do here.'})
        return acts

    if stop['is_blocker']:
        acts.append({'k': 'blocker', 'txt': 'Destroy the blocker',
                     'sub': 'This portal is in the way of a link.'})
        return acts

    if stop['first_visit']:
        acts.append({'k': 'capture', 'txt': 'Capture and deploy',
                     'sub': 'First time at this portal.'})

    if stop['keys']:
        acts.append(_key_action(stop['keys']))

    for link in stop['links']:
        # What the link completes is on the chip beside the name, so the
        # second line is left for aiming: how far and which way
        acts.append({'k': 'link', 'txt': link['name'], 'sub': None,
                     'link': link['action'], 'fields': link['fields'],
                     'dist': link['dist'], 'bearing': round(link['bearing'])})

    if stop['shields']:
        acts.append({'k': 'shields',
                     'txt': 'Shield up',
                     'sub': 'Last time here. %s to protect.'
                            % plansteps.plural(stop['shields']['links'], 'link')})

    return acts


def build_plan(a, workplan, stats, faction, travelmode, title=None):
    stops = plansteps.build_stops(a, workplan, travelmode)

    jstops = []
    for idx, stop in enumerate(stops):
        acts = _actions(stop)
        for n, act in enumerate(acts):
            act['id'] = 's%d.%s%d' % (idx, act['k'][0], n)
        jstop = {
            'i': idx,
            'num': stop['num'],
            'node': stop['node'],
            'name': stop['name'],
            'lat': stop['lat'],
            'lng': stop['lng'],
            'kind': 'waypoint' if stop['is_waypoint'] else ('blocker' if stop['is_blocker']
                                                            else 'portal'),
            'first': stop['first_visit'],
            'map': stop['mapurl'],
            'acts': acts,
        }
        if stop['travel']:
            jstop['travel'] = {
                'd': stop['travel']['dist'],
                't': stop['travel']['time'],
                'nice': stop['travel']['nicedist'],
                'moved': stop['travel']['moved'],
                'b': round(stop['travel']['bearing']),
            }
        jstops.append(jstop)

    # Every link in plan order, so the map can fill in as you check them off
    links = []
    for idx, stop in enumerate(stops):
        for act, link in zip([x for x in jstops[idx]['acts'] if x['k'] == 'link'],
                             stop['links']):
            links.append({'id': act['id'], 'a': stop['node'], 'b': link['node'],
                          'tri': link['triangles']})

    portals = []
    for p in range(a.order()):
        lat, lng = plansteps.latlng(a, p)
        portals.append({'n': a.nodes[p]['name'], 'lat': lat, 'lng': lng,
                        'w': a.nodes[p].get('special') is not None})

    plan = {
        'title': title or 'Fieldplan',
        'faction': faction if faction in FACTION_TINT else 'enl',
        'mode': travelmode,
        'modename': TRAVELNAME.get(travelmode, travelmode),
        'stats': {
            'ap': int(stats['ap']),
            'km': round(float(stats['dist']) / 1000.0, 2),
            'sqkm': round(float(stats['area']) / 1000000.0, 2),
            'time': stats['nicetime'].rsplit(':', 1)[0],
            'traveltime': stats['nicetraveltime'].rsplit(':', 1)[0],
            'links': int(stats['links']),
            'fields': int(stats['fields']),
            'hs': int(stats['hs']),
            'appmin': int(stats['appmin']),
        },
        'portals': portals,
        'stops': jstops,
        'links': links,
    }

    # Progress is keyed on this, so it must change whenever the plan does
    # and must not change when only the wrapper does
    blob = json.dumps([jstops, links], sort_keys=True, separators=(',', ':'))
    plan['id'] = hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]
    return plan


def render(plan):
    tint = FACTION_TINT[plan['faction']]
    # Escaped so the blob can never close the script element early
    blob = (json.dumps(plan, separators=(',', ':'))
            .replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026'))
    return (TEMPLATE
            .replace('__ACCENT__', tint['accent'])
            .replace('__GLOW__', tint['glow'])
            .replace('__TITLE__', plan['title'].replace('&', '&amp;').replace('<', '&lt;'))
            .replace('__PLAN__', blob))


def write_workplan(filename, a, workplan, stats, faction, travelmode='walking'):
    plan = build_plan(a, workplan, stats, faction, travelmode,
                      title='%0.2f km · %s AP' % (stats['dist'] / 1000.0,
                                                       '{:,}'.format(stats['ap'])))
    with open(filename, 'w', encoding='utf-8') as fh:
        fh.write(render(plan))
    logger.info('Wrote a phone-friendly plan into %s', filename)
    logger.info('  %d stops, %d actions, no network needed to use it',
                len(plan['stops']), sum(len(s['acts']) for s in plan['stops']))
    return filename


def default_filename(args):
    """Where to put the HTML when the user passed the flag without a path."""
    if getattr(args, 'textfile', None):
        base, _ext = os.path.splitext(args.textfile)
        return base + '_plan.html'
    return 'fieldplan.html'
