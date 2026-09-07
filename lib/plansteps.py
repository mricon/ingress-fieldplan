# -*- coding: utf-8 -*-
"""
The workplan as a list of stops, one per place the agent stands at.

make_workplan() returns a flat list of (p, q, f) tuples, and every output
format then has to walk it the same way: group the consecutive actions at
one location, look ahead to see how many keys that portal still needs and
whether we ever come back, decide whether a link completes a field. That
walk was copy-pasted into the Sheets writer and the text writer, and the
HTML writer would have made three. It lives here now.

build_stops() is deliberately presentation-free: no emoji, no markup, no
"1.2 km" strings. Writers format what they need from the numbers.
"""

import math

from lib import maxfield

TRAVELMOJI = {
    'walking': u"\U0001F6B6",
    'bicycling': u"\U0001F6B2",
    'transit': u"\U0001F68D",
    'driving': u"\U0001F697",
}

# Below this the plan does not consider it a move at all: you are close
# enough that the game lets you interact from where you already stand.
SAME_SPOT_M = 40


def plural(n, word):
    return '%d %s' % (n, word if n == 1 else word + 's')


def maps_url(pll, travelmode='walking'):
    return ('https://www.google.com/maps/dir/?api=1&destination=%s&travelmode=%s'
            % (pll, travelmode))


def nice_distance(dist):
    if dist >= 500:
        return '%0.1f km' % (dist / float(1000))
    return '%d m' % dist


def latlng(a, p):
    lat, lng = a.nodes[p]['pll'].split(',')
    return float(lat), float(lng)


def bearing(a, p, q):
    """Compass bearing from p to q, degrees clockwise from north."""
    lat1, lng1 = (math.radians(v) for v in latlng(a, p))
    lat2, lng2 = (math.radians(v) for v in latlng(a, q))
    dlng = lng2 - lng1
    y = math.sin(dlng) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlng)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def link_action(f):
    # How the plan has always labelled a link by what it completes
    if f > 1:
        return 'D'
    if f == 1:
        return 'F'
    return 'L'


def _key_counts(workplan, keys_t=None):
    """
    Per plan index, the key situation for the visit starting there:

    total:    every link still to be made into this portal.
    ensure:   only those made before we next come back, so it is what you
              must be holding when you walk away. Equal to total on the
              last visit, and zero when every link into the portal happens
              on a later visit.
    lastvisit:whether we ever come back.
    want:     what this visit needs, which is total on the last visit and
              ensure otherwise. Every writer needs the same distinction, so
              it is made once here rather than three times.
    hack:     how many you have to actually go and get, once the keys you
              already hold are taken off. Keys in hand are a budget for the
              whole run rather than per visit, so they are spent against
              the earliest visits that need them -- the same way
              maxfield.get_needed_keys() spends them when it works out how
              long the run takes. The two must agree: if the page says hack
              for four and the clock paid for two, the plan is lying about
              one of them. tests/test_htmlout.py holds them to it.
    """
    n = len(workplan)
    counts = dict()
    links_to = dict()      # portal -> links into it at indices > current
    at_next_visit = dict()  # portal -> links_to when its next visit began
    pending = dict()
    for idx in range(n - 1, -1, -1):
        p, q, f = workplan[idx]
        if idx == n - 1 or workplan[idx + 1][0] != p:
            # last action of a visit: everything after it is still to come
            total = links_to.get(p, 0)
            if p in at_next_visit:
                pending[p] = (total - at_next_visit[p], total, False)
            else:
                pending[p] = (total, total, True)
        if idx == 0 or workplan[idx - 1][0] != p:
            # first action of a visit
            ensure, total, last = pending[p]
            want = total if last else ensure
            counts[idx] = {'ensure': ensure, 'total': total, 'lastvisit': last,
                           'want': want, 'hack': want}
            at_next_visit[p] = links_to.get(p, 0)
        if q is not None:
            links_to[q] = links_to.get(q, 0) + 1

    if keys_t and any(keys_t):
        budget = dict()
        for idx in sorted(counts):
            c = counts[idx]
            if not c['hack']:
                continue
            p = workplan[idx][0]
            if p not in budget:
                budget[p] = keys_t[p]
            spend = min(budget[p], c['hack'])
            c['hack'] -= spend
            budget[p] -= spend

    return counts


def build_stops(a, workplan, travelmode='walking'):
    """
    Group a workplan into stops. Returns a list of dicts, one per arrival.

    Every stop carries where it is and how to get there; portal stops also
    carry the actions to perform once there, in the order to do them.
    """
    keys_t = [a.nodes[i].get('keys', 0) for i in range(a.order())]
    counts = _key_counts(workplan, keys_t)
    stops = []
    prev_p = None
    seen = set()

    for idx, (p, q, f) in enumerate(workplan):
        if p != prev_p:
            special = a.nodes[p].get('special')
            lat, lng = latlng(a, p)
            stop = {
                'num': len(stops) + 1,
                'node': int(p),
                'name': a.nodes[p]['name'],
                'pll': a.nodes[p]['pll'],
                'lat': lat,
                'lng': lng,
                'special': special,
                'is_waypoint': special in ('_w_start', '_w_end'),
                'is_blocker': special == '_w_blocker',
                'first_visit': p not in seen,
                'travel': None,
                'mapurl': maps_url(a.nodes[p]['pll'], travelmode),
                'keys': None,
                'shields': None,
                'links': [],
            }
            seen.add(p)

            if prev_p is not None:
                dist = int(maxfield.get_portal_distance(prev_p, p))
                stop['travel'] = {
                    'dist': dist,
                    'time': int(maxfield.get_portal_time(prev_p, p)),
                    'nicedist': nice_distance(dist),
                    'moved': dist > SAME_SPOT_M,
                    'bearing': bearing(a, prev_p, p),
                }

            count = counts[idx]
            if not stop['is_waypoint'] and not stop['is_blocker']:
                if count['total']:
                    stop['keys'] = dict(count, in_hand=a.nodes[p].get('keys', 0))
                if count['lastvisit']:
                    stop['shields'] = {
                        'links': a.out_degree(p) + a.in_degree(p),
                    }

            stops.append(stop)
            prev_p = p

        if q is None:
            continue

        stops[-1]['links'].append({
            'action': link_action(f),
            'node': int(q),
            'name': a.nodes[q]['name'],
            'pll': a.nodes[q]['pll'],
            'fields': int(f),
            'dist': int(maxfield.get_portal_distance(p, q, direct=True)),
            'bearing': bearing(a, p, q),
            'triangles': [[int(v) for v in t] for t in a.edges[p, q]['fields']] if f else [],
        })

    return stops
