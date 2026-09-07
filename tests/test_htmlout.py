#!/usr/bin/env python3
"""
Tests for the stop model and the HTML plan.

The model checks are cross-checked against maxfield itself: the key counts
the page shows have to agree with the ones the time estimate was built on,
or the plan tells you to hack for keys the clock never paid for.

The page checks end with a real browser when Playwright and its Chromium
are installed (pip install playwright && playwright install chromium).
Without them that part is skipped and says so -- everything above it still
runs, but nothing has then proved the page actually works.

Usage:
    python tests/test_htmlout.py
"""
import json
import os
import re
import sys
import tempfile

# test_gsheets repoints HOME at a temp dir on import, and that is where
# Playwright looks for its browsers. Pin the cache before that happens.
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH',
                      os.path.join(os.path.expanduser('~'), '.cache', 'ms-playwright'))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import logging  # noqa: E402
logging.getLogger('fieldplan').addHandler(logging.NullHandler())
logging.getLogger('fieldplan').setLevel(logging.CRITICAL)

from test_gsheets import solved_plan  # noqa: E402  (also pins HOME to a temp dir)
from lib import htmlout, maxfield, plansteps  # noqa: E402

_plans = dict()


def plan(fixture='waypoints.txt'):
    # Solving mutates maxfield's module globals, so a cached plan is only
    # usable once its graph is active again
    if fixture not in _plans:
        _plans[fixture] = solved_plan(fixture)
    a, workplan, stats = _plans[fixture]
    maxfield.active_graph = a
    return a, workplan, stats


def check_stops_cover_the_workplan():
    """Every action in the workplan shows up exactly once, in order."""
    fails = []
    a, workplan, stats = plan()
    stops = plansteps.build_stops(a, workplan)

    want_stops = sum(1 for i, (p, q, f) in enumerate(workplan)
                     if i == 0 or workplan[i - 1][0] != p)
    if len(stops) != want_stops:
        fails.append('built %d stops for %d arrivals' % (len(stops), want_stops))

    got = [(s['node'], link['node'], link['fields'])
           for s in stops for link in s['links']]
    want = [(p, q, f) for p, q, f in workplan if q is not None]
    if got != want:
        first = next((i for i in range(max(len(got), len(want)))
                      if got[i:i+1] != want[i:i+1]), 0)
        fails.append('links differ from the workplan at %d: %r, want %r'
                     % (first, got[first:first+1], want[first:first+1]))

    if stops and stops[0]['travel'] is not None:
        fails.append('the first stop has a travel leg into it')
    for s in stops[1:]:
        if s['travel'] is None:
            fails.append('stop %d has no travel leg' % s['num'])

    for s in stops:
        if not (s['is_waypoint'] or s['is_blocker']):
            continue
        for field in ('keys', 'shields'):
            if s[field]:
                fails.append('stop %d is not a portal but carries %s' % (s['num'], field))
        if s['links']:
            fails.append('stop %d is not a portal but makes links' % s['num'])

    seen = set()
    for s in stops:
        if s['first_visit'] != (s['node'] not in seen):
            fails.append('stop %d has first_visit=%s' % (s['num'], s['first_visit']))
        seen.add(s['node'])
    return fails


def check_keys_match_the_time_model():
    """
    The displayed key counts and the ones the clock was built on must agree.

    maxfield.get_needed_keys() drives how long the plan says hacking takes.
    If the page shows a different number, the plan is lying about one or
    the other. Compared with no keys in hand, since the model reports the
    raw counts and nets the ones you hold out separately.
    """
    fails = []
    # keys.txt carries keys in hand, so the netting path gets exercised too
    for fixture in ('waypoints.txt', 'keys.txt'):
        a, workplan, stats = plan(fixture)
        keys_t = [a.nodes[i].get('keys', 0) for i in range(a.order())]
        counts = plansteps._key_counts(workplan, keys_t)

        for raw, table in ((True, None), (False, keys_t)):
            needkeys, lastvisit = maxfield.get_needed_keys(workplan, table)
            bare = plansteps._key_counts(workplan, None) if raw else counts
            for idx in sorted(bare):
                c = bare[idx]
                if c['lastvisit'] != lastvisit[idx]:
                    fails.append('%s index %d: lastvisit %s here, %s in maxfield'
                                 % (fixture, idx, c['lastvisit'], lastvisit[idx]))
                if c['hack'] != needkeys[idx]:
                    fails.append('%s index %d: hack for %d here, %d in maxfield%s'
                                 % (fixture, idx, c['hack'], needkeys[idx],
                                    '' if raw else ' (with keys in hand)'))

        for idx in sorted(counts):
            c = counts[idx]
            if c['ensure'] > c['total']:
                fails.append('%s index %d: ensure %d is over the total %d'
                             % (fixture, idx, c['ensure'], c['total']))
            if c['lastvisit'] and c['ensure'] != c['total']:
                fails.append('%s index %d: last visit wants %d of %d'
                             % (fixture, idx, c['ensure'], c['total']))
            if c['want'] != (c['total'] if c['lastvisit'] else c['ensure']):
                fails.append('%s index %d: want %d, but the visit needs %d'
                             % (fixture, idx, c['want'],
                                c['total'] if c['lastvisit'] else c['ensure']))
            if c['hack'] > c['want']:
                fails.append('%s index %d: hacking for %d when %d is wanted'
                             % (fixture, idx, c['hack'], c['want']))
            # Independently: total is every link still to be made into that portal
            p = workplan[idx][0]
            after = sum(1 for w in workplan[idx:] if w[1] == p)
            if c['total'] != after:
                fails.append('%s index %d: total %d, but %d links into %s remain'
                             % (fixture, idx, c['total'], after, p))

        # Keys in hand must be spent once over the run, not once per visit
        if any(keys_t):
            for p in {w[0] for w in workplan}:
                idxs = [i for i in counts if workplan[i][0] == p]
                need = sum(counts[i]['want'] for i in idxs)
                hack = sum(counts[i]['hack'] for i in idxs)
                if need - hack > keys_t[p]:
                    fails.append('%s portal %s: credited %d keys in hand, only %d held'
                                 % (fixture, p, need - hack, keys_t[p]))
    return fails


def check_key_budget_across_visits():
    """
    Keys in hand are a budget for the whole run, not a refill at each visit.

    No fixture happens to need keys for one portal on two separate visits,
    so this builds that case by hand: portal 0 is linked to once before we
    come back to it and once after, and we hold a single key. The first
    visit spends it; the second still has to hack.
    """
    fails = []
    workplan = [(0, None, 0), (1, None, 0), (1, 0, 0),
                (0, None, 0), (2, None, 0), (2, 0, 0)]
    counts = plansteps._key_counts(workplan, [1, 0, 0])
    want = {0: {'ensure': 1, 'total': 2, 'lastvisit': False, 'want': 1, 'hack': 0},
            3: {'ensure': 1, 'total': 1, 'lastvisit': True, 'want': 1, 'hack': 1}}
    for idx in sorted(want):
        if idx not in counts:
            fails.append('no visit recorded at index %d' % idx)
        elif counts[idx] != want[idx]:
            fails.append('index %d is %r, want %r' % (idx, counts[idx], want[idx]))
    for idx in counts:
        if idx not in want and counts[idx]['hack']:
            fails.append('index %d wants keys and should not' % idx)

    # And maxfield, which prices the hacking, must reach the same numbers
    needkeys, _lastvisit = maxfield.get_needed_keys(workplan, [1, 0, 0])
    for idx in sorted(counts):
        if counts[idx]['hack'] != needkeys[idx]:
            fails.append('index %d: hack for %d here, %d in maxfield'
                         % (idx, counts[idx]['hack'], needkeys[idx]))
    return fails


def check_plan_json():
    fails = []
    a, workplan, stats = plan()
    p = htmlout.build_plan(a, workplan, stats, 'enl', 'walking')

    json.dumps(p)  # raises on anything numpy left behind

    ids = [act['id'] for s in p['stops'] for act in s['acts']]
    if len(ids) != len(set(ids)):
        fails.append('action ids are not unique (%d ids, %d distinct)' % (len(ids), len(set(ids))))

    link_ids = {act['id'] for s in p['stops'] for act in s['acts'] if act['k'] == 'link'}
    if link_ids != {l['id'] for l in p['links']}:
        fails.append('the map link list does not match the link actions')

    top = len(p['portals']) - 1
    for l in p['links']:
        for v in [l['a'], l['b']] + [w for t in l['tri'] for w in t]:
            if not 0 <= v <= top:
                fails.append('link refers to portal %d, only %d exist' % (v, top + 1))
    for s in p['stops']:
        if not 0 <= s['node'] <= top:
            fails.append('stop %d refers to portal %d' % (s['num'], s['node']))

    # Progress is keyed on the id, so it has to track the plan and only the plan
    again = htmlout.build_plan(a, workplan, stats, 'enl', 'walking')
    if again['id'] != p['id']:
        fails.append('the same plan hashed to two different ids')
    retitled = htmlout.build_plan(a, workplan, stats, 'res', 'walking', title='Other')
    if retitled['id'] != p['id']:
        fails.append('the id changed when only the title and faction did')
    shorter = htmlout.build_plan(a, workplan[:-1], stats, 'enl', 'walking')
    if shorter['id'] == p['id']:
        fails.append('a different workplan hashed to the same id')
    return fails


def check_page_is_self_contained():
    """
    Nothing may be fetched. The plan gets used in parks with no signal, and
    a page that quietly needs a CDN works on the desk and fails on the walk.
    """
    fails = []
    a, workplan, stats = plan()
    page = htmlout.render(htmlout.build_plan(a, workplan, stats, 'enl', 'walking'))

    for token in ('__PLAN__', '__ACCENT__', '__GLOW__', '__TITLE__'):
        if token in page:
            fails.append('%s was never substituted' % token)

    allowed = ('https://www.google.com/maps/',   # navigation, opened by a tap
               'http://www.w3.org/2000/svg')     # a namespace name, never fetched
    for url in set(re.findall(r'https?://[^\s"\'<>)\\]+', page)):
        if not url.startswith(allowed):
            fails.append('page refers to %s' % url)

    for pat in (r'<link\b', r'<script[^>]+\bsrc=', r'@import', r'url\(\s*[\'"]?http'):
        if re.search(pat, page, re.I):
            fails.append('page pulls in an external resource (%s)' % pat)

    blob = re.search(r'id="plandata">(.*?)</script>', page, re.S)
    if not blob:
        fails.append('the plan data is not in the page')
    elif '<' in blob.group(1):
        fails.append('the plan blob has a raw "<" and can close the script early')
    else:
        json.loads(blob.group(1))
    return fails


def check_sheet_and_page_agree():
    """The refactor put both writers on one model; they must still say the same."""
    fails = []
    from test_gsheets import FakeSheets
    from lib import gsheets

    a, workplan, stats = plan()
    rows = []

    class Rec(FakeSheets):
        def batchUpdate(self, spreadsheetId=None, body=None):
            for u in body.get('data', []):
                rows.extend([list(r) for r in u['values']])
            return FakeSheets.batchUpdate(self, spreadsheetId=spreadsheetId, body=body)

    gsheets.write_workplan(Rec(['Portals']), 'sid', a, workplan, stats, 'enl')
    sheet_links = [(r[0], r[1][1:]) for r in rows if len(r) > 1 and r[0] in ('L', 'F', 'D')]
    stops = plansteps.build_stops(a, workplan)
    page_links = [(l['action'], l['name']) for s in stops for l in s['links']]
    if sheet_links != page_links:
        fails.append('sheet lists %d links, the model lists %d, and they differ'
                     % (len(sheet_links), len(page_links)))
    return fails


def check_writers_net_keys_in_hand():
    """
    All three writers must count the keys you already hold.

    The time model has netted them since 5587e4c, but the Sheets and text
    writers went on printing the raw counts, so a plan could tell you to
    farm five keys that were in your pocket. Driven on keys.txt, which is
    the only fixture with keys in hand, and it fails if that fixture ever
    stops exercising the netting rather than quietly passing.
    """
    fails = []
    from test_gsheets import FakeSheets
    from lib import gsheets, text_interface

    a, workplan, stats = plan('keys.txt')
    stops = [s for s in plansteps.build_stops(a, workplan) if s['keys']]
    covered = [s for s in stops if s['keys']['want'] and not s['keys']['hack']]
    hacked = [s for s in stops if s['keys']['hack']]
    if not covered:
        fails.append('keys.txt no longer has a visit fully covered by keys in hand, '
                     'so nothing here tests the netting')
    if not hacked:
        fails.append('keys.txt no longer has a visit that needs hacking')

    rows = []

    class Rec(FakeSheets):
        def batchUpdate(self, spreadsheetId=None, body=None):
            for u in body.get('data', []):
                rows.extend([list(r) for r in u['values']])
            return FakeSheets.batchUpdate(self, spreadsheetId=spreadsheetId, body=body)

    gsheets.write_workplan(Rec(['Portals']), 'sid', a, workplan, stats, 'enl')
    sheet = [r[1] for r in rows if len(r) > 1 and r[0] == 'H']

    tmp = tempfile.mkdtemp(prefix='fieldplan-text-')
    src = os.path.join(tmp, 'p.txt')
    open(src, 'w').close()
    text_interface.write_workplan(src, a, workplan, stats, 'enl')
    with open(os.path.join(tmp, 'p_plan.txt'), encoding='utf-8') as fh:
        text = [l.strip()[4:] for l in fh if l.strip().startswith('[H]')]

    fails += _writers_say(a, workplan, stats, stops, sheet, text)

    # keys.txt only ever covers a visit fully or not at all, so on its own it
    # cannot tell 'hack' from 'want' in the last-visit line. Rather than churn
    # the recorded golden by editing the fixture, hold one portal's keys just
    # short of what a visit needs and re-run the writers over the same plan.
    short_by_one = dict()
    for s in stops:
        if s['keys']['want'] > 1 and not s['keys']['in_hand']:
            short_by_one.setdefault(s['node'], s['keys']['want'] - 1)
    if not short_by_one:
        fails.append('no visit in keys.txt needs two or more keys with none in hand, '
                     'so partial cover is untested')
    else:
        was = {n: a.nodes[n]['keys'] for n in short_by_one}
        try:
            for n, held in short_by_one.items():
                a.nodes[n]['keys'] = held
            short = [s for s in plansteps.build_stops(a, workplan) if s['keys']]
            # Both branches of the key line have to see a partly covered visit,
            # or swapping 'hack' for the raw count goes unnoticed in one of them
            for last in (True, False):
                if not [s for s in short if s['keys']['lastvisit'] is last
                        and 0 < s['keys']['hack'] < s['keys']['want']]:
                    fails.append('no %s visit is partly covered by keys in hand'
                                 % ('last' if last else 'return'))
            rows[:] = []
            gsheets.write_workplan(Rec(['Portals']), 'sid', a, workplan, stats, 'enl')
            psheet = [r[1] for r in rows if len(r) > 1 and r[0] == 'H']
            text_interface.write_workplan(src, a, workplan, stats, 'enl')
            with open(os.path.join(tmp, 'p_plan.txt'), encoding='utf-8') as fh:
                ptext = [l.strip()[4:] for l in fh if l.strip().startswith('[H]')]
            fails += _writers_say(a, workplan, stats, short, psheet, ptext)
        finally:
            for n, held in was.items():
                a.nodes[n]['keys'] = held
    return fails


def _writers_say(a, workplan, stats, stops, sheet, text):
    fails = []
    for name, lines in (('sheet', sheet), ('text', text)):
        if len(lines) != len(stops):
            fails.append('%s wrote %d key lines for %d stops that need keys'
                         % (name, len(lines), len(stops)))
            continue
        for line, stop in zip(lines, stops):
            k = stop['keys']
            asked = re.search(r'(?:ensure|Ensure) (\d+)', line)
            if not k['want']:
                continue  # nothing due yet; both writers report the later max
            if not k['hack']:
                if asked:
                    fails.append('%s at %s: %r asks for keys already in hand'
                                 % (name, stop['name'], line))
                elif 'in hand' not in line and 'carrying' not in line:
                    fails.append('%s at %s: %r does not say the keys are held'
                                 % (name, stop['name'], line))
            elif not asked:
                fails.append('%s at %s: %r names no number, %d must be hacked'
                             % (name, stop['name'], line, k['hack']))
            elif int(asked.group(1)) != k['hack']:
                fails.append('%s at %s: %r asks for %s, %d must be hacked'
                             % (name, stop['name'], line, asked.group(1), k['hack']))
    return fails


def check_in_a_browser():
    """Load the page in Chromium and use it the way a thumb would."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    a, workplan, stats = plan()
    page = htmlout.render(htmlout.build_plan(a, workplan, stats, 'enl', 'walking'))
    tmp = tempfile.mkdtemp(prefix='fieldplan-page-')
    path = os.path.join(tmp, 'plan.html')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(page)

    fails = []
    try:
        pw = sync_playwright().start()
    except Exception as exc:
        print('     (%s)' % exc)
        return None
    try:
        try:
            browser = pw.chromium.launch()
        except Exception as exc:
            print('     (%s)' % str(exc).splitlines()[0])
            return None
        pg = browser.new_page(viewport={'width': 390, 'height': 844},
                              is_mobile=True, has_touch=True)
        pg.on('pageerror', lambda e: fails.append('uncaught js error: %s' % e))
        pg.on('console', lambda m: fails.append('console error: %s' % m.text)
              if m.type == 'error' else None)
        pg.goto('file://' + path)
        pg.wait_for_timeout(500)

        probe = """() => {
          const deck = document.getElementById('deck');
          const cards = [...deck.children];
          const mid = deck.scrollLeft + deck.clientWidth / 2;
          let best = 0, off = 1e9;
          cards.forEach((c, i) => {
            const d = Math.abs(c.offsetLeft + c.clientWidth / 2 - mid);
            if (d < off) { off = d; best = i; }
          });
          return {cards: cards.length, at: best, off: Math.round(off),
                  scroll: Math.round(deck.scrollLeft),
                  fill: document.getElementById('fill').style.width,
                  spill: document.body.scrollWidth - document.body.clientWidth,
                  parented: cards.every(c => c.offsetParent === deck),
                  acts: document.querySelectorAll('.act').length};
        }"""

        st = pg.evaluate(probe)
        nstops = len(plansteps.build_stops(a, workplan))
        if st['cards'] != nstops + 1:
            fails.append('%d cards for %d stops plus a finish card' % (st['cards'], nstops))
        if st['off'] > 3:
            fails.append('the opening card sits %dpx off centre' % st['off'])
        if st['scroll'] != 0:
            fails.append('the first card should rest at scroll 0, it rests at %d' % st['scroll'])
        if st['spill'] > 0:
            fails.append('the page scrolls sideways by %dpx' % st['spill'])
        if not st['acts']:
            fails.append('no action rows rendered')
        if not st['parented']:
            fails.append('cards do not offset from the deck, so centring drifts '
                         'as soon as an ancestor is inset (safe areas, landscape)')

        # A part-finished stop must not read as finished
        partial = pg.evaluate('''() => {
            const cards = [...document.querySelectorAll('.deck > .card')];
            for (const c of cards){
              const acts = [...c.querySelectorAll('.act:not(.note)')];
              if (acts.length < 2) continue;
              acts[0].click();
              const early = c.classList.contains('done');
              acts.slice(1).forEach(a => a.click());
              const late = c.classList.contains('done');
              acts.forEach(a => a.click());
              return {early: early, late: late};
            }
            return null;
        }''')
        pg.wait_for_timeout(200)
        if partial is None:
            fails.append('no stop has two actions to check partial completion with')
        else:
            if partial['early']:
                fails.append('a stop with one of several actions ticked reads as done')
            if not partial['late']:
                fails.append('a stop with every action ticked does not read as done')

        # Tick everything on the second stop and check it reads as complete
        n = pg.evaluate("""() => {
            const c = document.querySelectorAll('.deck > .card')[1];
            const acts = [...c.querySelectorAll('.act:not(.note)')];
            acts.forEach(a => a.click());
            return acts.length;
        }""")
        pg.wait_for_timeout(200)
        if n and not pg.evaluate(
                "() => document.querySelectorAll('.deck > .card')[1].classList.contains('done')"):
            fails.append('a stop with every action ticked is not marked done')
        if pg.evaluate("() => document.getElementById('fill').style.width") == '0%':
            fails.append('the progress bar did not move')

        # Paging lands a card centred
        pg.evaluate("() => document.getElementById('next').click()")
        pg.wait_for_timeout(600)
        st = pg.evaluate(probe)
        if st['at'] != 1:
            fails.append('Next landed on card %d, not card 1' % st['at'])
        if st['off'] > 3:
            fails.append('card 1 sits %dpx off centre' % st['off'])

        # The overview draws, including the map
        pg.evaluate("() => document.getElementById('list').click()")
        pg.wait_for_timeout(300)
        sh = pg.evaluate("""() => ({
            open: !document.getElementById('sheet').hidden,
            drawn: document.getElementById('map').childElementCount,
            rows: document.querySelectorAll('#olist li').length,
            ok: document.querySelectorAll('#olist li.ok').length})""")
        if not sh['open']:
            fails.append('the overview did not open')
        if sh['rows'] != nstops:
            fails.append('the overview lists %d of %d stops' % (sh['rows'], nstops))
        if sh['drawn'] < 5:
            fails.append('the map drew only %d shapes' % sh['drawn'])
        if n and not sh['ok']:
            fails.append('the overview shows no finished stop')
        pg.evaluate("() => document.getElementById('shclose').click()")

        # Progress survives being closed and reopened
        pg.reload()
        pg.wait_for_timeout(500)
        if n and not pg.evaluate(
                "() => document.querySelectorAll('.deck > .card')[1].classList.contains('done')"):
            fails.append('progress did not survive a reload')

        # A different plan must not inherit those ticks
        other = htmlout.render(htmlout.build_plan(a, workplan[:-1], stats, 'enl', 'walking'))
        other_path = os.path.join(tmp, 'other.html')
        with open(other_path, 'w', encoding='utf-8') as fh:
            fh.write(other)
        pg.goto('file://' + other_path)
        pg.wait_for_timeout(400)
        if pg.evaluate("() => document.getElementById('fill').style.width") not in ('0%', ''):
            fails.append("a different plan picked up the first one's progress")

        browser.close()
    finally:
        pw.stop()
    return fails


def main():
    checks = (
        ('stops cover the workplan', check_stops_cover_the_workplan),
        ('keys match the time model', check_keys_match_the_time_model),
        ('key budget across visits', check_key_budget_across_visits),
        ('plan json', check_plan_json),
        ('page is self-contained', check_page_is_self_contained),
        ('sheet and page agree', check_sheet_and_page_agree),
        ('writers net keys in hand', check_writers_net_keys_in_hand),
        ('page in a browser', check_in_a_browser),
    )
    failed = False
    for name, check in checks:
        fails = check()
        if fails is None:
            print('SKIP %s: install playwright and its chromium to run it' % name)
            continue
        for msg in fails:
            print('FAIL %s: %s' % (name, msg))
        if fails:
            failed = True
        else:
            print('ok   %s' % name)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
