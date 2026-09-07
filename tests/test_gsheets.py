#!/usr/bin/env python3
"""
Tests for the Sheets write path that do not need Google.

The pure helpers are checked directly; write_workplan is driven against a
recording fake in place of the discovery client, which is enough to catch
the duplicate-title crash (issues #26/#27) and to assert that the range we
send is valid A1 notation for the title we were given back.

Usage:
    python tests/test_gsheets.py
"""
import os
import sys
import tempfile

os.environ['HOME'] = tempfile.mkdtemp(prefix='fieldplan-test-')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import logging  # noqa: E402
from lib import gsheets  # noqa: E402

logging.getLogger('fieldplan').addHandler(logging.NullHandler())

LONG = 'x' * 120


def check_a1_quote():
    fails = []
    cases = [
        ('Sheet1', "'Sheet1'"),
        ('🚶 1:23:45 (5.20km/10P/12,345AP)', "'🚶 1:23:45 (5.20km/10P/12,345AP)'"),
        ("Bob's plan", "'Bob''s plan'"),
    ]
    for title, want in cases:
        got = gsheets.a1_quote(title)
        if got != want:
            fails.append('a1_quote(%r) is %r, want %r' % (title, got, want))
    return fails


def check_unique_sheet_title():
    fails = []
    plan = '🚶 1:23:45 (5.20km/10P/12,345AP)'
    cases = [
        # (title, existing, want)
        (plan, [], plan),
        (plan, ['Portals'], plan),
        (plan, ['Portals', plan], plan + ' #2'),
        (plan, [plan, plan + ' #2'], plan + ' #3'),
        (plan, [plan, plan + ' #2', plan + ' #4'], plan + ' #3'),
        # matched case-insensitively and ignoring surrounding space
        ('Plan', ['  plan  '], 'Plan #2'),
        # over-length titles are trimmed, with room kept for the suffix
        (LONG, [], 'x' * gsheets.SHEET_TITLE_MAXLEN),
        (LONG, ['x' * gsheets.SHEET_TITLE_MAXLEN], 'x' * (gsheets.SHEET_TITLE_MAXLEN - 3) + ' #2'),
    ]
    for title, existing, want in cases:
        got = gsheets.unique_sheet_title(title, existing)
        if got != want:
            fails.append('unique_sheet_title(%r, %r) is %r, want %r' % (title, existing, got, want))
        if len(got) > gsheets.SHEET_TITLE_MAXLEN:
            fails.append('unique_sheet_title(%r, %r) is %d chars, over the limit'
                         % (title, existing, len(got)))
        if got.strip().lower() in {t.strip().lower() for t in existing}:
            fails.append('unique_sheet_title(%r, %r) returned a taken title' % (title, existing))

    # Many collisions in a row still terminate and stay unique
    existing = [plan] + ['%s #%d' % (plan, n) for n in range(2, 40)]
    got = gsheets.unique_sheet_title(plan, existing)
    if got != plan + ' #40':
        fails.append('unique_sheet_title after 39 collisions is %r, want %r' % (got, plan + ' #40'))
    return fails


class FakeSheets(object):
    """Minimal stand-in for service.spreadsheets() that records requests."""

    def __init__(self, titles):
        self.titles = list(titles)
        self.ranges = []
        self.added = []

    # service.spreadsheets()
    def spreadsheets(self):
        return self

    def get(self, spreadsheetId=None, fields=None):
        return _Exec({'sheets': [{'properties': {'title': t}} for t in self.titles]})

    def batchUpdate(self, spreadsheetId=None, body=None):
        # values().batchUpdate() carries 'data' instead of 'requests'
        for update in body.get('data', []):
            self.ranges.append(update['range'])
        replies = []
        for req in body.get('requests', []):
            if 'addSheet' not in req:
                replies.append({})
                continue
            title = req['addSheet']['properties']['title']
            if title.strip().lower() in {t.strip().lower() for t in self.titles}:
                raise AssertionError('addSheet asked for a duplicate title: %r' % title)
            self.titles.append(title)
            self.added.append(title)
            replies.append({'addSheet': {'properties': {'sheetId': len(self.titles), 'title': title}}})
        return _Exec({'replies': replies})

    # service.spreadsheets().values()
    def values(self):
        return self

class _Exec(object):
    def __init__(self, result):
        self.result = result

    def execute(self):
        return self.result


def solved_plan():
    """A real graph + workplan, so write_workplan sees what it sees in anger."""
    import numpy as np
    from lib import maxfield, text_interface

    maxfield.capture_cache = dict()
    maxfield.dist_matrix = list()
    maxfield.time_matrix = list()
    maxfield.direct_dist_matrix = list()
    maxfield.active_graph = None
    maxfield.waypoint_graph = None
    maxfield.minap = None
    maxfield.maxmu = False
    maxfield.maxtime = None
    maxfield.capture_search_ms = 0

    portals, waypoints = text_interface.get_portals_from_file(
        os.path.join(HERE, 'fixtures', 'waypoints.txt'))
    maxfield.populate_graphs(portals, waypoints)
    maxfield.gen_distance_matrix(None)
    np.random.seed(1234)
    while True:
        a = maxfield.portal_graph.copy()
        if not maxfield.max_fields(a):
            continue
        for t in a.triangulation:
            t.markEdgesWithFields()
        maxfield.extend_graph_with_waypoints(a)
        maxfield.active_graph = a
        workplan, stats = maxfield.make_workplan(a)
        if workplan is None:
            continue
        a.orig_workplan = workplan
        a.fixes = []
        return a, workplan, stats


def check_write_workplan_twice():
    """Two identical plans into one spreadsheet: the second used to crash."""
    fails = []
    a, workplan, stats = solved_plan()
    fake = FakeSheets(['Portals'])
    for run in (1, 2):
        try:
            gsheets.write_workplan(fake, 'sheet-id', a, workplan, stats, 'enl')
        except Exception as exc:
            fails.append('write_workplan run %d raised %s: %s' % (run, type(exc).__name__, exc))
            return fails

    if len(fake.added) != 2:
        fails.append('expected 2 sheets added, got %r' % (fake.added,))
    elif fake.added[0] == fake.added[1]:
        fails.append('both runs used the same title %r' % fake.added[0])
    elif not fake.added[1].endswith(' #2'):
        fails.append('second title %r is not suffixed' % fake.added[1])

    for rng in fake.ranges:
        if not rng.startswith("'") or "'!" not in rng:
            fails.append('range %r is not single-quoted A1 notation' % rng)
    if len(fake.ranges) != 2:
        fails.append('expected 2 value ranges, got %r' % (fake.ranges,))
    return fails


def main():
    failed = False
    for name, check in (('a1_quote', check_a1_quote),
                        ('unique_sheet_title', check_unique_sheet_title),
                        ('write_workplan duplicate run', check_write_workplan_twice)):
        fails = check()
        for msg in fails:
            print('FAIL %s: %s' % (name, msg))
        if fails:
            failed = True
        else:
            print('ok   %s' % name)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
