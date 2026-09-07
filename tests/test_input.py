#!/usr/bin/env python3
"""
Tests for the input layer: duplicate portal removal, the optional
keys-in-hand field, and how those keys are spent across a plan.

Usage:
    python tests/test_input.py
"""
import os
import sys
import tempfile

os.environ['HOME'] = tempfile.mkdtemp(prefix='fieldplan-test-')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import logging  # noqa: E402
from lib import maxfield, text_interface  # noqa: E402

logging.getLogger('fieldplan').addHandler(logging.NullHandler())

URL = 'https://intel.ingress.com/intel?pll=%s'


def write(lines):
    fd, path = tempfile.mkstemp(suffix='.txt', prefix='portals-')
    with os.fdopen(fd, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return path


def check_dedupe_portals():
    fails = []
    cases = [
        # (input rows, expected surviving names)
        ([('A', '1,1'), ('B', '2,2')], ['A', 'B']),
        ([('A', '1,1'), ('B', '1,1')], ['A']),
        ([('A', '1,1'), ('B', '2,2'), ('C', '1,1'), ('D', '2,2')], ['A', 'B']),
        # whitespace around the coordinates must not hide a duplicate
        ([('A', '1,1'), ('B', ' 1,1 ')], ['A']),
        # the first entry wins, not the last
        ([('Keep', '1,1'), ('Drop', '1,1')], ['Keep']),
        ([], []),
    ]
    for rows, want in cases:
        got = [r[0] for r in maxfield.dedupe_portals(rows)]
        if got != want:
            fails.append('dedupe_portals(%r) kept %r, want %r' % (rows, got, want))

    # rows are passed through untouched, not rebuilt
    rows = [('A', '1,1', None, 3), ('B', '1,1', None, 9)]
    got = maxfield.dedupe_portals(rows)
    if got != [rows[0]]:
        fails.append('dedupe_portals did not pass the surviving row through: %r' % (got,))
    return fails


def check_keys_parsing():
    fails = []
    path = write([
        'Plain; ' + URL % '45.50,-73.50',
        'WithKeys; ' + URL % '45.51,-73.51' + '; 4',
        'ZeroKeys; ' + URL % '45.52,-73.52' + '; 0',
        'Spaced; ' + URL % '45.53,-73.53' + ' ;   7  ',
        'Negative; ' + URL % '45.54,-73.54' + '; -2',
        'Garbage; ' + URL % '45.55,-73.55' + '; SBUL',
        '#!b Blocker; ' + URL % '45.56,-73.56',
    ])
    try:
        portals, waypoints = text_interface.get_portals_from_file(path)
    finally:
        os.unlink(path)

    got = {r[0]: r[3] for r in portals}
    want = {'Plain': 0, 'WithKeys': 4, 'ZeroKeys': 0, 'Spaced': 7,
            'Negative': 0, 'Garbage': 0}
    if got != want:
        fails.append('parsed keys %r, want %r' % (got, want))
    if len(waypoints) != 1:
        fails.append('expected the blocker waypoint to survive, got %r' % (waypoints,))

    # and the graph has to carry them through
    maxfield.waypoint_graph = None
    maxfield.populate_graphs(portals, waypoints)
    graph_keys = {maxfield.portal_graph.nodes[i]['name']: maxfield.portal_graph.nodes[i]['keys']
                  for i in range(maxfield.portal_graph.order())}
    if graph_keys != want:
        fails.append('graph keys %r, want %r' % (graph_keys, want))
    return fails


def check_key_budget():
    """Keys in hand are a budget for the whole run, not for each visit."""
    fails = []
    # Visit 0 twice with a link to it in between, so it needs a key at each
    # visit: (p, q, fields). Portal 0 is linked to from 1 and from 2.
    workplan = [
        (0, None, 0),
        (1, 0, 0),
        (0, None, 0),
        (2, 0, 0),
    ]
    base, _ = maxfield.get_needed_keys(workplan)
    total_needed = sum(base)
    if total_needed != 2:
        fails.append('fixture should need 2 keys for portal 0, needs %d (%r)' % (total_needed, base))
        return fails

    cases = [
        # (keys in hand for portal 0, expected total keys still to hack)
        (0, 2),
        (1, 1),
        (2, 0),
        (5, 0),   # more than needed must not go negative
    ]
    for held, want in cases:
        keys_t = [held, 0, 0]
        needed, _ = maxfield.get_needed_keys(workplan, keys_t)
        got = sum(needed)
        if got != want:
            fails.append('with %d keys in hand the plan still needs %d hacked keys, want %d (%r)'
                         % (held, got, want, needed))
        if any(k < 0 for k in needed):
            fails.append('with %d keys in hand needkeys went negative: %r' % (held, needed))

    # One key must be spent at the earliest visit that needs it
    needed, _ = maxfield.get_needed_keys(workplan, [1, 0, 0])
    first = [i for i, v in enumerate(base) if v]
    if needed[first[0]] != base[first[0]] - 1:
        fails.append('the single key was not spent at the first visit that needed one: %r' % (needed,))

    # Keys for a portal nothing links to change nothing
    untouched, _ = maxfield.get_needed_keys(workplan, [0, 9, 9])
    if untouched != base:
        fails.append('keys on never-linked portals changed the plan: %r vs %r' % (untouched, base))

    # A single visit needing more keys than are in hand must spend only
    # what is there, not go negative
    single = [(0, None, 0), (1, 0, 0), (2, 0, 0)]
    full, _ = maxfield.get_needed_keys(single)
    if sum(full) != 2:
        fails.append('single-visit fixture should need 2 keys, needs %d (%r)' % (sum(full), full))
    partial, _ = maxfield.get_needed_keys(single, [1, 0, 0])
    if sum(partial) != 1:
        fails.append('one key against a 2-key visit should leave 1 to hack, left %d (%r)'
                     % (sum(partial), partial))
    if any(k < 0 for k in partial):
        fails.append('one key against a 2-key visit went negative: %r' % (partial,))
    return fails


def main():
    failed = False
    for name, check in (('dedupe_portals', check_dedupe_portals),
                        ('keys parsing', check_keys_parsing),
                        ('key budget', check_key_budget)):
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
