#!/usr/bin/env python3
"""
Golden-output regression test for the plan solver.

Runs a fixed number of seeded solver iterations against the fixture portal
lists and compares every (workplan, stats) pair against the recorded JSON.
Any change to the algorithm that alters results will fail this test;
re-record deliberately with --record after reviewing the diff.

Usage:
    python tests/test_golden.py            # verify
    python tests/test_golden.py --record   # re-record golden files
"""
import json
import logging
import os
import sys
import tempfile

# Keep the distance cache out of the user's real ~/.cache
os.environ['HOME'] = tempfile.mkdtemp(prefix='fieldplan-test-')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
from lib import maxfield, text_interface  # noqa: E402

logging.getLogger('fieldplan').addHandler(logging.NullHandler())

FIXTURES = {
    'waypoints': ('waypoints.txt', 60),
    'rand30': ('rand30.txt', 40),
}
SEED = 1234


def reset_state():
    maxfield.capture_cache = dict()
    maxfield.dist_matrix = list()
    maxfield.time_matrix = list()
    maxfield.direct_dist_matrix = list()
    maxfield.smallest_triangle = None
    maxfield.largest_triangle = None
    maxfield.seen_subsets = list()
    maxfield.active_graph = None
    maxfield.waypoint_graph = None
    maxfield.minap = None


def run_fixture(filename, iterations):
    reset_state()
    portals, waypoints = text_interface.get_portals_from_file(os.path.join(HERE, 'fixtures', filename))
    maxfield.populate_graphs(portals, waypoints)
    maxfield.gen_distance_matrix(None)
    np.random.seed(SEED)
    results = []
    for _ in range(iterations):
        b = maxfield.portal_graph.copy()
        if not maxfield.max_fields(b):
            results.append(None)
            continue
        for t in b.triangulation:
            t.markEdgesWithFields()
        maxfield.extend_graph_with_waypoints(b)
        maxfield.active_graph = b
        workplan, stats = maxfield.make_workplan(b, False)
        if workplan is None:
            results.append(None)
            continue
        # JSON round-trip so recorded and live values compare equal
        # (node ids may arrive as numpy ints, hence default=int)
        results.append(json.loads(json.dumps({'workplan': [list(w) for w in workplan], 'stats': stats},
                                             default=int)))
    return results


def main():
    record = '--record' in sys.argv
    failed = False
    for name, (filename, iterations) in FIXTURES.items():
        golden_path = os.path.join(HERE, 'golden', name + '.json')
        results = run_fixture(filename, iterations)
        if record:
            os.makedirs(os.path.dirname(golden_path), exist_ok=True)
            with open(golden_path, 'w') as fh:
                json.dump(results, fh, indent=1, sort_keys=True)
            print('recorded %s (%d iterations)' % (golden_path, iterations))
            continue
        with open(golden_path) as fh:
            golden = json.load(fh)
        bad = [i for i, (g, r) in enumerate(zip(golden, results)) if g != r]
        if bad or len(golden) != len(results):
            failed = True
            print('FAIL %s: %d/%d iterations differ (first: %s)' % (name, len(bad), len(golden), bad[:5]))
        else:
            print('ok   %s: %d/%d iterations identical' % (name, len(results), len(golden)))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
