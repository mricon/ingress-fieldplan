#!/usr/bin/env python3
"""
Golden-output regression test for the plan solver.

Runs a fixed number of seeded solver iterations against the fixture portal
lists and compares every (workplan, stats) pair against the recorded JSON.
Any change to the algorithm that alters results will fail this test;
re-record deliberately with --record after reviewing the diff.

Covers both the full-graph path and the --maxtime subset path. The subset
runs drive make_subset / add_subset_portal / make_subset_graph on a fixed
grow-and-restart schedule instead of fieldplan.py's adaptive one, so the
solver is exercised deterministically without pinning the search policy.
They also record the subset membership, so a renumbering bug shows up as a
subset diff rather than only as an unexplained plan diff.

Alongside the golden comparison, check_subset_invariants() asserts the
things golden files cannot explain: that subset-graph node ids translate
through 'pos' to the right distance matrix rows, and that blockers are
still charged in a subset (regression guard for d60c469).

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

# Read at import, before reset_state() overwrites them, so check_module_defaults()
# can test what a library caller who sets nothing actually gets
SHIPPED_DEFAULTS = {name: getattr(maxfield, name) for name in ('minap', 'maxmu', 'maxtime')}

FIXTURES = {
    'waypoints': ('waypoints.txt', 60),
    'rand30': ('rand30.txt', 40),
    # same portals as waypoints.txt, but with keys already in hand on three of
    # them, so the whole pipeline is pinned with the key budget in play
    'keys': ('keys.txt', 60),
}
# name: (fixture file, iterations, maxmu). maxmu picks the other branch in
# both make_subset (largest vs smallest seed triangle) and add_subset_portal
# (random with seen_subsets dedup vs nearest-by-distance), so both are here.
SUBSET_FIXTURES = {
    'subset_waypoints': ('waypoints.txt', 48, False),
    'subset_maxmu': ('rand30.txt', 48, True),
}
# Iterations between subset restarts; the subset grows by one portal in
# between. waypoints.txt has only 10 portals, so its cycle also exercises
# add_subset_portal running out of candidates.
SUBSET_GROW_CYCLE = 12
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
    maxfield._active_tables = (None, None, None, None)
    maxfield.minap = None
    maxfield.maxmu = False
    maxfield.maxtime = None
    # Time-limited local search is not reproducible; golden runs use greedy routes
    maxfield.capture_search_ms = 0


def solve_iteration(b, is_subset):
    if not maxfield.max_fields(b):
        return None, None
    for t in b.triangulation:
        t.markEdgesWithFields()
    maxfield.extend_graph_with_waypoints(b)
    maxfield.active_graph = b
    return maxfield.make_workplan(b, is_subset)


def run_fixture(filename, iterations):
    reset_state()
    portals, waypoints = text_interface.get_portals_from_file(os.path.join(HERE, 'fixtures', filename))
    maxfield.populate_graphs(portals, waypoints)
    maxfield.gen_distance_matrix(None)
    np.random.seed(SEED)
    results = []
    for _ in range(iterations):
        b = maxfield.portal_graph.copy()
        workplan, stats = solve_iteration(b, False)
        if workplan is None:
            results.append(None)
            continue
        # JSON round-trip so recorded and live values compare equal
        # (node ids may arrive as numpy ints, hence default=int)
        results.append(json.loads(json.dumps({'workplan': [list(w) for w in workplan], 'stats': stats},
                                             default=int)))
    return results


def run_subset_fixture(filename, iterations, maxmu):
    reset_state()
    maxfield.maxmu = maxmu
    portals, waypoints = text_interface.get_portals_from_file(os.path.join(HERE, 'fixtures', filename))
    maxfield.populate_graphs(portals, waypoints)
    maxfield.gen_distance_matrix(None)
    np.random.seed(SEED)
    subset = maxfield.make_subset(4)
    results = []
    for i in range(iterations):
        # active_graph must be dropped before touching the full-graph node
        # space again, or make_subset/add_subset_portal read distances
        # through the previous subset's 'pos' translation
        maxfield.active_graph = None
        b = maxfield.make_subset_graph(subset).copy()
        workplan, stats = solve_iteration(b, True)
        entry = {'subset': list(subset)}
        if workplan is None:
            entry['workplan'] = None
            entry['stats'] = None
        else:
            entry['workplan'] = [list(w) for w in workplan]
            entry['stats'] = stats
        results.append(entry)
        maxfield.active_graph = None
        if i % SUBSET_GROW_CYCLE == SUBSET_GROW_CYCLE - 1:
            subset = maxfield.make_subset(4, random_start=True)
        else:
            maxfield.add_subset_portal(subset)
    # node ids arrive as numpy ints from np.random.choice, hence default=int
    return json.loads(json.dumps(results, default=int))


def check_subset_invariants():
    """Assert what golden diffs can show but not explain. Returns failures."""
    fails = []
    reset_state()
    portals, waypoints = text_interface.get_portals_from_file(os.path.join(HERE, 'fixtures', 'waypoints.txt'))
    maxfield.populate_graphs(portals, waypoints)
    maxfield.gen_distance_matrix(None)

    # A subset that deliberately excludes portal 0, so subset ids and
    # full-graph ids cannot coincide and a missing 'pos' translation shows up
    subset = list(range(1, 6))
    b = maxfield.make_subset_graph(list(subset)).copy()
    maxfield.extend_graph_with_waypoints(b)
    maxfield.active_graph = b

    if [b.nodes[i]['pos'] for i in range(len(subset))] != subset:
        fails.append('make_subset_graph lost the pos mapping')
    for i in range(b.order()):
        if b.nodes[i]['pll'] != maxfield.combined_graph.nodes[b.nodes[i]['pos']]['pll']:
            fails.append('subset node %d points at the wrong combined_graph node' % i)
            break

    # The hot loops read these tables by active-graph node id, so every cell
    # has to be the full-graph cell its endpoints' 'pos' values name
    dist, tim, blocker, keys = maxfield.get_active_tables()
    for name, table, master in (('dist', dist, maxfield.dist_matrix), ('time', tim, maxfield.time_matrix)):
        bad = [(i, j) for i in range(b.order()) for j in range(b.order())
               if table[i][j] != int(master[b.nodes[i]['pos']][b.nodes[j]['pos']])]
        if bad:
            i, j = bad[0]
            fails.append('subset %s[%d][%d] is %s, want %s (%d cells wrong)'
                         % (name, i, j, table[i][j], master[b.nodes[i]['pos']][b.nodes[j]['pos']], len(bad)))

    # Take the blockers from the graph, not from the table under test, so the
    # charge check below stays meaningful when the table itself is wrong
    blockers = [i for i in range(b.order()) if b.nodes[i].get('special') == '_w_blocker']
    if not blockers:
        fails.append('waypoints.txt fixture no longer has a blocker to check')
        return fails
    if [i for i, v in enumerate(blocker) if v] != blockers:
        fails.append('blocker table does not match the subset graph (see d60c469)')

    # And that the blocker is actually charged the 3 min to destroy it
    others = [i for i in range(b.order()) if i not in blockers]
    # portaltimes is indexed by node id, not by position in the plan
    plan = [(others[0], others[1], 0), (blockers[0], None, 0)]
    charged = maxfield.get_workplan_stats(plan)['portaltimes'][blockers[0]]
    if charged != 4.0:
        # 0.5 to arrive + 0.5 to capture + 3.0 to destroy the blocker
        fails.append('blocker charged %s min, want 4.0 (see d60c469)' % charged)
    return fails


def check_plan_accounting():
    """Every link and field in the plan must be counted, and the graph's
    own bookkeeping must survive improve_workplan.

    Both of these were broken: get_workplan_stats skipped links whose
    target was portal 0 because it tested 'if not q', and improve_workplan
    renumbered only the first a.size() workplan entries, so edge 'order'
    came out with duplicates and holes.
    """
    fails = []
    reset_state()
    portals, waypoints = text_interface.get_portals_from_file(os.path.join(HERE, 'fixtures', 'waypoints.txt'))
    maxfield.populate_graphs(portals, waypoints)
    maxfield.gen_distance_matrix(None)
    np.random.seed(SEED)

    checked = 0
    for _ in range(12):
        b = maxfield.portal_graph.copy()
        workplan, stats = solve_iteration(b, False)
        if workplan is None:
            continue
        checked += 1

        links = [w for w in workplan if w[1] is not None]
        if stats['links'] != len(links):
            fails.append('stats counted %d links, plan has %d (portal 0 as a target?)'
                         % (stats['links'], len(links)))
        if stats['fields'] != sum(w[2] for w in workplan):
            fails.append('stats counted %d fields, plan has %d'
                         % (stats['fields'], sum(w[2] for w in workplan)))

        # AP is fully determined by the counts, so this catches a miscount
        # from either direction
        want_ap = (b.order() * maxfield.CAPTUREAP + stats['links'] * maxfield.LINKAP
                   + stats['fields'] * maxfield.FIELDAP)
        if stats['ap'] != want_ap:
            fails.append('ap is %d, but the counts imply %d' % (stats['ap'], want_ap))

        orders = sorted(b.edges[e]['order'] for e in b.edges())
        if orders != list(range(b.size())):
            fails.append('edge orders are not 0..%d after improve_workplan: %s'
                         % (b.size() - 1, orders))

        # A triangle is completed by exactly one link, so it must be
        # recorded on exactly one edge
        homes = dict()
        for e in b.edges():
            for t in b.edges[e]['fields']:
                key = tuple(sorted(int(v) for v in t))
                homes[key] = homes.get(key, 0) + 1
        doubled = [k for k, v in homes.items() if v > 1]
        if doubled:
            fails.append('triangles recorded on more than one edge: %s' % (doubled[:3],))
        if fails:
            break

    if not checked:
        fails.append('no plans were produced to check')
    return fails


def check_depends_model():
    """The dependency relation, and that the optimizer respects it."""
    fails = []
    reset_state()
    portals, waypoints = text_interface.get_portals_from_file(os.path.join(HERE, 'fixtures', 'waypoints.txt'))
    maxfield.populate_graphs(portals, waypoints)
    maxfield.gen_distance_matrix(None)
    np.random.seed(SEED)

    checked = 0
    for _ in range(12):
        b = maxfield.portal_graph.copy()
        workplan, stats = solve_iteration(b, False)
        if workplan is None:
            continue
        checked += 1
        depends = maxfield.get_link_depends(b)

        if not maxfield.workplan_is_valid(workplan, depends):
            fails.append('improve_workplan produced a plan that is not playable')
            break

        # Every field must have its two other edges recorded as dependencies
        for e in b.edges():
            for t in b.edges[e]['fields']:
                others = {frozenset(x) for x in maxfield.triangle_edges(b, t)} - {frozenset(e)}
                if not others <= depends[frozenset(e)]:
                    fails.append('field %s on edge %s is missing dependencies' % (t, e))
                    break

        # The relation must actually bite: a plan with the last two links
        # swapped should usually be rejected, and reversing every link in
        # place must never break it (dependencies are undirected)
        flipped = [(q, p, f) if q is not None else (p, q, f) for p, q, f in workplan]
        for p, q, f in flipped:
            if q is not None and frozenset((p, q)) not in depends:
                fails.append('flipping the plan lost edge %s from the relation' % ((p, q),))
                break
        if fails:
            break

    if not checked:
        fails.append('no plans were produced to check')
        return fails

    # A hand-built plan that breaks a dependency must be rejected
    depends = {frozenset((0, 1)): set(), frozenset((1, 2)): set(),
               frozenset((0, 2)): {frozenset((0, 1)), frozenset((1, 2))}}
    good = [(0, None, 0), (1, None, 0), (2, None, 0), (0, 1, 0), (1, 2, 0), (0, 2, 1)]
    bad = [(0, None, 0), (1, None, 0), (2, None, 0), (0, 1, 0), (0, 2, 1), (1, 2, 0)]
    uncaptured = [(0, None, 0), (0, 1, 0)]
    toomany = [(0, None, 0)] + [(1, None, 0)] + [(0, 1, 0)] * 9
    for label, plan, want in (('a valid plan', good, True),
                              ('a field completed too early', bad, False),
                              ('a link to an uncaptured portal', uncaptured, False),
                              ('more than 8 outbound links', toomany, False)):
        got = maxfield.workplan_is_valid(plan, depends)
        if got != want:
            fails.append('workplan_is_valid said %s for %s' % (got, label))
    return fails


def check_module_defaults():
    """A library caller who configures nothing must still get the whole algorithm.

    Only the defaults that fail *silently* are worth checking here. A bad
    'cooling' or 'travelmode' already dies with a KeyError in the fixtures
    above; minap = np.inf did not, and quietly skipped capture routing for
    years.
    """
    fails = []
    reset_state()
    portals, waypoints = text_interface.get_portals_from_file(os.path.join(HERE, 'fixtures', 'waypoints.txt'))
    maxfield.populate_graphs(portals, waypoints)
    maxfield.gen_distance_matrix(None)

    plans = dict()
    for label, minap in (('shipped default', SHIPPED_DEFAULTS['minap']), ('explicit None', None)):
        maxfield.minap = minap
        maxfield.capture_cache = dict()
        maxfield.active_graph = None
        np.random.seed(SEED)
        b = maxfield.portal_graph.copy()
        workplan, stats = solve_iteration(b, False)
        if workplan is None:
            fails.append('%s produced no workplan at all' % label)
            return fails
        plans[label] = (workplan, stats)
        # A bare linkplan has a target on every step; a real workplan starts
        # with the capture route, whose steps have q None
        if not [w for w in workplan if w[1] is None]:
            fails.append('%s (minap=%r) returned a linkplan with no capture steps -- '
                         'make_workplan bailed at the minap guard' % (label, minap))
        if not hasattr(b, 'captureplan'):
            fails.append('%s (minap=%r) never ran the capture routing' % (label, minap))

    if plans['shipped default'] != plans['explicit None']:
        fails.append('the shipped minap default does not behave like None')

    # maxmu and maxtime silently change what is being optimised for, rather
    # than failing, so they have to default to off
    for name in ('maxmu', 'maxtime'):
        if SHIPPED_DEFAULTS[name] not in (None, False):
            fails.append('default %s is %r, expected off' % (name, SHIPPED_DEFAULTS[name]))
    return fails


def main():
    record = '--record' in sys.argv
    failed = False
    os.makedirs(os.path.join(HERE, 'golden'), exist_ok=True)
    for name, (filename, iterations) in FIXTURES.items():
        golden_path = os.path.join(HERE, 'golden', name + '.json')
        results = run_fixture(filename, iterations)
        if record:
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

    for name, (filename, iterations, maxmu) in SUBSET_FIXTURES.items():
        golden_path = os.path.join(HERE, 'golden', name + '.json')
        results = run_subset_fixture(filename, iterations, maxmu)
        if record:
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

    if not record:
        for label, check in (('subset invariants', check_subset_invariants),
                             ('plan accounting', check_plan_accounting),
                             ('depends model', check_depends_model),
                             ('module defaults', check_module_defaults)):
            fails = check()
            for msg in fails:
                print('FAIL %s: %s' % (label, msg))
            if fails:
                failed = True
            else:
                print('ok   %s' % label)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
