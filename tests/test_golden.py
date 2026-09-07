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
import subprocess
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


# Single-core ms per solver iteration, measured 2026-09-07 over 30 burn-in
# runs (3 random layouts x 2 seeds x 1000 iterations at each size). The cost
# model in lib/maxfield.py is fitted to these; if it drifts away from them the
# scaled -i default stops meaning anything.
MEASURED_ITERATION_MS = {10: 3.5, 15: 10.5, 20: 23.1, 30: 70.0, 40: 157.7}


def check_default_iterations():
    """The -i default scales with portal count, off a fitted cost model."""
    fails = []

    for n, want in sorted(MEASURED_ITERATION_MS.items()):
        got = maxfield.estimate_iteration_ms(n)
        err = abs(got / want - 1.0)
        if err > 0.10:
            fails.append('cost model says %.1f ms at n=%d, measured %.1f (%.0f%% off)'
                         % (got, n, want, 100 * err))

    prev = 0
    for n in range(3, 120):
        ms = maxfield.estimate_iteration_ms(n)
        if ms < prev:
            fails.append('cost model is not monotonic at n=%d' % n)
            break
        prev = ms

    for n in range(0, 300):
        it = maxfield.default_iterations(n)
        if not maxfield.ITERATIONS_MIN <= it <= maxfield.ITERATIONS_MAX:
            fails.append('default_iterations(%d) is %d, outside [%d, %d]'
                         % (n, it, maxfield.ITERATIONS_MIN, maxfield.ITERATIONS_MAX))
            break
        if it % 500:
            fails.append('default_iterations(%d) is %d, not a round number' % (n, it))
            break

    # Cheap lists get searched harder, expensive ones sit on the floor
    if maxfield.default_iterations(8) != maxfield.ITERATIONS_MAX:
        fails.append('a small list should get the maximum, got %d' % maxfield.default_iterations(8))
    if maxfield.default_iterations(40) != maxfield.ITERATIONS_MIN:
        fails.append('a large list should get the floor, got %d' % maxfield.default_iterations(40))
    if maxfield.default_iterations(10) <= maxfield.default_iterations(30):
        fails.append('10 portals should get more iterations than 30')

    # Never increasing as the list grows, and degenerate input is safe
    prev = maxfield.ITERATIONS_MAX + 1
    for n in range(3, 300):
        it = maxfield.default_iterations(n)
        if it > prev:
            fails.append('default_iterations rose from %d to %d at n=%d' % (prev, it, n))
            break
        prev = it
    for n in (0, 1, 2):
        if maxfield.default_iterations(n) != maxfield.ITERATIONS_MIN:
            fails.append('default_iterations(%d) should fall back to the floor' % n)

    # The floor has to stay where the convergence data put it: 5,000 landed
    # within ~1-2% of a very long run at n=10, 15 and 20
    if maxfield.ITERATIONS_MIN < 5000:
        fails.append('the floor dropped below the measured 5000')
    return fails


def check_blocker_guard():
    """
    workplan_is_valid()'s blocker rule, tested where the route cannot hide it.

    The capture route now visits blockers first, and that alone keeps plans
    clean: delete the guard entirely and check_no_links_before_blockers()
    still passes, because improve_workplan() never gets near a state the
    guard would have to refuse. That makes the end-to-end check blind to
    the very rule it is named after, so the rule is pinned directly here --
    once as a predicate, and once by handing improve_workplan() a capture
    route with the blocker last, which is what the route used to produce.

    The route ordering is the optimisation; this is the correctness.
    """
    fails = []
    reset_state()
    portals, waypoints = text_interface.get_portals_from_file(
        os.path.join(HERE, 'fixtures', 'waypoints.txt'))
    maxfield.populate_graphs(portals, waypoints)
    maxfield.gen_distance_matrix(None)
    np.random.seed(SEED)

    b = maxfield.portal_graph.copy()
    if not maxfield.max_fields(b):
        return ['could not build a graph to check the guard with']
    for t in b.triangulation:
        t.markEdgesWithFields()
    maxfield.extend_graph_with_waypoints(b)
    maxfield.active_graph = b

    blockers = maxfield.get_blockers(b)
    if not blockers:
        return ['waypoints.txt no longer has a blocker to check']
    blk = sorted(blockers)[0]
    depends = maxfield.get_link_depends(b)

    linkplan = [None] * b.size()
    for p, q in b.edges():
        linkplan[b.edges[p, q]['order']] = (p, q, len(b.edges[p, q]['fields']))
    others = [(n, None, 0) for n in range(b.order()) if n not in blockers]
    down_first = [(blk, None, 0)] + others
    down_last = others + [(blk, None, 0)]

    if not maxfield.workplan_is_valid(down_first + linkplan, depends, blockers=blockers):
        fails.append('taking the blocker down first was rejected')
    if not maxfield.workplan_is_valid(down_last + linkplan, depends, blockers=blockers):
        fails.append('taking the blocker down last, but still before any link, was rejected')

    # One link moved in front of the blocker is the whole failure mode
    early = others + linkplan[:1] + [(blk, None, 0)] + linkplan[1:]
    if not maxfield.workplan_is_valid(early, depends, blockers=None):
        fails.append('the early-link plan is unplayable for some other reason, '
                     'so this checks nothing -- fix the fixture or the construction')
    elif maxfield.workplan_is_valid(early, depends, blockers=blockers):
        fails.append('a link thrown while the blocker was still standing was accepted')

    # ALL the blockers, not just some. waypoints.txt has one, and with one
    # "all down" and "any down" agree, so the rule has to be checked on a
    # plan with two. workplan_is_valid() wants nothing from the graph but a
    # depends dict, so the plan here is made up rather than solved -- which
    # also keeps the check honest if the fixture ever changes.
    two = {90, 91}
    both = [(90, None, 0), (91, None, 0), (0, None, 0), (1, None, 0), (1, 0, 0)]
    half = [(90, None, 0), (0, None, 0), (1, None, 0), (1, 0, 0), (91, None, 0)]
    neither = [(0, None, 0), (1, None, 0), (1, 0, 0), (90, None, 0), (91, None, 0)]
    if not maxfield.workplan_is_valid(both, {}, blockers=two):
        fails.append('a plan with both blockers down before linking was rejected')
    if not maxfield.workplan_is_valid(half, {}, blockers=None):
        fails.append('the half-down plan is unplayable on its own terms, so it proves nothing')
    if maxfield.workplan_is_valid(half, {}, blockers=two):
        fails.append('linking with one of two blockers still standing was accepted')
    if maxfield.workplan_is_valid(neither, {}, blockers=two):
        fails.append('linking with both blockers still standing was accepted')

    # And the guard has to hold when improve_workplan is the one reordering
    maxfield.active_graph = b
    moved, stats = maxfield.improve_workplan(list(down_last + linkplan))
    standing = set()
    for idx, (p, q, f) in enumerate(moved):
        standing.add(p)
        if q is not None and not blockers.issubset(standing):
            fails.append('improve_workplan put a link at step %d, ahead of the blocker' % idx)
            break
    if not blockers.issubset({p for p, q, f in moved}):
        fails.append('improve_workplan dropped the blocker visit entirely')
    return fails


def check_no_links_before_blockers():
    """
    Every blocker has to be down before the first link is thrown.

    The portal list says only "there is an enemy portal here" -- nothing
    records which links it stands in the way of -- so the only safe reading
    is that it may block any of them. make_workplan() builds the plan as
    captures-then-links, which satisfies that by construction, but
    improve_workplan() then pulls links earlier to save backtracking. Until
    workplan_is_valid() learned about blockers, 86% of plans threw at least
    one link, and up to eight, while the blocker was still standing.

    Checked on the full graph and in subset mode, since improve_workplan()
    runs in both, and with the visit itself asserted: a plan that never goes
    to the blocker at all would pass a naive "nothing before it" test.
    """
    fails = []
    plans = 0

    def walk(label, workplan, graph):
        blockers = maxfield.get_blockers(graph)
        if not blockers:
            return 'the %s fixture no longer has a blocker to check' % label
        down = set()
        for idx, (p, q, f) in enumerate(workplan):
            down.add(p)
            if q is not None and not blockers.issubset(down):
                return ('%s: link at step %d with %d blocker(s) still standing'
                        % (label, idx, len(blockers - down)))
        if not blockers.issubset(down):
            return '%s: the plan never visits blocker %s' % (label, sorted(blockers - down))
        return None

    # Full graph
    reset_state()
    portals, waypoints = text_interface.get_portals_from_file(
        os.path.join(HERE, 'fixtures', 'waypoints.txt'))
    maxfield.populate_graphs(portals, waypoints)
    maxfield.gen_distance_matrix(None)
    np.random.seed(SEED)
    for i in range(25):
        maxfield.active_graph = None
        b = maxfield.portal_graph.copy()
        workplan, stats = solve_iteration(b, False)
        if workplan is None:
            continue
        plans += 1
        bad = walk('full graph iteration %d' % i, workplan, b)
        if bad:
            fails.append(bad)
            break

    # Subsets, which build their own graphs and renumber the nodes
    reset_state()
    maxfield.populate_graphs(portals, waypoints)
    maxfield.gen_distance_matrix(None)
    np.random.seed(SEED)
    subset = maxfield.make_subset(4)
    for i in range(25):
        maxfield.active_graph = None
        b = maxfield.make_subset_graph(subset).copy()
        workplan, stats = solve_iteration(b, True)
        if workplan is not None:
            plans += 1
            bad = walk('subset iteration %d' % i, workplan, b)
            if bad:
                fails.append(bad)
                break
        maxfield.active_graph = None
        if i % SUBSET_GROW_CYCLE == SUBSET_GROW_CYCLE - 1:
            subset = maxfield.make_subset(4, random_start=True)
        else:
            maxfield.add_subset_portal(subset)

    if plans < 20:
        fails.append('only %d plans were produced, too few to trust this' % plans)
    return fails


def check_minap_rejection():
    """
    An AP floor must reject a plan, never hand back a half-built one.

    Under --minap make_workplan gives up before solving the capture route,
    which saves real time in --maxtime mode. What it used to give back was
    the bare link order, and that is not a plan: nothing is captured, no
    waypoint is visited, no blocker comes down, and the first action links
    out of a portal never taken. It also scored *better* than a real plan,
    because get_workplan_stats() credits capture AP from the graph whether
    or not the plan captures anything, so it kept all the AP and paid none
    of the travel. With no --maxtime to reject it, that non-plan won.
    """
    fails = []
    reset_state()
    portals, waypoints = text_interface.get_portals_from_file(
        os.path.join(HERE, 'fixtures', 'waypoints.txt'))
    maxfield.populate_graphs(portals, waypoints)
    maxfield.gen_distance_matrix(None)

    def solve(minap):
        maxfield.minap = minap
        maxfield.capture_cache = dict()
        maxfield.active_graph = None
        np.random.seed(SEED)
        return solve_iteration(maxfield.portal_graph.copy(), False)

    workplan, stats = solve(None)
    if workplan is None:
        fails.append('no workplan at all with no AP floor')
        return fails
    reachable = stats['ap']

    # A floor it cannot reach: rejected, and told apart from a solver failure
    workplan, stats = solve(reachable + 1000000)
    if workplan is not None:
        fails.append('an unreachable AP floor still returned a workplan of %d steps; '
                     'captures=%d, so it is a bare linkplan'
                     % (len(workplan), len([w for w in workplan if w[1] is None])))
    if stats is None:
        fails.append('the AP floor rejection is indistinguishable from a solver failure')
    elif stats['ap'] >= reachable + 1000000:
        fails.append('rejected a plan that met the floor')

    # A floor it clears: a real plan, capture route and all
    workplan, stats = solve(1)
    if workplan is None:
        fails.append('a floor of 1 AP rejected everything')
    elif not [w for w in workplan if w[1] is None]:
        fails.append('a cleared AP floor returned a linkplan with no capture steps')

    # And the CLI must not let the two be used apart, since only --maxtime
    # ever consults the floor
    proc = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(HERE), 'fieldplan.py'),
         '--textfile', os.path.join(HERE, 'fixtures', 'waypoints.txt'), '--minap', '1000'],
        capture_output=True, text=True)
    if proc.returncode == 0:
        fails.append('--minap without --maxtime was accepted')
    elif 'minap' not in (proc.stderr + proc.stdout):
        fails.append('--minap without --maxtime failed without saying why: %s'
                     % (proc.stderr.strip().splitlines() or ['(silence)'])[-1])
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
                             ('default iterations', check_default_iterations),
                             ('minap rejection', check_minap_rejection),
                             ('blocker guard', check_blocker_guard),
                             ('blockers before links', check_no_links_before_blockers),
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
