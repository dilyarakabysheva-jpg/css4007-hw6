"""HW6 Part C: value iteration for the Grid World maps in data/maps.json.

Floor rules (rewards, slipping, terminal tiles) are read from data/maps.json,
so adding a new floor type only needs a new entry in "tiles" -- the solver
itself does not change.

Run:  python part_c_solver.py
Writes everything into outputs/.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent
CONFIG = json.loads((ROOT / "data" / "maps.json").read_text())
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

ACTIONS = CONFIG["actions"]                      # ["up", "right", "down", "left"]
MOVES = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
ARROWS = {"up": "^", "right": ">", "down": "v", "left": "<"}
PERPENDICULAR = {"up": ("left", "right"), "down": ("left", "right"),
                 "left": ("up", "down"), "right": ("up", "down")}
GAMMA = CONFIG["discount"]
TILES = CONFIG["tiles"]
TIE_EPS = 1e-9   # values closer than this count as a tie


# ---------------------------------------------------------------- model
class GridWorld:
    def __init__(self, rows):
        self.rows = rows
        self.n_rows, self.n_cols = len(rows), len(rows[0])
        self.states = [(r, c) for r in range(self.n_rows) for c in range(self.n_cols)
                       if TILES[rows[r][c]].get("traversable", True)]
        self.start = next(s for s in self.states if self.tile(s) == "S")

    def tile(self, s):
        return self.rows[s[0]][s[1]]

    def is_terminal(self, s):
        return TILES[self.tile(s)].get("terminal", False)

    def reward(self, s):
        return TILES[self.tile(s)]["reward"]

    def move(self, s, direction):
        """Deterministic result of moving one step; walls/boundary -> stay."""
        dr, dc = MOVES[direction]
        r, c = s[0] + dr, s[1] + dc
        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols):
            return s
        if not TILES[self.rows[r][c]].get("traversable", True):
            return s
        return (r, c)

    def direction_probs(self, s, action):
        """Which direction actually happens; depends on the tile we LEAVE."""
        rule = TILES[self.tile(s)]
        if rule.get("movement") == "slippery":
            a, b = PERPENDICULAR[action]
            return [(action, rule["intended"]), (a, rule["perpendicular_each"]),
                    (b, rule["perpendicular_each"])]
        return [(action, 1.0)]

    def transitions(self, s, action):
        """List of (probability, next_state, reward). Empty for terminal s."""
        if self.is_terminal(s):
            return []
        out = {}
        for direction, p in self.direction_probs(s, action):
            s2 = self.move(s, direction)
            out[s2] = out.get(s2, 0.0) + p
        return [(p, s2, self.reward(s2)) for s2, p in out.items()]


# ---------------------------------------------------------------- solver
def q_value(world, V, s, a):
    return sum(p * (r + GAMMA * V[s2]) for p, s2, r in world.transitions(s, a))


def bellman_update(world, V):
    """One full synchronous update: uses only the previous round's V."""
    new_V = {}
    for s in world.states:
        if world.is_terminal(s):
            new_V[s] = CONFIG["terminal_value"]
        else:
            new_V[s] = max(q_value(world, V, s, a) for a in ACTIONS)
    return new_V


def value_iteration(world):
    tol = CONFIG["solver"]["tolerance"]
    max_it = CONFIG["solver"]["max_iterations"]
    V = {s: float(CONFIG["initial_value"]) for s in world.states}
    history = [V]
    for k in range(1, max_it + 1):
        new_V = bellman_update(world, V)
        delta = max(abs(new_V[s] - V[s]) for s in world.states)
        V = new_V
        history.append(V)
        if delta < tol:
            return V, k, delta, True, history
    return V, max_it, delta, False, history


def best_actions(world, V, s):
    qs = {a: q_value(world, V, s, a) for a in ACTIONS}
    best = max(qs.values())
    return [a for a in ACTIONS if qs[a] >= best - TIE_EPS], qs


def extract_policy(world, V):
    """Tie-break: first action in the order up, right, down, left."""
    policy, ties = {}, {}
    for s in world.states:
        if world.is_terminal(s):
            continue
        tied, _ = best_actions(world, V, s)
        policy[s] = tied[0]
        ties[s] = tied
    return policy, ties


# ---------------------------------------------------------------- printing
def value_map(world, V):
    lines = []
    for r in range(world.n_rows):
        cells = []
        for c in range(world.n_cols):
            t = world.rows[r][c]
            cells.append("   ####  " if t == "#" else f"{V[(r, c)]:8.3f}{t}")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def policy_map(world, policy):
    lines = []
    for r in range(world.n_rows):
        row = ""
        for c in range(world.n_cols):
            s, t = (r, c), world.rows[r][c]
            row += t if (t == "#" or world.is_terminal(s)) else ARROWS[policy[s]]
            row += " "
        lines.append(row)
    return "\n".join(lines)


def ones(s):
    return f"({s[0] + 1},{s[1] + 1})"


# ---------------------------------------------------------------- simulation
def run_episode(world, policy, rng, max_steps):
    s, steps, total, disc = world.start, 0, 0.0, 0.0
    trace = [ones(s)]
    while steps < max_steps and not world.is_terminal(s):
        outcomes = world.transitions(s, policy[s])
        x, acc = rng.random(), 0.0
        for p, s2, r in outcomes:
            acc += p
            if x < acc:
                break
        disc += (GAMMA ** steps) * r
        total += r
        steps += 1
        s = s2
        trace.append(ones(s))
    if world.is_terminal(s):
        outcome = "goal" if world.tile(s) == "G" else "hole"
    else:
        outcome = "time_limit"
    return steps, total, disc, outcome, trace


def simulate(name, world, policy):
    sim = CONFIG["simulation"]
    rows, summary = [], []
    for b in sim["batch_seeds"]:
        res = []
        for i in range(sim["episodes_per_batch"]):
            seed = 1000 * b + i
            steps, total, disc, outcome, trace = run_episode(
                world, policy, random.Random(seed), sim["max_steps"])
            res.append((steps, disc, outcome))
            rows.append(f"{name},{b},{seed},{steps},{total},{disc:.6f},{outcome},{' '.join(trace)}")
        n = len(res)
        summary.append((name, b,
                        sum(o == "goal" for _, _, o in res),
                        sum(o == "hole" for _, _, o in res),
                        sum(o == "time_limit" for _, _, o in res),
                        sum(st for st, _, _ in res) / n,
                        sum(d for _, d, _ in res) / n))
    return rows, summary


# ---------------------------------------------------------------- rule checks
def rule_checks():
    lines = []
    w = GridWorld(CONFIG["extended_map"])
    ok = True

    def check(cond, text):
        nonlocal ok
        ok &= cond
        lines.append(("PASS " if cond else "FAIL ") + text)

    # wall / boundary
    check(w.move((0, 0), "up") == (0, 0), "(1,1) up: outside map -> stays at (1,1)")
    check(w.move((0, 0), "left") == (0, 0), "(1,1) left: outside map -> stays at (1,1)")
    check(w.move((0, 1), "down") == (0, 1), "(1,2) down: wall (2,2) -> stays at (1,2)")
    check(w.transitions((0, 1), "down") == [(1.0, (0, 1), -3)],
          "(1,2) down into wall: reward is mud -3 (tile where move ends)")
    check((1, 1) not in w.states, "wall (2,2) is not a state")
    # ice probabilities
    for a in ACTIONS:
        tr = w.transitions((1, 2), a)
        check(abs(sum(p for p, _, _ in tr) - 1) < 1e-12,
              f"ice (2,3) action {a}: probabilities sum to 1 -> {tr}")
    for s in w.states:
        for a in ACTIONS:
            tr = w.transitions(s, a)
            if tr:
                assert abs(sum(p for p, _, _ in tr) - 1) < 1e-12
    check(True, "all non-terminal states/actions: probabilities sum to 1")
    # floor rewards
    check(w.transitions((0, 0), "right") == [(1.0, (0, 1), -3)], "enter mud M (1,2): reward -3")
    check(w.transitions((2, 0), "right") == [(1.0, (2, 1), -2)], "enter sand D (3,2): reward -2")
    check(w.transitions((2, 2), "right") == [(1.0, (2, 3), -5)], "enter water W (3,4): reward -5")
    check(w.transitions((0, 2), "right") == [(1.0, (0, 3), 10)], "enter goal G (1,4): reward +10")
    check(w.transitions((2, 3), "up") == [(1.0, (1, 3), -10)], "enter hole H (2,4): reward -10")
    check(w.direction_probs((2, 1), "up") == [("up", 1.0)], "sand is deterministic")
    check(w.direction_probs((2, 3), "left") == [("left", 1.0)], "water is deterministic")
    # terminal
    for s in [(0, 3), (1, 3)]:
        check(all(w.transitions(s, a) == [] for a in ACTIONS),
              f"terminal {ones(s)}: no transitions, no further reward")
    V, *_ = value_iteration(w)
    check(V[(0, 3)] == 0 and V[(1, 3)] == 0, "terminal values stay 0")
    lines.append("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def main():
    report = []

    def say(text=""):
        print(text)
        report.append(text)

    say("=== RULE CHECKS ===")
    say(rule_checks())

    # --- compare with paper (Part A map)
    paper = GridWorld(CONFIG["paper_map"])
    V1 = bellman_update(paper, {s: 0.0 for s in paper.states})
    V2 = bellman_update(paper, V1)
    say("\n=== PAPER MAP: V1 ===")
    say(value_map(paper, V1))
    say("\n=== PAPER MAP: V2 at requested positions (all four actions) ===")
    for r1, c1 in CONFIG["paper_v2_positions"]:
        s = (r1 - 1, c1 - 1)
        tied, qs = best_actions(paper, V1, s)
        q_text = ", ".join(f"{a}={qs[a]:.3f}" for a in ACTIONS)
        say(f"({r1},{c1}) {paper.tile(s)}: {q_text} -> V2={V2[s]:.3f}, best={tied}")

    all_csv = ["map,batch_seed,episode_seed,steps,total_reward,discounted_return,outcome,trace"]
    all_summary = []
    for name, key in [("paper", "paper_map"), ("extended", "extended_map")]:
        w = GridWorld(CONFIG[key])
        V, iters, delta, converged, _ = value_iteration(w)
        policy, ties = extract_policy(w, V)
        say(f"\n=== {name.upper()} MAP: {CONFIG[key]} ===")
        say(f"converged={converged}, iterations={iters}, final max change={delta:.3e}")
        say("Values:")
        say(value_map(w, V))
        say("Policy (ties broken in order up, right, down, left):")
        say(policy_map(w, policy))
        tie_list = {ones(s): t for s, t in ties.items() if len(t) > 1}
        say(f"Tied states: {tie_list if tie_list else 'none'}")
        rows, summary = simulate(name, w, policy)
        all_csv += rows
        all_summary += summary

    say("\n=== SIMULATION (computed policy, 3 x 50 episodes) ===")
    say("| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |")
    say("| --- | --- | --- | --- | --- | --- | --- |")
    for m, b, g, h, t, st, d in all_summary:
        say(f"| {m} | {b} | {g} | {h} | {t} | {st:.2f} | {d:.3f} |")

    (OUT / "part_c_episodes.csv").write_text("\n".join(all_csv) + "\n")
    (OUT / "part_c_report.txt").write_text("\n".join(report) + "\n")
    print("\nSaved outputs/part_c_report.txt and outputs/part_c_episodes.csv")


if __name__ == "__main__":
    main()