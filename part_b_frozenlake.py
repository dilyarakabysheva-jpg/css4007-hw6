"""HW6 Part B: repeated FrozenLake-v1 (4x4) simulations with Gymnasium.

Run:  python part_b_frozenlake.py
Writes outputs/part_b_episodes.csv, outputs/part_b_summary.md,
outputs/part_b_traces.txt and outputs/part_b_example_episode.txt
"""
import csv
import random
from pathlib import Path

import gymnasium as gym

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

MAX_STEPS = 100
BATCH_SEEDS = [11, 22, 33]
EPISODES = 50
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3          # FrozenLake action numbers
NAMES = {LEFT: "Left", DOWN: "Down", RIGHT: "Right", UP: "Up"}

# Map (state = row*4 + col):
#   S F F F      0  1  2  3
#   F H F H      4  5  6  7
#   F F F H      8  9 10 11
#   H F F G     12 13 14 15
# Fixed policy written BEFORE running anything. Holes 5,7,11,12 and goal 15
# are terminal, so they need no action.
# Main path: 0 v 4 v 8 > 9 v 13 > 14 > 15. Other states lead back onto it.
FIXED_POLICY = {
    0: DOWN, 1: LEFT, 2: DOWN, 3: LEFT,
    4: DOWN, 6: DOWN,
    8: RIGHT, 9: DOWN, 10: DOWN,
    13: RIGHT, 14: RIGHT,
}


def run_episode(env, policy_name, seed, rng, keep_render=False):
    state, _ = env.reset(seed=seed)
    steps, total = 0, 0.0
    trace = [env.render()] if keep_render else []
    actions = []
    terminated = truncated = False
    while steps < MAX_STEPS and not (terminated or truncated):
        if policy_name == "random":
            action = rng.randrange(4)
        else:
            action = FIXED_POLICY[state]
        state, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        total += reward
        actions.append(NAMES[action])
        if keep_render:
            trace.append(env.render())
    tile = env.unwrapped.desc.flatten()[state].decode()
    if terminated and tile == "G":
        outcome = "goal"
    elif terminated and tile == "H":
        outcome = "hole"
    else:
        outcome = "time_limit"
    return steps, total, outcome, state, actions, trace


def main():
    rows, summary, traces = [], [], []
    for slippery in [False, True]:
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery,
                       render_mode="ansi", max_episode_steps=MAX_STEPS)
        for policy_name in ["random", "fixed"]:
            condition = f"slippery={slippery}, policy={policy_name}"
            saved = {"goal": False, "fail": False}
            for b in BATCH_SEEDS:
                rng = random.Random(b)     # seeds the random action generator
                res = []
                for i in range(EPISODES):
                    seed = 1000 * b + i
                    steps, total, outcome, final, actions, trace = run_episode(
                        env, policy_name, seed, rng, keep_render=True)
                    res.append((steps, outcome))
                    rows.append([condition, slippery, b, seed, policy_name,
                                 steps, total, outcome, final])
                    kind = "goal" if outcome == "goal" else "fail"
                    if not saved[kind]:
                        saved[kind] = True
                        traces.append(
                            f"##### {condition} | batch {b} | seed {seed} | "
                            f"outcome={outcome} | steps={steps}\n"
                            f"actions: {' '.join(actions)}\n" + "".join(trace))
                summary.append((slippery, policy_name, b,
                                sum(o == "goal" for _, o in res),
                                sum(o == "hole" for _, o in res),
                                sum(o == "time_limit" for _, o in res),
                                sum(s for s, _ in res) / EPISODES))
            for kind, found in saved.items():
                if not found:
                    traces.append(f"##### {condition}: no '{kind}' episode was observed\n")
        env.close()

    with open(OUT / "part_b_episodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "slippery", "batch_seed", "episode_seed", "policy",
                    "steps", "total_reward", "outcome", "final_state"])
        w.writerows(rows)

    table = ["| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | "
             "Time limits /50 | Success % | Mean steps |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for sl, p, b, g, h, t, ms in summary:
        table.append(f"| {sl} | {p} | {b} | {g} | {h} | {t} | {100 * g / EPISODES:.0f}% | {ms:.2f} |")
    (OUT / "part_b_summary.md").write_text("\n".join(table) + "\n")
    (OUT / "part_b_traces.txt").write_text("\n".join(traces))

    # one complete episode with text rendering (fixed policy, not slippery)
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False,
                   render_mode="ansi", max_episode_steps=MAX_STEPS)
    steps, total, outcome, _, actions, trace = run_episode(
        env, "fixed", 11000, random.Random(11), keep_render=True)
    (OUT / "part_b_example_episode.txt").write_text(
        f"fixed policy, is_slippery=False, seed 11000: outcome={outcome}, "
        f"steps={steps}, reward={total}\nactions: {' '.join(actions)}\n" + "".join(trace))
    env.close()

    print("\n".join(table))
    print("\nSaved files in outputs/")


if __name__ == "__main__":
    main()
    