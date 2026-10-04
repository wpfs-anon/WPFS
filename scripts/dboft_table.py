import argparse, glob, json, os, random

HOME = os.environ.get("CORRECTOR_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS = ["StackCube", "Carrot", "Spoon", "Eggplant"]


def load_run(root, prefix, seeds, n_ep):
    out, ms, steps = {}, 0.0, 0
    for t in TASKS:
        for s in seeds:
            fs = sorted(glob.glob(f"{root}/{prefix}_r{s}/env_{t}/*/results.json"))
            fs = [f for f in fs if json.load(open(f)).get("total_episodes") == n_ep]
            if not fs:
                return None, None
            out.setdefault(t, {})[str(s)] = [bool(v) for v in json.load(open(fs[-1]))["success_array"]]
    for s in seeds:
        logs = glob.glob(f"{root}/calls_{prefix}_r{s}.jsonl") or glob.glob(f"{root}/goi_{prefix}_r{s}_*.jsonl")
        for f in logs:
            for line in open(f):
                r = json.loads(line)
                if r.get("kind") in (None, "cal") or "seg" not in r:
                    continue
                ms += r["ms"]; steps += r["seg"]
    return out, (ms / steps if steps else None)


def sign_flip(diff, n=20000, seed=0):
    obs = abs(sum(diff))
    rng = random.Random(seed)
    hit = sum(1 for _ in range(n) if abs(sum(d if rng.random() < 0.5 else -d for d in diff)) >= obs)
    return (hit + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="*", help="name=prefix, the first is the reference")
    ap.add_argument("--root", default=f"{HOME}/dboft/results/dboft")
    ap.add_argument("--seeds", default="0,2,6,8,12")
    ap.add_argument("--episodes-per-task", type=int, default=24)
    ap.add_argument("--episodes", default="", help="a per-episode JSON instead of run directories")
    a = ap.parse_args()
    rows = {}
    if a.episodes:
        d = json.load(open(a.episodes))
        for name, per in d["per_episode"].items():
            rows[name] = ({t: {s: [e["ok"] for e in v] for s, v in per[t].items()} for t in TASKS},
                          d.get("ms_per_step", {}).get(name))
    else:
        seeds = [int(s) for s in a.seeds.split(",")]
        for spec in a.rows:
            name, prefix = spec.split("=", 1)
            res, ms = load_run(a.root, prefix, seeds, a.episodes_per_task)
            if res is None:
                print(f"  {name}: incomplete, skipped"); continue
            rows[name] = (res, ms)
    if not rows:
        return
    ref_name = next(iter(rows))
    ref, ref_ms = rows[ref_name]
    print("  %-16s | %9s %7s %6s %9s | %7s | %8s | %7s | paired vs %s" % ("", *TASKS, "average", "ms/step", "speedup", ref_name))
    for name, (res, ms) in rows.items():
        pct = [100 * sum(sum(v) for v in res[t].values()) / sum(len(v) for v in res[t].values()) for t in TASKS]
        sp = f"{ref_ms / ms:6.2f}x" if ms and ref_ms else "      -"
        paired = ""
        if name != ref_name:
            diff = [sum(res[t][s][i] for s in res[t]) - sum(ref[t][s][i] for s in ref[t])
                    for t in TASKS for i in range(len(next(iter(res[t].values()))))]
            paired = "%+d episodes, p = %.3f" % (sum(diff), sign_flip(diff))
        print("  %-16s | %9.1f %7.1f %6.1f %9.1f | %7.1f | %8s | %7s | %s"
              % (name, *pct, sum(pct) / 4, f"{ms:.1f}" if ms else "-", sp, paired))


if __name__ == "__main__":
    main()
