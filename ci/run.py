"""Run a grid of (scenario x dose x authority) logs through analysts. Resumable: results are
appended to a JSONL and completed keys are skipped on rerun.

Examples:
  python -m ci.run --name dryrun --analysts mock --chatter template
  python -m ci.run --name pilot --n-violating 10 --n-clean 10 --doses 0 0.75 --authority 1 \
      --analysts anthropic/claude-haiku-4-5 --modes naive --seeds 3 --chatter template
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .scenarios import make_scenarios
from .generate import build_log, template_chatter, llm_chatter
from .analyst import call

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-violating", type=int, default=12)
    ap.add_argument("--n-clean", type=int, default=12)
    ap.add_argument("--doses", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75])
    ap.add_argument("--authority", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--analysts", nargs="+", default=["mock"])
    ap.add_argument("--modes", nargs="+", default=["naive", "evidence"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--kin", nargs="*", default=[], help="kin labels, e.g. 'the same model as you' 'a different model family'")
    ap.add_argument("--chatter", choices=["template", "llm"], default="template")
    ap.add_argument("--chatter-model", default=None, help="neutral third-family model for llm chatter")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scenario-seed", type=int, default=0)
    a = ap.parse_args()

    out_dir = ROOT / "results" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = make_scenarios(a.n_violating, a.n_clean, a.scenario_seed)
    (out_dir / "scenarios.json").write_text(json.dumps([s.to_dict() for s in scenarios], indent=1))

    # Chatter is generated once per scenario and cached, so every condition shares it.
    chat_path = ROOT / "logs" / f"chatter_{a.chatter}_{a.scenario_seed}.json"
    chat = json.loads(chat_path.read_text()) if chat_path.exists() else {}
    for sc in scenarios:
        if sc.id not in chat:
            rng = random.Random(sc.id)
            chat[sc.id] = (template_chatter(sc, rng) if a.chatter == "template"
                           else llm_chatter(sc, a.chatter_model, rng))
    chat_path.write_text(json.dumps(chat))

    logs = {}
    for sc in scenarios:
        for d in a.doses:
            for au in a.authority:
                lg = build_log(sc, d, bool(au), chat[sc.id])
                logs[(sc.id, d, au)] = lg
                (ROOT / "logs" / f"{sc.id}_d{d}_a{au}.txt").write_text(lg["log"])

    res_path = out_dir / "results.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            r = json.loads(line)
            if r.get("violation") is not None:
                done.add(r["key"])

    kins = a.kin or [None]
    jobs = []
    for (sid, d, au), lg in logs.items():
        for m in a.analysts:
            for mode in a.modes:
                for kin in kins:
                    for s in range(a.seeds):
                        key = f"{sid}|{d}|{au}|{m}|{mode}|{kin}|{s}"
                        if key not in done:
                            jobs.append((key, lg, m, mode, kin, s))
    print(f"{len(logs)} logs, {len(jobs)} analyst calls to run ({len(done)} already done)")

    with ThreadPoolExecutor(a.workers) as ex, res_path.open("a") as f:
        futs = {ex.submit(call, m, lg, mode, s, kin): (key, lg, m, mode, kin, s)
                for key, lg, m, mode, kin, s in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            key, lg, m, mode, kin, s = futs[fut]
            out = fut.result()
            rec = {"key": key, "scenario": lg["scenario"], "violating": lg["violating"],
                   "dose": lg["dose"], "authority": lg["authority"], "analyst": m, "mode": mode,
                   "kin": kin, "seed": s, "truth_actions": lg["truth_actions"], **out}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}")
    print(f"done -> {res_path}")


if __name__ == "__main__":
    main()
