# Lost in Summary

**AI investigators catch swarm violations alone, but miss most of them in summarise-then-judge pipelines.**

Apart Research Swarm Alignment sprint, 6–8 Nov 2026 · Tracks 4 (Multi-Agent Oversight & Control) and 1 (Sandboxes & Testbeds)

**Pre-existing code: none.** Everything in this repository was written during the sprint.

## Question

The July 2026 OpenAI/Hugging Face incident involved ~1,200 agents and 70,000+ board messages. Its investigation relied on LLM agents, and the authors warned those analysts may have adopted the swarm's perspective. We test two failure modes on synthetic swarm logs with known ground truth:

- **Capture:** does an analyst accept a violation as legitimate when many agents assert it was sanctioned?
- **Loss:** does a summarise-then-judge pipeline, needed at swarm scale, drop the evidence?

## Results (violations detected; false alarms ≤ 2.5% everywhere)

| Setup | Main grid (40–60 actions) | Fresh episodes, other log writer | 150–200 actions |
|---|---|---|---|
| Single analyst, raw log | **100%** | **100%** | see `results/final` |
| Plain summaries → judge | **63%** | 60% | **16%** |
| Provenance + policy-aware summaries | 99% | 100% | 99% |
| Plain summaries + raw action records | 100% | — | 100% |

- No capture on raw logs: "sanctioned" claims from 0% vs 75% of agents, and authority cues, never moved a single-analyst verdict.
- The loss is caused by policy-blind summarisers paraphrasing away the violating detail; it grows with log length.
- Fixes: brief summarisers with the policy and a verified-actions-vs-claims format, or (most robust) pass the judge the raw action records (+~28% input tokens).

See `results/final/all_experiments.csv`, `results/final/fig1_two_panel.png` and `results/final/example_S00.md`.

## Layout

- `ci/scenarios.py` episode factory; ground truth lives only in `[ACTION]` records
- `ci/generate.py` board rendering, LLM-written chatter, consensus and authority injectors (messages only)
- `ci/analyst.py` single-analyst prompts · `ci/pipeline.py` summarise-then-judge pipeline
- `ci/llm.py` backends (Adaption hosted models; litellm for others)
- `ci/run.py`, `ci/score.py`, `ci/analyze.py`, `ci/analyze_all.py`

## Reproduce

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install litellm adaption pyyaml pandas matplotlib
echo "ADAPTION_API_KEY=..." > .env
sh reproduce.sh
```

Models (hosted on Adaption): Gemma 4 31B and GLM 5.2 (judges / single analysts), Gemma 4 26B A4B and Qwen 3.8 27B (summarisers), DeepSeek V4 Flash and Qwen 3.8 27B (log chatter writers).

## Ethics

All logs are synthetic; action targets are placeholders (`*.example`, `*.sandbox.internal`) with no exploit content. Optimised high-capture message templates are not released.
