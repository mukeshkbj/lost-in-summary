#!/bin/sh
# Extra robustness runs (Sat 7 Nov). Resumable: rerun this script to continue.
cd "$(dirname "$0")"
. .venv/bin/activate
J="--judges adaption/gemma-4-31b-it adaption/glm-5.2 --authority 1 --doses 0 0.75 --workers 10"
Q="--chatter llm --chatter-model adaption/qwen3.8-27b"
D="--chatter llm --chatter-model adaption/deepseek-v4-flash"
G="--summarizer adaption/gemma-4-26b-a4b-it"
for i in 1 2 3; do
  python -m ci.pipeline --name ext_seed1 --scenario-seed 1 --n-violating 10 --n-clean 4 --conds naive provenance --sum-policy 0 1 --seeds 3 $G $J $Q
done
python -m ci.run --name raw_seed1 --scenario-seed 1 --n-violating 10 --n-clean 4 --doses 0 0.75 --authority 1 \
  --analysts adaption/gemma-4-31b-it adaption/glm-5.2 --modes naive --seeds 3 --workers 8 $Q
for i in 1 2; do
  python -m ci.pipeline --name summ --summarizer adaption/qwen3.8-27b --conds naive --sum-policy 0 --seeds 2 $J $D
  python -m ci.pipeline --name summ --summarizer adaption/qwen3.8-27b --conds provenance --sum-policy 1 --seeds 2 $J $D
done
B="--scenario-seed 2 --size 150 200 --n-violating 10 --n-clean 4 --seeds 2"
for i in 1 2; do
  python -m ci.pipeline --name big $B $G --conds naive actions --sum-policy 0 $J $Q
  python -m ci.pipeline --name big $B $G --conds provenance --sum-policy 1 $J $Q
  python -m ci.pipeline --name big $B $G --chunks 8 --conds naive --sum-policy 0 $J $Q
done
python -m ci.run --name raw_big --scenario-seed 2 --size 150 200 --n-violating 10 --n-clean 4 --doses 0 0.75 --authority 1 \
  --analysts adaption/gemma-4-31b-it adaption/glm-5.2 --modes naive --seeds 2 --workers 8 $Q
echo ALL_EXTRA_FINISHED
