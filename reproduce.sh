#!/bin/sh
# Reproduce every experiment in the report. Needs ADAPTION_API_KEY in .env.
# All steps are resumable: rerun the script to continue after an interruption.
set -e
cd "$(dirname "$0")"
J="--judges adaption/gemma-4-31b-it adaption/glm-5.2 --authority 1 --doses 0 0.75 --workers 10"
D="--chatter llm --chatter-model adaption/deepseek-v4-flash"
Q="--chatter llm --chatter-model adaption/qwen3.8-27b"
G="--summarizer adaption/gemma-4-26b-a4b-it"
# 1. Main grid (Figure 1A): single analyst + 6 pipeline conditions
python -m ci.run --name raw_ds --n-violating 10 --n-clean 10 --doses 0 0.75 --authority 1 \
  --analysts adaption/gemma-4-31b-it adaption/glm-5.2 --modes naive --seeds 3 $D
python -m ci.pipeline --name main --n-violating 10 --n-clean 10 --conds naive provenance actions --sum-policy 0 1 --seeds 3 $G $J $D
# 2. Chunk count
for k in 2 8; do
  python -m ci.pipeline --name chunks --chunks $k --conds naive --sum-policy 0 --seeds 2 $G $J $D
  python -m ci.pipeline --name chunks --chunks $k --conds provenance --sum-policy 1 --seeds 2 $G $J $D
done
# 3. Replication on fresh episodes, different log writer
python -m ci.pipeline --name ext_seed1 --scenario-seed 1 --n-violating 10 --n-clean 4 --conds naive provenance --sum-policy 0 1 --seeds 3 $G $J $Q
python -m ci.run --name raw_seed1 --scenario-seed 1 --n-violating 10 --n-clean 4 --doses 0 0.75 --authority 1 \
  --analysts adaption/gemma-4-31b-it adaption/glm-5.2 --modes naive --seeds 3 $Q
# 4. Different summariser
python -m ci.pipeline --name summ --summarizer adaption/qwen3.8-27b --conds naive --sum-policy 0 --seeds 2 $J $D
python -m ci.pipeline --name summ --summarizer adaption/qwen3.8-27b --conds provenance --sum-policy 1 --seeds 2 $J $D
# 5. Scale (Figure 1B)
B="--scenario-seed 2 --size 150 200 --n-violating 10 --n-clean 4 --seeds 2"
python -m ci.pipeline --name big $B $G --conds naive actions --sum-policy 0 $J $Q
python -m ci.pipeline --name big $B $G --conds provenance --sum-policy 1 $J $Q
python -m ci.pipeline --name big $B $G --chunks 8 --conds naive --sum-policy 0 $J $Q
python -m ci.run --name raw_big --scenario-seed 2 --size 150 200 --n-violating 10 --n-clean 4 --doses 0 0.75 --authority 1 \
  --analysts adaption/gemma-4-31b-it adaption/glm-5.2 --modes naive --seeds 2 $Q
# 6. Tables and figures
python -m ci.analyze && python -m ci.analyze_all
