#!/usr/bin/env bash
# Usage: bash scripts/train_on_policy_script/run_kaggle_grpo.sh smoke|experience|assignment [CLI overrides]
set -euo pipefail
cd "$(dirname "$0")/../.."
profile="${1:-smoke}"
if [ "$#" -gt 0 ]; then shift; fi
case "$profile" in
  smoke)
    profile_args=(--train-steps 2 --prompts-per-step 1 --group-size 2
      --gradient-accumulation-steps 2 --max-prompt-tokens 256 --max-new-tokens 64
      --eval-every 1 --eval-size 2 --save-every 2)
    ;;
  experience)
    profile_args=(--train-steps 50 --prompts-per-step 1 --group-size 8
      --gradient-accumulation-steps 8 --max-prompt-tokens 256 --max-new-tokens 512
      --eval-every 10 --eval-size 32 --save-every 10)
    ;;
  assignment)
    # Same response batch as PDF; microbatch=1 to accommodate a T4.
    profile_args=(--train-steps 200 --train-size 6400 --prompts-per-step 32 --group-size 8
      --gradient-accumulation-steps 256 --max-prompt-tokens 256 --max-new-tokens 512
      --eval-every 10 --eval-size 1024 --save-every 25)
    ;;
  *) echo "Unknown profile: $profile (use smoke, experience, assignment)" >&2; exit 2 ;;
esac
output_dir="${GRPO_OUTPUT_DIR:-/kaggle/working/grpo-${profile}}"
mkdir -p "$output_dir"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
if [ -d /kaggle/temp ]; then
  export HF_HOME="${HF_HOME:-/kaggle/temp/huggingface}"
fi
python -u scripts/train_on_policy_script/train_on_policy_script.py \
  --train-gpu 0 --rollout-gpu 1 --rollout-backend transformers \
  --optimizer adamw_cpu --learning-rate 1e-5 --max-grad-norm 1.0 \
  --rollout-batch-size 4 --keep-checkpoints 1 \
  --output-dir "$output_dir" "${profile_args[@]}" "$@" \
  2>&1 | tee -a "$output_dir/console.log"
