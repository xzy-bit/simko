# Session Notes

## Collaboration

- Unless explicitly asked to submit, only provide commands. Do not `sbatch` on the user's behalf.

## Training

- Main training repo: `/home/users/astar/cfar/stuziyang/SimKO`
- Model outputs: `/scratch/prj0000000224/models`

## Recent 3B TS2 nosquare runs

- `mix_topk_coef=0.01`
  - model dir: `/scratch/prj0000000224/models/MATH-Llama-3.2-3B-SimKO-TS2-nosquare-mix001`
- `mix_topk_coef=0.05`
  - model dir: `/scratch/prj0000000224/models/MATH-Llama-3.2-3B-SimKO-TS2-nosquare-mix005`
- `mix_topk_coef=0.1`
  - model dir: `/scratch/prj0000000224/models/MATH-Llama-3.2-3B-SimKO-TS2-nosquare-mix01`

- Prefer evaluating `global_step_160`.

## Validation config

- Added `trainer.test_start_step`
- Intended behavior for formal training:
  - start validation after step 100
  - validate every 10 steps

