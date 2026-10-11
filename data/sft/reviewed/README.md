# Reviewed training examples

Put expert-written or expert-reviewed examples here as `*.jsonl`, one conversation per line,
in the same shape `tools/build_sft.py` writes (see docs/TRAINING_GUIDE.md, step 5).
`python -m tools.build_sft` merges every file in this folder into train/valid.

Only quote text from sources marked `allow_training: true` in tools/sources.py.
