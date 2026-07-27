# Formal evaluation artifacts

This directory is reserved for generated, reviewable result manifests. Do not
hand-edit result JSON or copy historical log values into it.

Run the evaluation from the committed source revision on AutoDL:

```bash
# Inspect commands without running a model.
python3 scripts/run_formal_evaluations.py --dry_run

# CIFAR-10: real MDLC v2 calibration first, then the full teacher-forced test set.
python3 scripts/run_formal_evaluations.py cifar10 \
    --verify_cifar_images 2

# ImageNet64: full formal score on four GPUs. Sequential verification is optional
# because the 12,288-token no-KV-cache codec is extremely expensive.
python3 scripts/run_formal_evaluations.py imagenet64 \
    --nproc_per_node 4
```

These commands reuse the existing pretrained `best.pth` files; they do not
train or modify model weights. Run `--dry_run` on WSL, then execute the emitted
commands on AutoDL where the checkpoints, datasets, and CUDA runtime are
available. Keep verification and evaluation on the same source revision and
runtime so the codec identity can be matched.

Expected tracked files:

- `cifar10/teacher_forced.json`
- `cifar10/sequential_roundtrip.json` when calibration was run
- `imagenet64/teacher_forced.json`
- `imagenet64/sequential_roundtrip.json` only when actually run

The `teacher_forced.json` manifest reports model NLL and never claims that its
full test-set score came from arithmetic coding. An attached roundtrip manifest
only proves sequential MDLC v2 decoding for the explicitly listed subset and
must match the evaluation's dataset fingerprint. It is valid to omit sequential
verification when its no-KV-cache cost is impractical; the manifest will retain
an explicit `not_run` state. Existing evidence is never auto-attached merely
because a file is present in the output directory; reuse requires the explicit
`--cifar_verification_json` or `--imagenet64_verification_json` option.
Generated `.bin` payloads are intentionally ignored because the JSON records
their checksums and rate accounting.
