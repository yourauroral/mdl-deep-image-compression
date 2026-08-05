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

# ImageNet64: full formal score on one H800. Sequential verification is optional
# because the 12,288-token no-KV-cache codec is extremely expensive.
python3 scripts/run_formal_evaluations.py imagenet64 \
    --nproc_per_node 1 --imagenet64_batch_size 2
```

These commands reuse the existing pretrained `best.pth` files; they do not
train or modify model weights. Run `--dry_run` on WSL, then execute the emitted
commands on AutoDL where the checkpoints, datasets, and CUDA runtime are
available. Keep verification and evaluation on the same source revision and
runtime so the codec identity can be matched.

Expected tracked files:

- `cifar10/teacher_forced.json`
- `cifar10/sequential_roundtrip.json` when calibration was run
- `cifar10/linear_probe_v3_no_coarse_ctx.{json,csv}`
- `cifar10/linear_probe_v3_with_ctx.{json,csv}`
- `imagenet64/teacher_forced.json`
- `imagenet64/sequential_roundtrip.json` only when actually run
- `imagenet64/linear_probe_transfer_v3.csv`
- `imagenet64/traditional_codecs_val_full.json` for the completed classical baseline run

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

The committed ImageNet64 teacher-forced manifest currently reports:

| Protocol | Samples | bpd | Per-image std | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| single `best.pth`, no TTA, teacher-forced | 49,999 | **3.4812** | 0.9040 | [3.4734, 3.4898] |

The result uses fp32 logits and fp64 softmax, with one model member and one
forward per image. Its `sequential_roundtrip.status` remains `not_run` because
the later verification was generated as a separate manifest. The independent
`imagenet64/sequential_roundtrip.json` verifies sample index 0 with pixel-exact
decode and reports payload `3.1597 bpd`, packed payload `3.1602 bpd`, and
complete-file `4.5000 bpd`. This is one-image evidence only; the 3.4812 score is
still an ideal-model teacher-forced NLL, not a full-set arithmetic/file rate.

The completed AutoDL CIFAR-10 run reports the following formal score:

| Protocol | Samples | bpd | Per-image std | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| single `best.pth`, no TTA, teacher-forced | 10,000 | **2.8328** | 0.6719 | [2.8201, 2.8456] |

This score uses the same fp32-logit/fp64-softmax teacher-forced protocol. The
tracked `cifar10/sequential_roundtrip.json` reports `verified_on_subset` for
sample indices 0 and 1: both images are pixel-exact, integrity-verified, and
RGB-checksum-verified under the same codec identity. On this two-image subset,
mean payload bpd is 2.7074, packed payload bpd is 2.7109, and complete-file bpd
is 8.0703. These payload/file rates are subset measurements, not the full-set
teacher-forced score.

The v3 linear-probe artifacts use a stratified validation split for layer
selection, five classifier seeds, and a final retrain on the full training set.
The CIFAR-10 native probe reaches 78.558% with context (layer 18; 95% bootstrap
CI [77.7278%, 79.3700%]) and 70.010% without context (layer 15; CI
[69.1315%, 70.8866%]). The ImageNet64-to-CIFAR-10 transfer probe reaches
72.656% (layer 18; CI [71.8059%, 73.4800%]).

The probe and ImageNet64 roundtrip manifests record execution commit `74f945`
with a dirty worktree and a source fingerprint from before the later Triton
small-head fallback commit. The evaluated production configurations use
`d_k=64`, so that fallback is not selected; nevertheless, rerun on a clean
current revision before treating these artifacts as clean-commit release
evidence.

Traditional PNG/WebP baselines are separate from the model manifests. The
completed ImageNet64 validation-set run (49,999 images, AutoDL, 2026-08-06)
reports:

| Codec | Samples | Complete-file bpd | Per-image std |
|---|---:|---:|---:|
| PNG (`optimize=True`) | 49,999 | **5.7063** | 0.9151 |
| WebP (`lossless=True`) | 49,999 | **4.6365** | 0.9448 |

The manifest is
[`imagenet64/traditional_codecs_val_full.json`](imagenet64/traditional_codecs_val_full.json).
The denominator is `H*W*C` and encoded file headers are included. Runtime
versions are Pillow 10.3.0, zlib 1.2.13, and libwebp 1.3.2. The earlier
2,000-image prefix diagnostic (PNG 5.718 bpd / WebP 4.640 bpd) remains a
separate exploratory record. These complete-file rates must not be conflated
with the CC-iGPT teacher-forced ideal-model NLL, which is not a measured
arithmetic payload or complete-file rate.
