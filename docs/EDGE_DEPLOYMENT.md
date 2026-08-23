# Edge deployment and benchmarking

DrishtiPath exports Ultralytics models to ONNX or NCNN for edge targets such as Raspberry Pi.
Benchmark reports always apply only to the **current machine** (`hardware_scope:
current_machine_only`). Raspberry Pi detection is automatic and conservative.

## Supported precision

| Format | FP32 | FP16 | INT8 |
|---|---|---|---|
| ONNX | yes | yes | yes (requires `--data`) |
| NCNN | yes | yes | **no** |

Ultralytics `quantize` mapping:

- `fp32` → no quantize argument
- `fp16` → `quantize=16`
- `int8` → `quantize=8` plus representative calibration YAML via `--data`

Deprecated `--int8` remains as an alias for `--precision int8`.

## Example commands

ONNX FP32 with uniform video sampling:

```bash
python optimize_model.py \
  --model models/road_hazards.pt \
  --format onnx \
  --precision fp32 \
  --source clips/1.mp4 \
  --device cpu \
  --imgsz 640 \
  --sampling uniform \
  --warmup-runs 5 \
  --benchmark-frames 50 \
  --confidence 0.25 \
  --iou 0.70 \
  --report artifacts/edge_bench/road_hazards_onnx_fp32.json
```

NCNN FP16:

```bash
python optimize_model.py \
  --model models/road_hazards.pt \
  --format ncnn \
  --precision fp16 \
  --source clips/1.mp4 \
  --device cpu \
  --imgsz 640 \
  --sampling uniform \
  --warmup-runs 5 \
  --benchmark-frames 50 \
  --report artifacts/edge_bench/road_hazards_ncnn_fp16.json
```

Optional validation (distinct from INT8 calibration):

```bash
python optimize_model.py \
  --model models/road_hazards.pt \
  --format onnx \
  --precision fp32 \
  --source clips/1.mp4 \
  --validation-data datasets/road_hazards.yaml \
  --report artifacts/edge_bench/road_hazards_onnx_fp32_validated.json
```

## Benchmark methodology

1. Validate source media, model path, numeric settings and calibration/validation YAML early.
2. For videos, sample unique frames across the whole clip (`--sampling uniform` by default).
3. Warm up source and exported backends before recording timings.
4. Record separately:
   - backend inference median/P95 (`result.speed`)
   - wall mean / median / P95 and total wall time
   - `measured_end_to_end_fps = sample_count / (total_wall_time_ms / 1000)`
   - optional `fps_from_median_wall_ms` (theoretical median-derived throughput only)
5. Never label `1000 / inference_ms` or `1000 / median_wall_ms` as measured end-to-end FPS.
6. Pass the requested `--device` into both inference and `model.export()`.

## Prediction parity

The report includes `prediction_parity`: same-class greedy IoU matching on unique sampled
frames only. Aggregate mean IoU and confidence deltas are **detection-weighted**, not
frame-weighted.

This is **not** validation accuracy. Empty-vs-empty frames are counted separately. A single
matched detection is weak evidence and insufficient to claim model quality.

## Artifact integrity

Reports include measured bytes, MiB, SHA-256 hashes and directory manifests. Size change is
computed from actual artifacts. Positive change is labelled size reduction; negative change is
labelled size increase.

## Raspberry Pi detection

On Linux, `/proc/device-tree/model` is read when available. `raspberry_pi_benchmarked` is true
only when that text clearly contains `Raspberry Pi`. On macOS, Windows, missing files or
permission errors it remains false. There is no CLI flag to override this claim.

`hardware_scope` remains `current_machine_only` even on a Pi, because results apply only to that
specific machine.
