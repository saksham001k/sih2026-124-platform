# Edge deployment and benchmarking

DrishtiPath exports Ultralytics models to ONNX or NCNN for edge targets such as Raspberry Pi.
Benchmark reports are generated on the **current machine only** unless a separate Pi report is
produced on the device itself.

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

ONNX FP32:

```bash
python optimize_model.py \
  --model models/road_hazards.pt \
  --format onnx \
  --precision fp32 \
  --source clips/1.mp4 \
  --device cpu \
  --imgsz 640 \
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

1. Validate source media, model path, numeric settings and calibration YAML when required.
2. Load only the frames needed for warm-up plus benchmark samples (bounded memory).
3. Warm up source and exported backends before recording timings.
4. Record separately:
   - backend inference median/P95 (`result.speed`)
   - prediction-call wall-time median/P95
   - measured end-to-end FPS from wall time
5. Never label `1000 / inference_ms` as end-to-end FPS.

## Prediction parity

The report includes `prediction_parity`: same-class greedy IoU matching on identical frames.
This is **not** validation accuracy. Empty-vs-empty frames are counted separately and do not
prove model quality.

## Artifact integrity

Reports include measured bytes, MiB, SHA-256 hashes and directory manifests. Size reduction is
computed from actual artifacts — never hard-coded marketing numbers.

## Raspberry Pi scope

Reports set:

```json
"hardware_scope": "current_machine_only",
"raspberry_pi_benchmarked": false
```

Run the same command on a Raspberry Pi to produce a device-specific report before making edge
performance claims.
