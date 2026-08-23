# DrishtiPath presentation runbook

Use this checklist for a repeatable judge demonstration. It deliberately separates model
proposals from verified evidence and keeps an offline fallback ready.

## Before entering the venue

1. Put the trained road model at `models/road_hazards.pt`.
2. Put the two frozen regression clips under `clips/`:
   - a positive road-damage clip;
   - `2.mp4`, the no-number-plate clip with several visible potholes.
3. Start the dashboard and run both clips once so model downloads and video codecs are warm.
4. Preserve the generated `artifacts/ui_runs/` directories. Do not edit their metrics.
5. Verify the offline-safe 3D renderer before disconnecting the network.

```bash
source .venv/bin/activate
python -m pytest
python -m ruff check .
streamlit run dashboard.py
```

## Three-minute live story

1. **New scan:** drag in a dashcam clip, select **High recall**, and start a full scan.
2. **Mission control:** scrub through the route and show that evidence appears at the correct
   video time and GPS position.
3. **Evidence review:** open a hazard, compare its context frame and crop, then mark it
   confirmed, rejected, or requiring field inspection.
4. **Operations:** show vehicle density, occupancy, bottleneck episodes, and the fleet map.
5. **Engineering:** only if asked, show measured edge latency, evidence bytes, offline delivery,
   and the virtual-Pi envelope. Never label a Mac or simulated result as a physical-Pi test.

## Frozen `clips/2.mp4` regression

Run the road model in the high-recall presentation profile:

```bash
python orchestrator.py \
  --input clips/2.mp4 \
  --gps gps_data.csv \
  --model models/road_hazards.pt \
  --output-dir artifacts/regression_clip2_road \
  --classes pothole,longitudinal_crack,transverse_crack,alligator_crack \
  --confidence 0.05 \
  --frame-skip 1 \
  --image-size 768 \
  --inference-iou 0.70 \
  --augment \
  --road-roi-top 0.25 \
  --window-size 7 \
  --min-hits 2
```

Run ANPR with the strict multi-frame quality gate:

```bash
python anpr_pipeline.py \
  --input clips/2.mp4 \
  --gps gps_data.csv \
  --gps-source-type synthetic_demo \
  --output-dir artifacts/regression_clip2_anpr \
  --confidence 0.35 \
  --sample-fps 5 \
  --min-observations 4 \
  --min-winning-votes 3 \
  --min-mean-ocr-confidence 0.85 \
  --min-vote-ratio 0.60 \
  --min-mean-detector-confidence 0.45
```

Acceptance criteria:

| Check | Required result |
|---|---|
| ANPR confirmed tracks | `0` |
| ANPR plate-like proposals | May be non-zero; clearly labelled as proposals |
| Road visible-pothole recall | Manually count confirmed true potholes / labelled visible potholes |
| False hazards | Review and export as hard negatives |
| GPS | Explicitly labelled `synthetic_demo` unless real telemetry is supplied |

If pothole recall remains low, do not hide it by lowering thresholds indefinitely. Label missed
frames from multiple routes, add them to a separate training split, retrain, and compare against
a frozen validation set. Threshold tuning is not a substitute for representative data.

## Judge-safe language

- Say **plate-like proposals** for raw ANPR boxes and **verified track** only after all gates pass.
- Say **candidate incident** and **pending human review**, not a legally proven offence.
- Say **timestamp-interpolated GPS** and state whether telemetry is real or synthetic.
- Report validation precision, recall, and mAP; never call a confidence score “accuracy.”
- State the named benchmark device. A virtual-Pi envelope is capacity evidence, not physical
  Raspberry Pi validation.
