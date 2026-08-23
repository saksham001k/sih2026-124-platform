# SIH 26124 coverage and claim boundary

This matrix is the source of truth for demos and judging. “Implemented” means the software
path and tests exist. It does not mean a model has been validated for every Indian road,
weather, camera, or Raspberry Pi.

| Requirement | Implementation | Runtime model/load | Evidence boundary |
|---|---|---|---|
| Potholes and road damage | `orchestrator.py`, RDD2022 preparation/training, temporal and spatial dedup | One road-damage model | Use validation mAP, never confidence as accuracy |
| Waterlogging and damaged assets | `asset_inspection.py`, `train_urban_assets.py` | One low-rate urban-assets model | Custom labelled weights still required |
| Missing divider/crossing/sign | GIS `AssetInventoryInspector` | No extra model beyond visible-assets model | Only camera-visible inventory entries; human review |
| Vehicle detection/classification/count | `traffic_analytics.py`, ByteTrack | Generic traffic model | Untracked boxes excluded from unique totals |
| Density and bottlenecks | ROI occupancy + fixed windows + episode detector | Lightweight rules | Image-space proxy, not municipal calibration |
| Vulnerable pedestrian situations | Reuses person/vehicle tracks + crossing ROI + school geofence | No additional resident model | Does not infer a person’s age or “school child” identity |
| Rash driving | Track-motion candidate rules | No additional resident model | No legal speed or intent claim |
| Suspected hit-and-run | Proximity/departure candidate + masked ANPR correlation | Event/review pipeline | Investigative candidate; authority review required |
| ANPR | FastALPR multi-frame consensus and evidence | Event-triggered/offline stage | Plate masked in dashboard; full text remains local |
| Live dashcam and GPS | Newest-frame edge agent + serial NMEA | Three resident models maximum | Capture FPS is separate from analytics FPS |
| Low-bandwidth transfer | Atomic outbox + authenticated retries + idempotent server | Metadata/evidence packets | Full video remains local unless policy requests a clip |
| Fleet GIS and deficiencies | SQLite ingestion, cross-bus clusters, GeoJSON export | Central lightweight service | Spatial cluster is not automatic municipal verification |
| Origin–destination patterns | Mission start/end aggregation by route | Central lightweight service | Bus mission OD, not passenger OD |
| Raspberry Pi proof | Runtime telemetry + `field_validate.py` claim gate | Device measured | Not verified until a physical run passes every gate |

## Runtime model budget

The live agent keeps at most three models resident:

1. traffic/person detection at the highest mixed rate;
2. road damage at a lower mixed rate;
3. urban assets/waterlogging at a low or geofenced rate.

Missing-infrastructure logic, density windows, safety candidates, incident correlation,
geospatial deduplication, and OD aggregation are rules over existing detections. ANPR is an
event/review workload. Six independent models are neither required nor loaded on every frame.

## Acceptance gates still requiring real-world inputs

Two proof items cannot be manufactured by repository code:

- train and validate `models/urban_assets.pt` on licensed, India-relevant labels;
- run the named Pi with its actual camera and serial GPS, then pass `field_validate.py`.

Until those are supplied, say “software path implemented; field/model validation pending.”
