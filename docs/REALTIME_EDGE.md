# Real-time edge deployment

`edge_agent.py` is the boundary between a live dashcam and DrishtiPath evidence. It is
designed to answer two separate questions honestly:

1. Can the camera capture continuously?
2. How many fresh frames can each analytics capability process on this device?

It never presents the camera capture rate as the AI inference rate.

## Runtime topology

```mermaid
flowchart TD
    A["USB/UVC dashcam"] --> B["Bounded newest-frame buffer"]
    G["GPS provider"] --> C["Priority scheduler"]
    B --> C
    C --> D["Resident model adapters"]
    D --> E["Temporal confirmation"]
    E --> F["Atomic evidence outbox"]
    D --> H["Device status + metrics"]
```

The capture thread owns `VideoCapture`. The inference worker owns every model. This avoids
six Python processes independently loading large runtimes and competing for the CPU. If
inference falls behind, the buffer keeps the newest frame and records how many stale
analysis frames were dropped. Source recording, if required by policy, is a separate local
retention responsibility.

## Capability budget

| Capability | Current live state | Activation strategy |
|---|---|---|
| Traffic objects and persons | Implemented | Always, mixed rate |
| Road damage | Implemented | Always, mixed rate + 3-of-5 confirmation |
| Urban assets / waterlogging | Executable visual + inventory pipeline; custom weights required | Low-rate/geofenced |
| Missing asset condition | Inventory-aware absence candidate | No additional model |
| Vulnerable pedestrian situation | Person/vehicle track conflict + school geofence | No additional model |
| ANPR | Reviewed event/upload pipeline | Event-triggered or retained clip |

The six capabilities do not run as six independent models. Inventory, vulnerable-risk,
motion and incident logic consume detections from the three resident models. ANPR remains
an event/review workload.

## Dashcam preflight

Many consumer dashcams expose only storage when connected through USB. On Raspberry Pi,
confirm a live UVC device before changing application code:

```bash
v4l2-ctl --list-devices
ls -l /dev/video*
```

Use `--source 0` only when the camera appears as `/dev/video0`. Otherwise use a supported
HDMI-to-USB capture adapter, a CSI camera, an RTSP URL, or a recorded route replay labelled
as replay.

## GPS modes

Live serial NMEA telemetry:

```bash
--gps-nmea-device /dev/ttyUSB0 --gps-source-type real_telemetry
```

Timestamped GPS replay or telemetry:

```bash
--gps-csv route_gps.csv --gps-source-type real_telemetry
```

Fixed coordinate for an explicitly labelled table demo:

```bash
--fixed-gps 28.6139,77.2090 --gps-source-type synthetic_demo
```

The fixed mode must never be described as real bus telemetry. Serial/NMEA ingestion is the
live provider adapter; GPS interpolation and provenance fields use the shared core.

## Virtual Pi test boundary

`virtual_pi_test.py` provides a deterministic pre-hardware integration mission. It injects
synthetic camera frames, Pi-like resource telemetry, deterministic model latencies and valid
NMEA RMC sentences while retaining the production scheduler, temporal filter, outbox,
delivery client, fleet store and field validator.

The resulting report separates what it proves from what it cannot prove:

| Verified by the virtual mission | Still requires physical hardware |
|---|---|
| Capture/inference concurrency and load shedding | ARM runtime compatibility and speed |
| Mixed-rate scheduler and compute-budget math | USB/UVC camera compatibility |
| Serial NMEA parsing and GPS provenance | UART/USB electrical stability and antenna fix |
| Temporal evidence, outbox and retry delivery | Sustained temperature, throttling and power |
| Authenticated idempotent fleet ingestion | Road vibration, lighting and field accuracy |

The field validator requires both Raspberry Pi detection and
`execution_environment: physical_device`. Container, VM and emulated results can therefore
never be promoted into a physical field claim by changing only the model name.

## Geofences

`--geofences config/geofences.example.json` accepts a list of objects:

```json
[
  {
    "key": "school-zone-demo",
    "latitude": 28.6139,
    "longitude": 77.209,
    "radius_m": 150
  }
]
```

The runtime records active context keys. Later triggered/cascade adapters use these keys to
avoid running school-zone or expected-asset logic across the whole route.

## Outputs

Each mission directory contains:

- `detections.ndjson` — compact geotagged observations from every scheduled model pass
- `events.ndjson` — temporally confirmed road/asset events only
- `evidence/` — context and crop JPEGs for confirmed events
- `outbox/pending/` — atomic evidence packets awaiting an authenticated uploader
- `device_status.json` — refreshable privacy-safe device telemetry
- `metrics.json` — final capture, scheduler, latency, error, queue and load report

The outbox stores no credentials. `deliver_outbox.py` reads its token from an environment
variable, retries failures, and moves a packet from `pending/` to `sent/` only after a 2xx
acknowledgement from `fleet_server.py`.

## Tuning rule

For a single inference worker:

```text
estimated utilization = Σ(target FPS × measured mean latency ms) / 1000
```

If the result exceeds 1.0, the requested schedule is mathematically overloaded. In field
use, leave headroom for decoding, tracking, evidence encoding, temperature, and uploads.
Reduce target rates, resolution, or activation scope and run the same named-device test
again. Do not publish a Raspberry Pi FPS figure measured on a Mac or Colab GPU.
