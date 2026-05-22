# SENTINEL

**On-device visual triage AI. Zero network egress.**

Submission for the [webAI YOLO26 MLX Build Challenge — May 2026](https://community.webai.com/t/the-yolo26-mlx-build-challenge-may-2026/16).

> **Watch the 60-second demo:** [TBD-VIDEO-URL]

---

## TL;DR

SENTINEL is a real-time visual triage classifier that runs entirely on Apple
Silicon. It detects every person in frame using YOLO26 (MLX format) and
classifies each one into one of three triage categories by body posture:

- **T1 — IMMEDIATE** — person lying down (red)
- **T2 — DELAYED** — person sitting (orange)
- **T3 — AMBULATORY** — person standing (green)

Designed for mass-casualty scenarios where network connectivity is unavailable,
compromised, or operationally forbidden: tactical environments, disaster zones,
austere medical settings, secured facilities.

**Zero network egress, by design.** The entire pipeline runs locally on the
device. The demo video was recorded with WiFi disabled.

---

## Submission details

| | |
|---|---|
| **Track** | Enterprise |
| **Team size** | 2 |
| **Hardware** | MacBook Pro 14-inch (Nov 2024), Apple M4, 32 GB RAM |
| **OS** | macOS 26.3 |
| **Model variant** | `yolo26n` |
| **Measured FPS** | ~16 at 720p |

---

## Why on-device matters

| | Cloud-based vision AI | SENTINEL (on-device) |
|---|---|---|
| Network dependency | Required | None |
| Works in faraday-shielded environments | No | Yes |
| End-to-end latency | 100–500 ms round-trip | < 70 ms local |
| PHI / PII exposure to third parties | Yes | Zero |
| Operates in denied-comms scenarios | No | Yes |
| Subject to cloud-vendor terms / outages | Yes | No |

---

## The triage logic

YOLO26 detects every person in frame (COCO class 0). For each detection,
SENTINEL applies a posture heuristic based on the bounding-box aspect ratio:

| Triage | Posture | Aspect ratio (h/w) | Visual color |
|---|---|---|---|
| **T3** | Standing | > 1.6 | Green |
| **T2** | Sitting | 1.0 – 1.6 | Orange |
| **T1** | Lying | < 1.0 | Red |

The classifier is intentionally simple and explainable — no black-box behavior
on a safety-relevant classification. The heuristic runs per detection, so
multiple people in frame are classified independently. See
[docs/ARCHITECTURE_DECISIONS.md](docs/ARCHITECTURE_DECISIONS.md) for the
reasoning, and the [Roadmap](#roadmap) for the path to a fine-tuned posture
model.

---

## Run locally

Requires Python 3.10+, Apple Silicon Mac (M1 or newer), macOS 14+.

```bash
git clone https://github.com/nomad-link-id/sentinel-mlx.git
cd sentinel-mlx
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p models
hf download webAI-Official/yolo26n-mlx yolo26n.npz --local-dir models
python webcam_demo.py
```

Press `Q` in the window to quit.

**To verify zero network egress:** disable WiFi before running. The pipeline
continues uninterrupted.

---

## Architecture

Single-file Python application. No services, no servers, no async, no IPC.

| Component | Implementation |
|---|---|
| Detection model | YOLO26-nano (MLX format, `webAI-Official/yolo26n-mlx`) |
| Inference engine | `mlx` 0.30.6 + `mlx-metal` (Apple Silicon GPU/ANE) |
| Camera capture | `opencv-python` 4.13 (AVFoundation backend) |
| Posture classifier | Heuristic (bbox aspect ratio) |
| UI / dashboard | OpenCV overlay rendering |
| Concurrency model | Single-thread synchronous loop |

The complete rationale for these choices is in
[docs/ARCHITECTURE_DECISIONS.md](docs/ARCHITECTURE_DECISIONS.md).

---

## Roadmap (post-hackathon)

| | |
|---|---|
| **Fine-tuned posture model** | Replace the aspect-ratio heuristic with a posture classifier trained on triage-specific data (prone, supine, recovery position, seated-slumped, etc.) |
| **Multi-camera fusion** | Aggregate triage state across devices on a local mesh — still zero internet egress |
| **Vital-sign overlay** | rPPG (heart rate from skin pixels) for T1 confirmation |
| **Spatial audio cueing** | Alerts for new T1 detections during multi-victim incidents |
| **iOS / iPadOS port** | MLX runs on iPhone and iPad — same pipeline, mobile form factor |
| **Clinical validation pathway** | Engage with relevant regulatory bodies for any production deployment in medical settings |

---

## Team

**[@nomad-link-id](https://github.com/nomad-link-id)** — Engineering and
implementation. Background in healthtech and AI infrastructure.

**Lexi Armstrong** — Industrial Security Specialist with 8+ years in US
defense contracting at Booz Allen Hamilton, Boeing, and Leidos. US Navy
veteran. Domain framing and use-case validation.

---

## Disclaimer

> SENTINEL is a research demonstration of on-device computer vision. **It is
> not a medical device.** It is not cleared by the FDA, ANVISA, CE-MDR, or
> any regulatory authority and must not be used for clinical triage decisions.
> Any production deployment in a medical or emergency-response setting would
> require regulatory clearance, clinical validation by qualified medical
> professionals, and integration with appropriate human-in-the-loop safety
> protocols.

---

## License

[Apache 2.0](LICENSE).

---

## Acknowledgments

- [**webAI**](https://webai.com) — for YOLO26-MLX, the MLX-format weights, and the build challenge.
- [**Ultralytics**](https://github.com/ultralytics/ultralytics) — for the YOLO architecture lineage.
- [**Apple MLX team**](https://github.com/ml-explore/mlx) — for making on-device inference this performant on consumer hardware.

---

*Built for the [webAI YOLO26 MLX Build Challenge — May 2026](https://community.webai.com/t/the-yolo26-mlx-build-challenge-may-2026/16). Enterprise track. Co-hosted by webAI, HackAI, and AITX + Antler.*
