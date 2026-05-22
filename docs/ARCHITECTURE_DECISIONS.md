# Architecture Decision Records

Major architectural decisions made during the 7-day build window of the webAI
YOLO26 MLX Build Challenge. ADR format: context, options considered, decision,
rationale, tradeoff.

---

## ADR-001 · Single-file Python deliverable

**Context.** 7-day build window for a hackathon demo, not a maintained product.

**Options considered.** Multi-module package (`app/inference`, `app/ui`, `app/triage`); single-file script.

**Decision.** Single file (`webcam_demo.py`).

**Rationale.** A one-file deliverable is auditable end-to-end in under five minutes. Module boundaries impose cognitive cost without proportional value at this scale.

**Tradeoff.** Won't scale past ~500 lines. Acceptable for a demo; the roadmap notes a refactor as the first step toward productisation.

---

## ADR-002 · Synchronous single-thread capture loop

**Context.** Camera capture and inference on macOS 26 with OpenCV's AVFoundation backend.

**Options considered.** Async pipeline (FastAPI + WebSocket + ThreadPoolExecutor for camera); synchronous loop (`while True: cap.read() -> model.predict()`).

**Decision.** Synchronous loop.

**Rationale.** An async pipeline with camera capture in an executor thread surfaced intermittent SIGTRAP crashes on macOS 26, caused by OpenCV's `CaptureDelegate` releasing CoreFoundation objects from a non-main thread. The crash trace consistently bottomed out at `CFRelease` from `cv2.abi3.so -[CaptureDelegate captureOutput:...]`. The synchronous loop sidesteps the issue entirely while sustaining 16 FPS at 720p on an M4.

**Tradeoff.** No backend/frontend separation, no remote control surface. Acceptable: the demo is a single end-user application, not a deployed service.

---

## ADR-003 · Heuristic posture classifier

**Context.** Triage classification logic with a 7-day timeline and uncertain dataset availability.

**Options considered.** Fine-tune YOLO26 head on a triage-specific posture dataset; bounding-box aspect-ratio heuristic with three thresholds.

**Decision.** Heuristic (`height / width` against {1.6, 1.0}).

**Rationale.** An attempted fine-tune on a 5,688-image Roboflow posture dataset collapsed early in training (loss to zero by epoch 2, mAP 0.0 throughout 8 epochs); without time to diagnose dataset/loss-function issues, the heuristic is explainable, deterministic, and good enough to validate the user experience. The classification logic is one function (`classify_posture`) that any reviewer can audit in five seconds.

**Tradeoff.** The heuristic fails when the detected bounding box does not cover the full body — e.g., webcam close-ups where only the upper body is visible. Documented in the README and addressed in the roadmap.

---

## ADR-004 · `yolo26n` model variant

**Context.** Trading model quality vs inference latency on Apple M4.

**Options considered.** `yolo26n` (nano, 3.3M params); `yolo26s/m/l/x` (larger); a fine-tuned custom model.

**Decision.** `yolo26n` pretrained.

**Rationale.** Person detection is COCO class 0 — pretrained YOLO26 handles it well without domain-specific fine-tuning. The nano variant sustains the demo's primary UX constraint (real-time framerate) at 16 FPS on 720p input.

**Tradeoff.** Lower recall on partially-occluded or distant subjects vs `yolo26s/m`. Acceptable for the demo scenario.

---

## ADR-005 · OpenCV overlay rendering for the dashboard UI

**Context.** Tactical-dashboard look-and-feel within a single-file deliverable.

**Options considered.** PySide6 / Tkinter native window with proper layout; HTML+JS browser UI driven via WebSocket; OpenCV `imshow` with overlay-rendered UI.

**Decision.** OpenCV `imshow` with overlay rendering.

**Rationale.** Keeps the entire pipeline in one file with one dependency stack. A native UI toolkit would add ~50 MB and rendering complexity for marginal demo benefit. The browser UI option was independently ruled out by ADR-002.

**Tradeoff.** OpenCV graphics primitives are limited — no rounded corners, no antialiased fills beyond what is computed manually. Acceptable for the intentionally angular "ops dashboard" aesthetic.

---

## ADR-006 · Apache 2.0 license

**Context.** Code licensing for public release.

**Options considered.** MIT, Apache 2.0, GPL-3.0.

**Decision.** Apache 2.0.

**Rationale.** Apache 2.0 provides explicit patent grant in addition to MIT's permissive terms, which matters for any downstream commercial or defense deployment. GPL-3.0 ruled out as too restrictive for the intended use case.

**Tradeoff.** Slightly longer license header than MIT. Negligible.

---

## What is NOT an ADR

Decisions not material enough to warrant their own ADR, documented here for completeness:

- **No tracking / no IDs across frames.** Each frame is classified independently. Acceptable for the demo; documented in the roadmap as a follow-up.
- **Hardcoded confidence threshold of 0.40.** Conservative default; would be parameterised in a production version.
- **No telemetry, no analytics, no remote logging.** Aligns with the zero-network-egress claim.
