# Runtime Dependency Matrix

| Component | Purpose | Local Required |
|---|---|---:|---:|---|
| Samachar | Intelligence runtime | YES |
| Vision Runtime | Image inference | YES |
| YOLO model | Vessel detection | YES |
| EfficientNet model | Vessel classification | YES |
| EasyOCR models | OCR | YES |
| Replay Store | Replay | YES |
| Bucket | Artifact persistence | YES |
| SVACS | Ecosystem integration | YES for offline certification |
| TANTRA | Runtime execution | YES for offline certification |

---

## Mandatory Runtime Assets

The following assets must exist locally:

- Python environment
- Python dependencies
- Vision Runtime source
- OCR models
- YOLO model
- EfficientNet model
- Replay storage
- Bucket runtime
- Svacs runtime source
- Configuration files

---

## Prohibited Mandatory Dependencies

The certified offline runtime must not require:

- ngrok
- Render
- Vercel
- Cloud inference
- Remote OCR
- Remote model downloads

---

## Certification Rule

If any mandatory runtime stage fails when network connectivity is removed, the stage is not considered sovereign-offline certified.
