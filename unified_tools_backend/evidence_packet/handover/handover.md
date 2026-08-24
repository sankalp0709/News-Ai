# HANDOVER

## Samachar / Guptachar Runtime

### BHIV Ecosystem Production Certification

---

## 1. Runtime

Samachar/Guptachar acts as the canonical intelligence ingestion runtime.

Primary API:

`POST /api/v1/intelligence/image`

---

## 2. Local Dependencies

The demonstrated offline runtime uses:

- Local Python environment
- Local Vision Runtime
- Local OCR models
- Local YOLO model
- Local EfficientNet classifier
- Local Replay Store
- Local Bucket runtime
- Local Svacs

---

## 3. Runtime Flow

```text
Operator
   ↓
Samachar / Guptachar
   ↓
Local Vision Runtime
   ↓
OCR / Vessel Inference
   ↓
Replay Store
   ↓
Canonical Intelligence
   ↓
Svacs
   ↓
Bucket
   ↓
Tantra 
