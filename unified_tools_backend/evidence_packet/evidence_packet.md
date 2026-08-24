# Evidence Packet

## Task 3 — Joint Ecosystem Production Certification

### System

Samachar

---

## Evidence Matrix

| Requirement | Evidence | Status |
| --- | --- | --- |
| Local execution | Samachar running on `127.0.0.1:8001` | PASS |
| Local Vision Runtime | Vision Runtime on `127.0.0.1:8080` | PASS |
| Local OCR | EasyOCR local models | PASS |
| Local inference | YOLO + EfficientNet local models | PASS |
| Replay MISS | First image execution | PASS |
| Replay HIT | Second identical image execution | PASS |
| Deterministic fingerprint | Same SHA-256 fingerprint | PASS |
| Trace propagation | trace ID | PASS |
| Execution ID | `EXEC-*` identifier | PASS |
| Vision replay | `46999cda-...` | PASS |
| Bucket persistence | Artifact successfully stored | PASS |
| Hash chaining | Parent hash → generated hash | PASS |
| Offline execution | Tested without LAN | PASS |
| Runtime observability | Runtime logs and IDs | PASS |

---

## Replay Evidence

### MISS

Input fingerprint:

`sha256:d773952582b10a72e331290d4e8a8de298ab5cac11808c2a067718d8035417a0`

### HIT

The same fingerprint produced:

`REPLAY HIT`

This confirms replay recognition for repeated identical input.

---

## Bucket Evidence

Bucket returned:

```json
{
  "success": true,
  "artifact_id": "a22329d8-9f61-41ef-bde7-b6e95c611b1b",
  "hash": "f231cd7bd50aef6af2bf942c1e1541a5aa23e3f3325d57fbe23f2f86f0bf06e2",
  "parent_hash": "14c81ca1b7a6aed0a6a36bfd1c29ee1a3eb6cfab2a6fc4d6cc21cd9896265f8e",
  "storage_type": "append_only"
}
```

## SVACS Evidence  

``` json  

{
    "replay_id": "03e4cbd1-9951-42d6-925f-9ffa8299cb83",
    "detections": [
        {
            "label": "LPG Carrier",
            "confidence": 0.9163,
            "bounding_box": {
                "x_min": 204.8849639892578,
                "y_min": 161.66220092773438,
                "x_max": 378.08465576171875,
                "y_max": 193.1523895263672
            },
            "top_predictions": [
                {
                    "class_name": "LPG Carrier",
                    "confidence": 91.63
                },
                {
                    "class_name": "Container Ship",
                    "confidence": 4.54
                },
                {
                    "class_name": "Oil Tanker",
                    "confidence": 2.69
                }
            ]
        }
    ],
    "ocr_results": [],
    "explainable_image_base64": null
}
```  

## Bucket evidence  

``` json
{
  "artifact": {
    "artifact_id": "30ba1533-0111-4e1b-9e9f-c2b1c6739b47",
    "trace_id": "SAM-62938002-136e-4c8e-abf6-c0d3c5c85246",
    "timestamp_utc": "2026-08-22T12:21:00.303158+00:00",
    "schema_version": "1.0.0",
    "source_module_id": "samachar",
    "artifact_type": "canonical_intelligence",
    "parent_hash": "44b4ee1f30d09228592d5f2669881d6e73ba9be95bc2cef2113babf5c0e2d873",
    "payload": {
      "schema_version": "1.0.0",
      "execution_id": "EXEC-35644708-68e6-4732-8364-05de2711ad99",
      "trace_id": "SAM-62938002-136e-4c8e-abf6-c0d3c5c85246",
      "timestamp": "2026-08-22T12:21:00.303158+00:00",
      "source": {
        "input_type": "image",
        "source_system": "samachar",
        "filename": "ship5.jpg"
      },
      "provenance": {
        "origin": "operator_image",
        "processed_by": [
          "samachar",
          "vision_runtime"
        ],
        "vision_runtime_invoked": true,
        "vision_replay_id": "b3f79532-4a8f-45eb-bdb2-7d4534d6bbff",
        "input_fingerprint": "sha256:d773952582b10a72e331290d4e8a8de298ab5cac11808c2a067718d8035417a0",
        "normalization": {
          "ocr_results_received": 3,
          "ocr_results_normalized": 3
        }
      },
      "vision_intelligence": {
        "replay_id": "b3f79532-4a8f-45eb-bdb2-7d4534d6bbff",
        "detections": [
          {
            "label": "Container Ship",
            "confidence": 0.5757,
            "bounding_box": {
              "x_min": 76.68743133544922,
              "y_min": 18.27726936340332,
              "x_max": 459.97467041015625,
              "y_max": 259.9080505371094
            },
            "top_predictions": [
              {
                "class": "Container Ship",
                "confidence": 57.57
              },
              {
                "class": "Passenger Ferry",
                "confidence": 36.8
              },
              {
                "class": "Cruise Ship",
                "confidence": 5.62
              }
            ]
          }
        ],
        "ocr_results": [
          {
            "text": "MERMAID",
            "confidence": 0.9959215842985021,
            "bounding_box": {
              "x_min": 122,
              "y_min": 85.03285434825837,
              "x_max": 425,
              "y_max": 210.96714565174162
            }
          }
        ],
        "normalized_ocr_results": [
          {
            "text": "MERMAID",
            "confidence": 0.9959215842985021,
            "source": "vision_runtime_ocr"
          }
        ],
        "explainable_image_base64": null
      }
    }
  }
}

