# FlushFan — detector sketch

Arduino IDE sketch for the Nano 33 BLE Sense. Training is the Colab notebook, not this folder.

```
arduino/FlushFan/
  FlushFan.ino      open this in Arduino IDE
  model.h           int8 CNN (replace after Colab)
  flush_int8.tflite same model as model.h (optional; Colab also writes this)
  mfe.h             MFE constants and mfe_frame()
  mfe.cpp           1024-pt FFT + 32-band log-mel
  mfe_tables.h      Hann window and packed mel weights
```

## What the sketch does

1. PDM mic at 16 kHz, gain 127.
2. Every **32 ms** (512 new samples), build a **64 ms** FFT window (50% overlap) and keep **32** log-mel values. PCM is discarded. Those 32 values become the newest row of a **250 × 32** ring (about **8 s** of audio).
3. The first CNN run waits until that ring is full (~8 s from start). After that, inference is **not** once per 8 s: every **11 hops (~352 ms)** the latest 250 rows are quantized and run through TensorFlow Lite Micro. Older rows stay in the ring; only the newest 11 replace the oldest 11. Each score is a sliding 8 s window, shifted by ~352 ms.
4. Softmax → **p**. Green if `p ≥ 0.5`, else red (hysteresis: stays green until `p < 0.35`). With `FLUSH_SERIAL 1` (default), Serial prints `flush p=` on every inference (~3 times per second), not once per 8 s.

**RGB is active LOW** (LOW = on).

Set `#define FLUSH_SERIAL 0` in `FlushFan.ino` to compile out Serial and the timing / `fifo_peak` / `dropped` counters. FIFO overflow still discards samples; the LED path is unchanged.

## Arduino IDE

1. Board: **Arduino Nano 33 BLE Sense** (Mbed OS Nano Boards).
2. Library: **Arduino_TensorFlowLite**.
3. Open `FlushFan.ino` from this folder.
4. After Colab, overwrite **only** `model.h`. Leave `mfe.*` as they are.
5. Upload. Serial Monitor 115200 (only if `FLUSH_SERIAL` is 1).

If AllocateTensors fails, raise `kTensorArenaSize` in `FlushFan.ino` (currently 48 KB).

## Colab

1. Upload **`colab/colab_train_flush.ipynb` only**.
2. Run cells from the top.
3. Upload **`dataset/dataset_toiletflush_16k.zip`** (`positives/` = flush, `negatives/` = not flush). Prefer 16 kHz mono ~8 s; other rates are resampled.
4. Download `model.h` (and optionally `flush_int8.tflite`) → copy `model.h` over `arduino/FlushFan/model.h` → flash.

The notebook trains on your folder labels only. Confirm MFE shape `(…, 250, 32, 1)`.

To change hop or mel count, see `utils/README.md` (`features.py` / `generate_tables.py`). You do not need those files for a normal retrain.
