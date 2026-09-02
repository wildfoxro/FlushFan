# Toilet flush TinyML (Nano 33 BLE Sense)

On-device detector so a **bathroom fan can start after a toilet flush** (clear the air) without a humidity sensor or a wall switch. Occupancy and humidity already cover “someone is in the room” and showers; this targets the flush itself.

The sketch runs on an **Arduino Nano 33 BLE Sense**. Training is in **Colab**; the board only runs inference. This repo drives the built-in RGB LED as a stand-in for the fan: **green = flush** (fan would start), **red = not flush**. No relay is wired yet.

```
ToiletFlush/
  LICENSE
  README.md
  arduino/FlushFan/     detector sketch (flash this to run the model)
  colab/                Colab notebook only
  dataset/              training zip (~38 MB)
  utils/                USB recorder + optional MFE table generator
```

Current training set: **`dataset/dataset_toiletflush_16k.zip`** (`positives/` flush, `negatives/` not flush). See `dataset/README.md`. Unzip **once** — the zip already contains the `dataset_toiletflush_16k/` folder.

## What to flash

## What to flash

| Path | Use |
|---|---|
| `arduino/FlushFan` | Flush detector |
| `utils/PdmUsbRecord` | Record wavs from the onboard mic over USB |

Only one sketch can be on the board at a time. After recording, flash FlushFan again.

## Typical flow

1. (Optional) Record your own bathroom audio: see `utils/README.md`.
2. Put clips in `positives/` and `negatives/` inside the unzipped `dataset_toiletflush_16k` folder, then zip **that folder** again (so the zip root is `dataset_toiletflush_16k/positives/`, not a nested extra folder).
3. Train: upload **only** `colab/colab_train_flush.ipynb` to Colab, then upload **`dataset/dataset_toiletflush_16k.zip`**. See `arduino/FlushFan/README.md`.
4. Copy the new `model.h` into `arduino/FlushFan/`, compile, flash.

## Hardware / software

- Board: Arduino Nano 33 BLE Sense (or Sense Rev2)
- Arduino IDE: **Arduino Mbed OS Nano Boards**
- Detector library: **Arduino_TensorFlowLite**
- Recorder: **PDM** (bundled with the board core)
- PC recorder: `pip install pyserial`

## Model input (Colab and board must match)

8 s of 16 kHz audio → **250 × 32** log-mel (32 ms hop, 64 ms FFT, 50% overlap).

## Limits

- LED only; no fan relay is wired.
- The included `model.h` was trained on the **mixed** zip (Freesound + Nano clips), not a Nano-only set. Nineteen on-device flushes alone are not enough.
- Freesound flushes are often louder than the onboard mic. Bathroom LED behavior can differ from Colab val accuracy.
- Scores update every ~352 ms on a sliding 8 s window, not once per 8 s.

## License

- **Code** (Arduino sketches, Colab notebook, `utils/`, generated `model.h` / `flush_int8.tflite`): [MIT](LICENSE). Arduino, TensorFlow Lite Micro, and other libraries you install stay under their own licenses.
- **Bathroom WAV clips** recorded in the author’s bathroom with the Arduino Nano 33 BLE Sense microphone: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). See `dataset/README.md`.
- **Freesound clips** (`Freesound.org__…` filenames): from [Freesound](https://freesound.org/), [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). See `dataset/README.md`.
