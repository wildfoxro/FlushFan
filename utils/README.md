# utils

Helpers that are **not** part of Colab training. The board does not use these files.

```
utils/
  PdmUsbRecord/PdmUsbRecord.ino   USB mic streamer
  record_usb.py                   PC script → recordings/
  features.py                     optional: MFE constants matching the notebook / mfe.cpp
  generate_tables.py              optional: rewrites arduino/FlushFan/mfe_tables.h
  README.md
```

## USB recorder

This sketch **streams** PCM over USB. Same mic settings as the detector: **16 kHz**, mono, int16, `PDM.setGain(127)`.

1. Flash `utils/PdmUsbRecord/PdmUsbRecord.ino` (board: Nano 33 BLE Sense). No TensorFlow library.
2. Close Serial Monitor. If COM5 will not open, unplug USB, wait 2 s, plug back in.
3. LED **red** = waiting, **green** = streaming, short **red blinks** = ring overrun.

```
pip install pyserial
python record_usb.py --list-ports
python record_usb.py --port COM5 --seconds 8
python record_usb.py --port COM5 --seconds 8 --out bathroom.wav
```

Files are written to **`ToiletFlush/recordings/`** (same folder as `arduino/`, `colab/`, `utils/`), not wherever you launched the terminal. Each run increments: `bathroom_001.wav`, `bathroom_002.wav`, … Ctrl+C still saves what was received. Then flash **FlushFan** again.

| File | Role |
|---|---|
| `PdmUsbRecord/PdmUsbRecord.ino` | PDM → USB.  |
| `record_usb.py` | Writes 16 kHz mono WAVs into `recordings/`; auto-increments the filename |

## Optional: regenerate MFE tables

Skip this unless you change hop, FFT size, or mel count. Colab and `mfe.cpp` already match the current **250 × 32** setup.

1. Edit `features.py` and the MFE cell in `colab/colab_train_flush.ipynb` (same numbers).
2. From this folder: `python generate_tables.py` → writes `arduino/FlushFan/mfe_tables.h`.
3. Retrain in Colab and flash a new `model.h`.

| File | Role |
|---|---|
| `features.py` | Python copy of the log-mel recipe |
| `generate_tables.py` | Dumps Hann + packed mel weights into `mfe_tables.h` |
