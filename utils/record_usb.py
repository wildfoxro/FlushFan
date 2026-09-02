"""
Record 16 kHz mono int16 PCM from a Nano 33 BLE Sense running PdmUsbRecord.

The board streams over USB; this script writes a WAV on the PC.

    pip install pyserial

    python record_usb.py --list-ports
    python record_usb.py --port COM5 --seconds 10
    python record_usb.py --port COM5 --seconds 3600 --out bathroom.wav

WAVs go in <workspace>/recordings/ (next to arduino/, colab/, utils/),
not the current working directory. Each run adds _001, _002, ...

Close the Arduino Serial Monitor first. Opening the port resets the board;
the script waits, then sends 'R' to start the stream.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import wave
from pathlib import Path

SAMPLE_RATE = 16000
MAGIC = b"PDM16K01"
WORKSPACE = Path(__file__).resolve().parent.parent
RECORD_DIR = WORKSPACE / "recordings"


def require_serial():
    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        sys.exit("pyserial is required:  pip install pyserial")
    return serial, list_ports


def list_serial_ports() -> None:
    _, list_ports = require_serial()
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    for p in ports:
        print(f"  {p.device:12}  {p.description}  {p.hwid}")


def guess_port() -> str | None:
    _, list_ports = require_serial()
    keys = ("arduino", "nrf", "nano 33", "usb serial", "vid:2341", "vid:1b4f")
    for p in list_ports.comports():
        blob = f"{p.description} {p.manufacturer} {p.hwid}".lower()
        if any(k in blob for k in keys):
            return p.device
    return None


def resolve_out(out: Path) -> Path:
    """Put relative names under workspace/recordings/. Absolute --out is unchanged."""
    out = Path(out)
    if not out.is_absolute():
        out = RECORD_DIR / out.name
    return next_indexed_path(out)


def next_indexed_path(out: Path) -> Path:
    """bathroom.wav / bathroom_001.wav -> bathroom_002.wav if 001 exists."""
    out = Path(out)
    suffix = out.suffix or ".wav"
    stem = out.stem
    numbered = re.match(r"^(.*)_(\d+)$", stem)
    if numbered:
        stem = numbered.group(1)
    parent = out.parent
    if str(parent) == "":
        parent = Path(".")
    pat = re.compile(rf"^{re.escape(stem)}_(\d+)$", re.IGNORECASE)
    max_i = 0
    if parent.exists():
        for p in parent.iterdir():
            if not p.is_file() or p.suffix.lower() != suffix.lower():
                continue
            m = pat.match(p.stem)
            if m:
                max_i = max(max_i, int(m.group(1)))
            elif p.stem.lower() == stem.lower():
                max_i = max(max_i, 0)
    return parent / f"{stem}_{max_i + 1:03d}{suffix}"


def wait_magic(ser, leftover: bytes = b"") -> bytes:
    buf = leftover
    deadline = time.time() + 8.0
    while time.time() < deadline:
        chunk = ser.read(4096)
        if chunk:
            buf += chunk
            i = buf.find(MAGIC)
            if i >= 0:
                return buf[i + len(MAGIC) :]
            if len(buf) > 65536:
                buf = buf[-32:]
        else:
            time.sleep(0.02)
    raise RuntimeError(
        "Did not receive PDM16K01. Reflash PdmUsbRecord, close Serial Monitor, "
        "unplug/replug USB if COM is stuck, then retry."
    )


def stop_and_close(ser) -> None:
    """Tell the board to stop TX before closing, or Windows CDC can stay busy."""
    try:
        ser.write(b"S")
        ser.flush()
        time.sleep(0.25)
        ser.reset_input_buffer()
    except Exception:
        pass
    try:
        ser.close()
    except Exception:
        pass
    time.sleep(0.5)


def open_serial(serial, port: str):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = 115200
    ser.timeout = 0.25
    ser.write_timeout = 1.0
    ser.dsrdtr = False
    ser.rtscts = False
    ser.open()
    return ser


def record(port: str, seconds: float, out: Path) -> None:
    serial, _ = require_serial()
    want_bytes = int(seconds * SAMPLE_RATE) * 2
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Opening {port} ...")
    ser = open_serial(serial, port)
    got = 0
    t0 = time.time()
    try:
        time.sleep(2.2)
        ser.reset_input_buffer()
        ser.write(b"R")
        ser.flush()
        leftover = wait_magic(ser)
        print(f"Streaming {seconds:g} s ({want_bytes / 1e6:.2f} MB) -> {out}")

        wav = wave.open(str(out), "wb")
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)

        buf = leftover
        t0 = time.time()
        last_print = t0
        last_data = t0
        try:
            while got < want_bytes:
                chunk = ser.read(4096)
                if chunk:
                    last_data = time.time()
                    buf += chunk
                    take = min(len(buf), want_bytes - got)
                    take -= take % 2
                    if take:
                        wav.writeframes(buf[:take])
                        got += take
                        buf = buf[take:]
                now = time.time()
                if now - last_print >= 1.0:
                    audio_s = got / (SAMPLE_RATE * 2)
                    print(f"  {audio_s:7.1f} s / {seconds:g} s  ({got / 1e6:.2f} MB)")
                    last_print = now
                    if got == 0 and now - last_data >= 3.0:
                        print("  still no PCM — reflash PdmUsbRecord.ino if this continues")
                        last_data = now
        except KeyboardInterrupt:
            print("\nStopped early (Ctrl+C).")
        finally:
            wav.close()
    finally:
        stop_and_close(ser)

    if got == 0:
        print("Got handshake but no audio. Reflash utils/PdmUsbRecord/PdmUsbRecord.ino and retry.")
        return
    audio_s = got / (SAMPLE_RATE * 2)
    wall = time.time() - t0
    print(f"Wrote {out}  ({audio_s:.2f} s of audio, {wall:.1f} s wall time)")
    if got < want_bytes:
        print("File is shorter than requested — board unplugged or USB stalled.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", help="COM port, e.g. COM5 (Windows) or /dev/ttyACM0")
    p.add_argument("--seconds", type=float, default=10.0, help="audio length to capture (default 10; use 3600 for 1 hour)")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("nano_record.wav"),
        help="base filename; saved under workspace/recordings/ as _001, _002, ...",
    )
    p.add_argument("--list-ports", action="store_true")
    args = p.parse_args()

    if args.list_ports:
        list_serial_ports()
        return

    port = args.port or guess_port()
    if not port:
        list_serial_ports()
        sys.exit("Pass --port COMx  (use --list-ports to see names)")

    if args.seconds <= 0:
        sys.exit("--seconds must be > 0")

    record(port, args.seconds, resolve_out(args.out))


if __name__ == "__main__":
    main()
