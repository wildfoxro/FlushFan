/*
  Stream onboard PDM mic to USB as 16 kHz mono int16 PCM.

  Board: Arduino Nano 33 BLE Sense (or Sense Rev2)
  Library: PDM (bundled with the Mbed Nano core)

  Pair with record_usb.py. Close Serial Monitor first.

  Host commands (single ASCII byte):
    R = start / restart (sends PDM16K01, then int16 LE PCM)
    S = stop (so the PC can close the port without wedging USB)

  RGB: red = waiting, green = streaming, blink red = ring overrun.
*/

#include <PDM.h>
#include <string.h>

constexpr int kSampleRate = 16000;
constexpr int kRingSamples = 8192;
constexpr char kMagic[] = "PDM16K01";

static int16_t ring[kRingSamples];
static volatile uint32_t widx = 0;
static volatile uint32_t ridx = 0;
static volatile uint32_t overflows = 0;
static bool streaming = false;
static uint32_t last_overflow_seen = 0;
static uint32_t blink_ms = 0;

static void leds_off() {
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
}

static void led_red() {
  leds_off();
  digitalWrite(LEDR, LOW);
}

static void led_green() {
  leds_off();
  digitalWrite(LEDG, LOW);
}

static void stop_stream() {
  streaming = false;
  led_red();
}

static void onPDMdata() {
  int bytes = PDM.available();
  if (bytes <= 0) {
    return;
  }
  static int16_t tmp[256];
  int samples = bytes / 2;
  if (samples > 256) {
    samples = 256;
  }
  PDM.read(tmp, samples * 2);

  uint32_t w = widx;
  uint32_t r = ridx;
  for (int i = 0; i < samples; i++) {
    uint32_t next = (w + 1) % kRingSamples;
    if (next == r) {
      overflows++;
      break;
    }
    ring[w] = tmp[i];
    w = next;
  }
  widx = w;
}

static void drain_to_usb() {
  // Do not use availableForWrite(): Mbed USB CDC on the Nano 33 often
  // returns 0 even when the host is reading, which sent zero PCM after the magic.
  uint8_t chunk[256];
  int n = 0;
  uint32_t r = ridx;
  const uint32_t w = widx;
  while (r != w && n + 2 <= (int)sizeof(chunk)) {
    const int16_t s = ring[r];
    r = (r + 1) % kRingSamples;
    chunk[n++] = (uint8_t)(s & 0xff);
    chunk[n++] = (uint8_t)((s >> 8) & 0xff);
  }
  ridx = r;
  if (n > 0) {
    Serial.write(chunk, n);
  }
}

static void handle_host() {
  while (Serial.available() > 0) {
    const int c = Serial.read();
    if (c == 'R') {
      ridx = widx;
      Serial.write(kMagic, 8);
      Serial.flush();
      streaming = true;
      led_green();
    } else if (c == 'S') {
      stop_stream();
    }
  }
}

void setup() {
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  led_red();

  Serial.begin(115200);

  PDM.onReceive(onPDMdata);
  PDM.setGain(127);
  if (!PDM.begin(1, kSampleRate)) {
    while (true) {
      led_red();
      delay(150);
      leds_off();
      delay(150);
    }
  }
}

void loop() {
  handle_host();
  if (!streaming) {
    return;
  }
  drain_to_usb();

  const uint32_t ov = overflows;
  if (ov != last_overflow_seen) {
    last_overflow_seen = ov;
    blink_ms = millis();
    led_red();
  } else if (blink_ms != 0 && millis() - blink_ms > 80) {
    blink_ms = 0;
    led_green();
  }
}
