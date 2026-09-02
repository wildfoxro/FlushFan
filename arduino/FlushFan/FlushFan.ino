/*
  Flush detector — Arduino Nano 33 BLE Sense (LED only).

  Pipeline:
    16 kHz PDM mic
      -> ISR packs samples into 32 ms hops (512 samples) in a 12-slot FIFO
      -> loop() pops a hop, overlaps it with the previous hop (64 ms FFT window)
      -> one 32-band log-mel row per hop into a 250-row ring (~8 s of audio)
      -> after the ring fills once, run the int8 CNN every 11 hops (~352 ms)
         on the latest 8 s (sliding window, not a new clip from scratch)

  If the FIFO is full, the ISR discards samples (must not overwrite unread hops).

  FLUSH_SERIAL 1 = Serial logs + timing / fifo_peak / dropped (diagnostics only).
  FLUSH_SERIAL 0 = compile those out; LED path is unchanged.

  Built-in RGB (active LOW): green = flush (p >= 0.5), red = not flush.
  LED hysteresis: stay green until p < 0.35 so the color does not flicker.
*/

#include <PDM.h>
#include <math.h>
#include <string.h>

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "mfe.h"     // kHopSamples=512, kFrameSamples=1024, kNFrames=250, kNMels=32
#include "model.h"   // g_flush_model[] from Colab (replace this file after retrain)

// 1 = Serial Monitor stats. 0 = no Serial, no micros() timing, no drop/peak counters.
#ifndef FLUSH_SERIAL
#define FLUSH_SERIAL 1
#endif

// CNN working RAM (activations). Weights live in flash in model.h.
constexpr int kTensorArenaSize = 48 * 1024;
// New spectrogram rows between CNN runs. 11*32 ms = 352 ms of new audio.
// 10 hops of backlog during Invoke() will fill any FIFO; 11 is the stable setting.
constexpr int kInferEveryHops = 11;
// ISR queue of complete hops. 12*32 ms ≈ 384 ms, enough for ~260 ms Invoke() plus jitter.
constexpr int kHopSlots = 12;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static tflite::MicroErrorReporter micro_error_reporter;
static tflite::AllOpsResolver resolver;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;   // int8 250x32 log-mel
static TfLiteTensor* output_tensor = nullptr;  // int8 logits: [not-flush, flush]

// --- Hop FIFO: ISR writes complete 512-sample hops; loop() reads them. ---
static int16_t hop_fifo[kHopSlots][kHopSamples];
static volatile uint8_t hop_w = 0;  // next slot the ISR will fill
static volatile uint8_t hop_r = 0;  // next slot loop() will pop
static volatile uint8_t hop_n = 0;  // complete hops waiting (0 .. kHopSlots)
#if FLUSH_SERIAL
static volatile uint8_t hop_n_max = 0;      // peak hop_n in this infer interval
static volatile uint32_t hops_dropped = 0;  // hops discarded while FIFO was full
#endif

// Overlap buffer: last two hops concatenated = 1024 samples (64 ms) for the FFT.
static int16_t fft_window[kFrameSamples];
static int hop_count = 0;  // need 2 hops before the first MFE row is valid

// Rolling 8 s spectrogram. spec_write is the next row to overwrite.
// After wrap, spec_write is also the oldest live row (passed to quantize).
static float spectrogram[kNFrames][kNMels];
static int spec_write = 0;
static int hops_since_infer = 0;  // rows added since last CNN; fire at kInferEveryHops
static bool ring_full = false;    // false until 250 rows exist (~8 s from boot)

#if FLUSH_SERIAL
static uint32_t hop_us_sum = 0;
static uint32_t hop_us_max = 0;
static int pdm_drop_fill = 0;  // discarded samples while FIFO is full
#endif

static int pdm_count = 0;       // samples written into the current hop slot (0..511)

// RGB is active LOW: HIGH = off, LOW = on.
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

// PDM DMA callback. Keep it short: copy samples into hop_fifo, never run MFE/CNN here.
static void onPDMdata() {
  int bytes = PDM.available();
  if (bytes <= 0) {
    return;
  }
  // Cap this callback so the ISR does not run too long; leftover bytes stay in the driver.
  static int16_t tmp[256];
  int samples = bytes / 2;
  if (samples > 256) {
    samples = 256;
  }
  PDM.read(tmp, samples * 2);

  for (int i = 0; i < samples; i++) {
    // FIFO full and not mid-hop: must discard. Do not overwrite an unread hop.
    if (pdm_count == 0 && hop_n >= kHopSlots) {
#if FLUSH_SERIAL
      if (++pdm_drop_fill >= kHopSamples) {
        pdm_drop_fill = 0;
        hops_dropped++;
      }
#endif
      continue;
    }
#if FLUSH_SERIAL
    pdm_drop_fill = 0;  // a successful write ends a drop streak
#endif
    hop_fifo[hop_w][pdm_count++] = tmp[i];
    if (pdm_count < kHopSamples) {
      continue;  // hop slot not full yet
    }
    // 512 samples in: hop is complete. Advance write index; leftover tmp[] starts the next hop.
    pdm_count = 0;
    hop_w = (uint8_t)((hop_w + 1) % kHopSlots);
    hop_n++;
#if FLUSH_SERIAL
    if (hop_n > hop_n_max) {
      hop_n_max = hop_n;
    }
#endif
  }
}

// Copy one complete hop out of the FIFO. False if empty (loop() should idle).
static bool pop_hop(int16_t* dst) {
  noInterrupts();  // ISR also touches hop_n / hop_r
  if (hop_n == 0) {
    interrupts();
    return false;
  }
  memcpy(dst, hop_fifo[hop_r], kHopSamples * sizeof(int16_t));
  hop_r = (uint8_t)((hop_r + 1) % kHopSlots);
  hop_n--;
  interrupts();
  return true;
}

// Flatten the ring into the CNN input in time order, oldest row first.
// oldest == spec_write after increment (the next overwrite is the oldest live row).
static void quantize_spectrogram(int oldest) {
  const float scale = input_tensor->params.scale;
  const int zp = input_tensor->params.zero_point;
  int8_t* dst = input_tensor->data.int8;
  int n = 0;
  for (int t = 0; t < kNFrames; t++) {
    const int src = (oldest + t) % kNFrames;
    for (int m = 0; m < kNMels; m++) {
      int q = (int)lroundf(spectrogram[src][m] / scale) + zp;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      dst[n++] = (int8_t)q;
    }
  }
}

// Run the CNN and return P(flush) via softmax on the two dequantized logits.
static float flush_probability() {
  interpreter->Invoke();
  const float scale = output_tensor->params.scale;
  const int zp = output_tensor->params.zero_point;
  const int8_t* o = output_tensor->data.int8;
  const float z0 = (o[0] - zp) * scale;  // not-flush logit
  const float z1 = (o[1] - zp) * scale;  // flush logit
  const float m = z0 > z1 ? z0 : z1;     // subtract max so exp() does not overflow
  const float e0 = expf(z0 - m);
  const float e1 = expf(z1 - m);
  return e1 / (e0 + e1);
}

static bool setup_model() {
  const tflite::Model* model = tflite::GetModel(g_flush_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
#if FLUSH_SERIAL
    Serial.println("Model schema mismatch");
#endif
    return false;
  }
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
#if FLUSH_SERIAL
    Serial.println("AllocateTensors failed — raise kTensorArenaSize");
#endif
    return false;
  }
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
#if FLUSH_SERIAL
  Serial.print("input bytes=");
  Serial.print(input_tensor->bytes);  // expect 8000 = 250 * 32
  Serial.print("  arena used=");
  Serial.println(interpreter->arena_used_bytes());
#endif
  return true;
}

void setup() {
#if FLUSH_SERIAL
  Serial.begin(115200);
  delay(1500);  // USB CDC needs a moment before the first print
#endif
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  led_red();

  if (!setup_model()) {
    while (true) {  // fatal: blink red
      led_red();
      delay(200);
      leds_off();
      delay(200);
    }
  }

  PDM.onReceive(onPDMdata);
  PDM.setGain(127);  // max; must match the USB recorder / training clips
  if (!PDM.begin(1, 16000)) {  // mono 16 kHz — same as Colab MFE
#if FLUSH_SERIAL
    Serial.println("PDM.begin failed");
#endif
    while (true) {
      led_red();
      delay(200);
      leds_off();
      delay(200);
    }
  }
#if FLUSH_SERIAL
  Serial.println("ring 8s, hop FIFO 12, infer every 352 ms  (green if p>=0.5)");
#endif
}

void loop() {
  int16_t hop[kHopSamples];
  if (!pop_hop(hop)) {
    return;  // nothing queued; ISR will fill the FIFO
  }

  // Slide the 64 ms window: previous hop -> front, new hop -> back (50% overlap).
#if FLUSH_SERIAL
  const uint32_t hop_t0 = micros();
#endif
  memmove(fft_window, fft_window + kHopSamples, kHopSamples * sizeof(int16_t));
  memcpy(fft_window + kHopSamples, hop, sizeof(hop));
  hop_count++;
  if (hop_count < 2) {
    return;  // first hop only: window is half uninitialized
  }

  // One log-mel row from 1024 PCM samples. Healthy: ~6 ms vs 32 ms hop period.
  mfe_frame(fft_window, spectrogram[spec_write]);
#if FLUSH_SERIAL
  const uint32_t hop_us = micros() - hop_t0;
  hop_us_sum += hop_us;
  if (hop_us > hop_us_max) {
    hop_us_max = hop_us;
  }
#endif

  spec_write = (spec_write + 1) % kNFrames;
  if (spec_write == 0) {
    ring_full = true;  // wrapped once: 250 rows exist, CNN may run
  }
  hops_since_infer++;

  // Warm-up: wait for 8 s of rows. After that: CNN every 11 new rows (~352 ms), not every 8 s.
  if (!ring_full || hops_since_infer < kInferEveryHops) {
    return;
  }

#if FLUSH_SERIAL
  const int n_hops = hops_since_infer;
#endif
  hops_since_infer = 0;

  quantize_spectrogram(spec_write);
#if FLUSH_SERIAL
  const uint32_t infer_t0 = micros();
#endif
  const float p = flush_probability();
#if FLUSH_SERIAL
  const uint32_t infer_us = micros() - infer_t0;

  // Snapshot stats, then start a new interval. hop_n_max resets to current depth, not 0.
  const uint32_t dropped = hops_dropped;
  const uint8_t peak = hop_n_max;
  hops_dropped = 0;
  hop_n_max = hop_n;
  const float hop_avg_ms = (hop_us_sum / (float)n_hops) / 1000.0f;
  Serial.print("flush p=");
  Serial.print(p, 3);
  Serial.print("  infer_ms=");
  Serial.print(infer_us / 1000.0f, 1);
  Serial.print("  hop_avg_ms=");
  Serial.print(hop_avg_ms, 2);
  Serial.print("  hop_max_ms=");
  Serial.print(hop_us_max / 1000.0f, 2);
  Serial.print("  fifo_peak=");
  Serial.print(peak);
  Serial.print("/");
  Serial.print(kHopSlots);
  Serial.print("  dropped=");
  Serial.println(dropped);

  hop_us_sum = 0;
  hop_us_max = 0;
#endif

  // Hysteresis: 0.35..0.50 keeps the previous color.
  if (p >= 0.5f) {
    led_green();
  } else if (p < 0.35f) {
    led_red();
  }
}
