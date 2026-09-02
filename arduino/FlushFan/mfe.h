#pragma once

#include <stdint.h>

constexpr int kSampleRate = 16000;
constexpr int kFrameSamples = 1024;  // FFT window (64 ms)
constexpr int kHopSamples = 512;      // 32 ms, 50% overlap
constexpr int kNMels = 32;
constexpr int kNFrames = 250;         // 8 s / 32 ms
constexpr float kLogEps = 1e-6f;

// Fills mels[32] from 1024 int16 PDM samples (same scaling as Python float32 WAV).
void mfe_frame(const int16_t* pcm, float* mels);
