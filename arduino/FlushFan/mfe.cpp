#include "mfe.h"
#include "mfe_tables.h"

#include <math.h>
#include <string.h>

static void bit_reverse(float* re, float* im, int n) {
  int j = 0;
  for (int i = 1; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      float tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      float ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
}

// In-place radix-2 FFT on n=1024 real+imag arrays.
static void fft_radix2(float* re, float* im, int n) {
  bit_reverse(re, im, n);
  for (int len = 2; len <= n; len <<= 1) {
    const float ang = -2.0f * 3.14159265f / len;
    const float wlen_re = cosf(ang);
    const float wlen_im = sinf(ang);
    for (int i = 0; i < n; i += len) {
      float w_re = 1.0f;
      float w_im = 0.0f;
      const int half = len >> 1;
      for (int j = 0; j < half; j++) {
        const int u = i + j;
        const int v = u + half;
        const float vr = re[v] * w_re - im[v] * w_im;
        const float vi = re[v] * w_im + im[v] * w_re;
        re[v] = re[u] - vr;
        im[v] = im[u] - vi;
        re[u] += vr;
        im[u] += vi;
        const float nre = w_re * wlen_re - w_im * wlen_im;
        w_im = w_re * wlen_im + w_im * wlen_re;
        w_re = nre;
      }
    }
  }
}

void mfe_frame(const int16_t* pcm, float* mels) {
  static float re[kNFft];
  static float im[kNFft];
  static float power[kNFftBins];
  for (int i = 0; i < kNFft; i++) {
    re[i] = (pcm[i] / 32768.0f) * kHannWindow[i];
    im[i] = 0.0f;
  }
  fft_radix2(re, im, kNFft);

  power[0] = re[0] * re[0];
  power[kNFft / 2] = re[kNFft / 2] * re[kNFft / 2];
  for (int k = 1; k < kNFft / 2; k++) {
    power[k] = re[k] * re[k] + im[k] * im[k];
  }

  for (int m = 0; m < kNMels; m++) {
    float acc = 0.0f;
    const int start = kMelStart[m];
    const int n = kMelLength[m];
    const int off = kMelWeightOffset[m];
    for (int i = 0; i < n; i++) {
      acc += power[start + i] * kMelWeights[off + i];
    }
    mels[m] = logf(acc + kLogEps);
  }
}
