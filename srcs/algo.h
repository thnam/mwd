#ifndef ALGO_HH_8GE72YHP
#define ALGO_HH_8GE72YHP

#include <stdint.h>
typedef struct Waveform {
  uint32_t size;
  double samples[];
} Waveform;

Waveform * Deconvolute(Waveform * wf, double f);
Waveform * OffsetDifferentiate(Waveform * wf, uint32_t M);
Waveform * MovingAverage(Waveform * wf, uint32_t L);
Waveform * MWD(Waveform * wf,  double f, uint32_t M, uint32_t L);

#endif /* end of include guard: ALGO_HH_8GE72YHP */
