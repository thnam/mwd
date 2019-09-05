#ifndef ALGO_HH_8GE72YHP
#define ALGO_HH_8GE72YHP

#include <vector>
#include <stdint.h>

extern std::vector<double> Deconvolute(std::vector<double> wf, double f,
    double * time=NULL);
extern std::vector<double> OffsetDifferentiate(std::vector<double> wf,
    uint32_t M, double * time=NULL);
extern std::vector<double> MovingAverage(std::vector<double> wf, uint32_t L,
    double * time=NULL);
extern std::vector<double> MWD(std::vector<double> wf, double f, uint32_t M,
    uint32_t L, double * tDeconv=NULL, double * tDiff=NULL,
    double * tMavg=NULL);

#endif /* end of include guard: ALGO_HH_8GE72YHP */
