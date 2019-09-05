#include <chrono> 
#include <ctime>
#include "algo.hh"

std::vector<double> Deconvolute(std::vector<double> wf, double f, double * time){
  if (wf.size() <= 2) {
    return wf;
  }
	auto start = std::chrono::high_resolution_clock::now();
  std::vector<double> A;
  A.push_back(wf.at(0));
  for (uint32_t i = 1; i < wf.size(); ++i) {
    A.push_back(wf.at(i) - f*wf.at(i-1) + A.at(i-1));
  }

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = 
    std::chrono::duration_cast<std::chrono::milliseconds> (stop - start);
  *time = duration.count();
  return A;
}

std::vector<double> OffsetDifferentiate(std::vector<double> wf, uint32_t M, double * time){
  if (wf.size() <= M) {
    return wf;
  }
	auto start = std::chrono::high_resolution_clock::now();
  std::vector<double> D;
  for (uint32_t i = M; i < wf.size(); ++i) {
    D.push_back(wf.at(i) - wf.at(i-M));
  }
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = 
    std::chrono::duration_cast<std::chrono::milliseconds> (stop - start);
  *time = duration.count();
  return D;
}

std::vector<double> MovingAverage(std::vector<double> wf, uint32_t L, double * time){
  if (wf.size() <= L) {
    return wf;
  }

	auto start = std::chrono::high_resolution_clock::now();
  double sum = 0.;
  std::vector<double> MA;
  for (uint32_t i = 0; i < L; ++i) {
    sum += wf.at(i);
  }
  MA.push_back(sum / L);

  for (uint32_t i = L; i < wf.size(); ++i) {
    sum += wf.at(i) - wf.at(i - L);
    MA.push_back(sum / L);
  }
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = 
    std::chrono::duration_cast<std::chrono::milliseconds> (stop - start);
  *time = duration.count();
  return MA;
}

std::vector<double> MWD(std::vector<double> wf, double f, uint32_t M, uint32_t L,
    double * tDeconv, double * tDiff, double * tMavg)
{
  return MovingAverage(OffsetDifferentiate(Deconvolute(wf, f, tMavg), M, tDiff), L, tDeconv);
}
