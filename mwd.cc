#include <stdio.h>
#include <chrono> 
#include <ctime>
#include <string>
#include <vector>
#include <sstream> //istringstream
#include <iostream> // cout
#include <fstream> // ifstream
#include <cstdlib>
#include <iomanip>
#include <stdint.h>

std::vector<double> ReadWF(std::string filename);
std::vector<double> Deconvolute(std::vector<double> wf, double f);
std::vector<double> OffsetDifferentiate(std::vector<double> wf, uint32_t M);
std::vector<double> MovingAverage(std::vector<double> wf, uint32_t L);
std::vector<double> MWD(std::vector<double> wf, double f, uint32_t M, uint32_t L){
  return MovingAverage(OffsetDifferentiate(Deconvolute(wf, f), M), L);
}

int main(int argc, char *argv[]) {
  std::vector<double> wf0 = ReadWF("samples/purdue_full_wf0.csv");
  std::vector<double> mwd = MWD(wf0, 0.999993, 6000, 600);
  for (uint32_t i = 0; i < mwd.size(); ++i) {
    // std::cout << mwd.at(i) << ",";
    std::cout << mwd.at(i) << std::endl;
  }

  std::cout << std::endl;
  return 0;
}

std::vector<double> ReadWF(std::string filename){
  std::vector<double> wf0;
	auto readStart = std::chrono::high_resolution_clock::now();
  std::ifstream inputFile(filename);
  std::string line;
  while (std::getline(inputFile, line)){
    std::stringstream ss(line);
    std::string sval;
    bool save = false;
    while (std::getline(ss, sval, ',')){
      if (save) {
        wf0.push_back(atof(sval.c_str()));
      }
      save = !save;
    }
  }
	auto readStop = std::chrono::high_resolution_clock::now();
	auto readDuration = 
    std::chrono::duration_cast<std::chrono::milliseconds> (readStop - readStart);

  std::cout << "Read in " << wf0.size() << " samples" << std::endl;
  std::cout << " in " << readDuration.count() <<" ms"<< std::endl;
  return wf0;
}


std::vector<double> Deconvolute(std::vector<double> wf, double f){
  if (wf.size() <= 2) {
    return wf;
  }
  std::vector<double> A;
  A.push_back(wf.at(0));
  for (uint32_t i = 1; i < wf.size(); ++i) {
    A.push_back(wf.at(i) - f*wf.at(i-1) + A.at(i-1));
  }

  return A;
}

std::vector<double> OffsetDifferentiate(std::vector<double> wf, uint32_t M){
  if (wf.size() <= M) {
    return wf;
  }
  std::vector<double> D;
  for (uint32_t i = M; i < wf.size(); ++i) {
    D.push_back(wf.at(i) - wf.at(i-M));
  }
  return D;
}

std::vector<double> MovingAverage(std::vector<double> wf, uint32_t L){
  if (wf.size() <= L) {
    return wf;
  }

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
  return MA;
}
