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

#include "algo.h"

std::vector<double> ReadWF(std::string filename, double * time=NULL);

int main(int argc, char *argv[]) {
    
  double readTime;
  std::vector<double> wf0 = ReadWF("samples/purdue_full_wf0.csv", &readTime);
  std::cout << "read: " << readTime << std::endl;

  Waveform * wf = (Waveform *) malloc(sizeof(uint32_t) + wf0.size() * sizeof(double));
  wf->size = wf0.size();
  for (uint32_t i = 0; i < wf->size; ++i) {
    wf->samples[i] = wf0.at(i);
  }

  Waveform * mwd = MWD(wf, 0.999993, 6000, 600);
  for (int i = 0; i < mwd->size; ++i) {
    std::cout << mwd->samples[i] << std::endl;
  }
  // std::cout << tDeconv << "," << tDiff << "," << tMavg<< std::endl;

  return 0;
}

std::vector<double> ReadWF(std::string filename, double * time){
  std::vector<double> wf0;
	auto start = std::chrono::high_resolution_clock::now();
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
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = 
    std::chrono::duration_cast<std::chrono::milliseconds> (stop - start);
  *time = duration.count();
  return wf0;
}

