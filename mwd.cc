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

#include "algo.hh"

std::vector<double> ReadWF(std::string filename, double * time=NULL);

int main(int argc, char *argv[]) {
    
  double readTime;
  std::vector<double> wf0 = ReadWF("samples/purdue_full_wf0.csv", &readTime);
  std::cout << "read: " << readTime << std::endl;
  double tDeconv, tDiff, tMavg;
  std::cout << "deconv,diff,mavg" << std::endl;
  std::vector<double> mwd = MWD(wf0, 0.999993, 6000, 600, &tDeconv, &tDiff, &tMavg);
  std::cout << tDeconv << "," << tDiff << "," << tMavg<< std::endl;

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

