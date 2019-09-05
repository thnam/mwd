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


int main(int argc, char *argv[]) {
  std::vector<double> wf0;
  unsigned int nSamples = 0;
	auto readStart = std::chrono::high_resolution_clock::now();
  std::ifstream inputFile("samples/purdue_full_wf0.csv");
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
    nSamples ++;
  }
  std::cout << "Read in " << nSamples << " samples" << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::cout <<std::fixed << std::setprecision(10) <<wf0[i] << std::endl;
  }

	auto readStop = std::chrono::high_resolution_clock::now();
	auto readDuration = 
    std::chrono::duration_cast<std::chrono::microseconds> (readStop - readStart);

  std::cout << "Done csv reading: " << readDuration.count() <<" us"<< std::endl;
  return 0;
}
