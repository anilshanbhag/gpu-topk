#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#include <cub/util_allocator.cuh>

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <algorithm>

#include "printFunctions.cuh"
#include "generateProblems.cuh"
#include "sortTopK.cuh"
#include "radixSelectTopK.cuh"
#include "bitonicTopK.cuh"

#define SETUP_TIMING() cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

#define TIME_FUNC(f,t) { \
    cudaEventRecord(start, 0); \
    f; \
    cudaEventRecord(stop, 0); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&t, start,stop); \
}

using namespace std;

CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

#define NUMBEROFALGORITHMS 3
const char* namesOfTimingFunctions[NUMBEROFALGORITHMS] = {"Sort", "Radix Select", "Bitonic TopK"};

template<typename KeyT>
void compareAlgorithms(uint size, uint k, uint numTests,uint *algorithmsToTest, uint generateType){
  KeyT *d_vec;
  KeyT *d_vec_copy;
  KeyT *d_res;
  float timeArray[NUMBEROFALGORITHMS][numTests];
  float totalTimesPerAlgorithm[NUMBEROFALGORITHMS];
  float minTimesPerAlgorithm[NUMBEROFALGORITHMS];
  KeyT* resultsArray[NUMBEROFALGORITHMS][numTests];
  std::fill_n(minTimesPerAlgorithm, NUMBEROFALGORITHMS, 2000);

  uint winnerArray[numTests];
  uint timesWon[NUMBEROFALGORITHMS];
  uint i,j,m,x;
  int runOrder[NUMBEROFALGORITHMS];

  unsigned long long seed;
  timeval t1;
  float runtime;

  SETUP_TIMING()

  typedef cudaError_t (*ptrToTimingFunction)(KeyT*, uint, uint, KeyT*, CachingDeviceAllocator&);
  typedef void (*ptrToGeneratingFunction)(KeyT*, uint, curandGenerator_t, CachingDeviceAllocator&);
  //these are the functions that can be called
  ptrToTimingFunction arrayOfTimingFunctions[NUMBEROFALGORITHMS] = {&sortTopK<KeyT> , &radixSelectTopK<KeyT>, &bitonicTopK<KeyT>};
  /*ptrToTimingFunction arrayOfTimingFunctions[NUMBEROFALGORITHMS] = {NULL, NULL, NULL};*/

  ptrToGeneratingFunction *arrayOfGenerators;
  const char** namesOfGeneratingFunctions;
  //this is the array of names of functions that generate problems of this type, ie float, double, or uint
  namesOfGeneratingFunctions = returnNamesOfGenerators<KeyT>();
  arrayOfGenerators = (ptrToGeneratingFunction *) returnGenFunctions<KeyT>();

  //zero out the totals and times won
  bzero(totalTimesPerAlgorithm, NUMBEROFALGORITHMS * sizeof(uint));
  bzero(timesWon, NUMBEROFALGORITHMS * sizeof(uint));
  //allocate space for d_vec, and d_vec_copy
  cudaMalloc(&d_vec, size * sizeof(KeyT));
  cudaMalloc(&d_vec_copy, size * sizeof(KeyT));
  cudaMalloc(&d_res, k * sizeof(KeyT));

  //create the random generator.
  curandGenerator_t generator;
  srand(unsigned(time(NULL)));

  printf("The distribution is: %s\n", namesOfGeneratingFunctions[generateType]);
  for(i = 0; i < numTests; i++){
    // cudaDeviceReset();
    gettimeofday(&t1, NULL);
    seed = t1.tv_usec * t1.tv_sec;

    for(m = 0; m < NUMBEROFALGORITHMS;m++){
      runOrder[m] = m;
    }
    std::random_shuffle(runOrder, runOrder + NUMBEROFALGORITHMS);
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator,seed);
    curandSetPseudoRandomGeneratorSeed(generator,0);
    printf("Running test %u of %u for size: %u and k: %u\n", i + 1, numTests,size,k);
    //generate the random vector using the specified distribution
    arrayOfGenerators[generateType](d_vec, size, generator, g_allocator);

    //copy the vector to d_vec_copy, which will be used to restore it later
    cudaMemcpy(d_vec_copy, d_vec, size * sizeof(KeyT), cudaMemcpyDeviceToDevice);

    uint* h_vec = new uint[size];
    cudaMemcpy(h_vec, d_vec, size * sizeof(KeyT), cudaMemcpyDeviceToHost);

/*    uint *b = new uint[size];*/
    /*uint *c = new uint[size];*/
    /*uint *d = new uint[size];*/

    /*int x=0,y=0,z=0;*/
    /*for (int r=0; r<size; r++) {*/
      /*if ((h_vec[r] & 0xff000000) == 0xff000000) {*/
        /*b[x] = h_vec[r];*/
        /*x += 1;*/
      /*}*/
    /*}*/

/*    for (int i=0; i<x; i++) {*/
      /*if (b[i] & 0x0f00 == 0x0f00) {*/
        /*c[y] = b[i];*/
        /*y += 1;*/
      /*}*/
    /*}*/

    /*for (int i=0; i<x; i++) {*/
      /*if (b[i] & 0x0f00 == 0x0f00) {*/
        /*c[y] = b[i];*/
        /*y += 1;*/
      /*}*/
    /*}*/
/*    std::sort(b, b+x, std::greater<uint>());*/
    /*cout << "Count " << x << endl;*/
    /*for (int r=0; r<k; r++) cout << b[r] << endl;*/

/*    std::sort(h_vec, h_vec+size, std::greater<uint>());*/
    /*for (int r=0; r<k; r++) cout << h_vec[r] << endl;*/



    winnerArray[i] = 0;
    float currentWinningTime = INFINITY;
    //run the various timing functions
    for(x = 0; x < NUMBEROFALGORITHMS; x++){
      j = runOrder[x];
      if(algorithmsToTest[j]){

        //run timing function j
        printf("TESTING: %u %s\n", j, namesOfTimingFunctions[j]);
        TIME_FUNC(arrayOfTimingFunctions[j](d_vec_copy, size, k, d_res, g_allocator), runtime);

          // check for error
          cudaError_t error = cudaGetLastError();
            if(error != cudaSuccess)
            {
                  // print the CUDA error message and exit
                  printf("CUDA error: %s\n", cudaGetErrorString(error));
                      exit(-1);
            }

        //record the time result
        timeArray[j][i] = runtime;

        KeyT* h_res = new KeyT[k];
        cudaMemcpy(h_res, d_res, k * sizeof(KeyT), cudaMemcpyDeviceToHost);
        std::sort(h_res, h_res+k, std::greater<KeyT>());

        //record the value returned
        resultsArray[j][i] = h_res;
        //update the current "winner" if necessary
        if(timeArray[j][i] < currentWinningTime){
          currentWinningTime = runtime;
          winnerArray[i] = j;
        }

        //perform clean up
        cudaMemcpy(d_vec_copy, d_vec, size * sizeof(KeyT), cudaMemcpyDeviceToDevice);
      }
    }
    curandDestroyGenerator(generator);
/*    //for(x = 0; x < NUMBEROFALGORITHMS; x++){*/
      //if(algorithmsToTest[x]){
        //fileCsv << namesOfTimingFunctions[x] << ","<< resultsArray[x][i] <<","<< timeArray[x][i] <<",";
      //}
    /*}*/
 //   uint flag = 0;
/*    T tempResult = resultsArray[0][i];*/
    //for(m = 1; m < NUMBEROFALGORITHMS;m++){
      //if(algorithmsToTest[m]){
        //if(resultsArray[m][i] != tempResult){
          //flag++;
        //}
      //}
    /*}*/
    //fileCsv << flag << "\n";
  }

  //calculate the total time each algorithm took
  for(i = 0; i < numTests; i++){
    for(j = 0; j < NUMBEROFALGORITHMS;j++){
      if(algorithmsToTest[j]){
        totalTimesPerAlgorithm[j] += timeArray[j][i];
        minTimesPerAlgorithm[j] = min(minTimesPerAlgorithm[j], timeArray[j][i]);
      }
    }
  }

  //count the number of times each algorithm won.
  for(i = 0; i < numTests;i++){
    timesWon[winnerArray[i]]++;
  }

  printf("\n\n");

  //print out the average times
  for(i = 0; i < NUMBEROFALGORITHMS; i++){
    if(algorithmsToTest[i]){
      printf("%-20s averaged: %f ms\n", namesOfTimingFunctions[i], totalTimesPerAlgorithm[i] / numTests);
    }
  }
  for(i = 0; i < NUMBEROFALGORITHMS; i++){
    if(algorithmsToTest[i]){
      printf("%-20s minimum: %f ms\n", namesOfTimingFunctions[i], minTimesPerAlgorithm[i]);
    }
  }
  for(i = 0; i < NUMBEROFALGORITHMS; i++){
    if(algorithmsToTest[i]){
      printf("%s won %u times\n", namesOfTimingFunctions[i], timesWon[i]);
    }
  }
  if (algorithmsToTest[0]) {
    for(i = 0; i < numTests; i++){
      for(j =1 ; j< NUMBEROFALGORITHMS; j++){
        if(algorithmsToTest[j]){
          for (int m=0; m<k; m++)
            if(resultsArray[j][i][m] != resultsArray[0][i][m]){
              std::cout <<namesOfTimingFunctions[j] <<" did not return the correct answer on test" << i + 1 << std::endl;
              std::cout << "Method:\t";
              PrintFunctions::printArray<KeyT>(resultsArray[j][i], k);
              std::cout << "Sort:\t";
              PrintFunctions::printArray<KeyT>(resultsArray[0][i], k);
              break;
            }
        }
      }
    }
  }

  //free d_vec and d_vec_copy
  cudaFree(d_vec);
  cudaFree(d_vec_copy);
}

template<typename KeyT>
void runTests(uint generateType, int K, uint startPower, uint stopPower, uint timesToTestEachK = 3) {
  // Algorithms To Run
  // timeSort, timeRadixSelect, timeBitonicTopK
  uint algorithmsToRun[NUMBEROFALGORITHMS]= {1,1,1};
  uint size;
  for (size = (1 << startPower); size <= (1 <<stopPower);size *= 2){
    //  cudaDeviceReset();
    /*cudaThreadExit();*/
    printf("NOW STARTING A NEW K\n\n");
    compareAlgorithms<KeyT>(size, K, timesToTestEachK,algorithmsToRun,generateType);
  }
}

int main(int argc, char** argv)
{
  uint testCount;
  int K;
  uint type,distributionType,startPower,stopPower;
  if (argc == 7) {
    type = atoi(argv[1]);
    distributionType = atoi(argv[2]);
    K = atoi(argv[3]);
    testCount = atoi(argv[4]);
    startPower = atoi(argv[5]);
    stopPower = atoi(argv[6]);
  } else {
    printf("Please enter the type of value you want to test:\n1-float\n2-double\n3-uint\n");
    cin >> type;
    printf("Please enter distribution type: ");
    cin >> distributionType;
    printf("Please enter K: ");
    cin >> K;
    printf("Please enter number of tests to run per K: ");
    cin >> testCount;
    printf("Please enter start power (dataset size starts at 2^start)(max val: 29): ");
    cin >> startPower;
    printf("Please enter stop power (dataset size stops at 2^stop)(max val: 29): ");
    cin >> stopPower;
  }

  switch(type){
  case 1:
    runTests<float>(distributionType,K,startPower,stopPower,testCount);
    break;
  case 2:
    // runTests<double>(distributionType,K,startPower,stopPower,testCount);
    break;
  case 3:
     runTests<unsigned int>(distributionType,K,startPower,stopPower,testCount);
    break;
  default:
    printf("You entered and invalid option, now exiting\n");
    break;
  }

  return 0;
}

