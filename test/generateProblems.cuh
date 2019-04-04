#pragma once

#include <cstdlib>
#include <typeinfo>
#include <cuda.h>
#include <curand.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>
/*#include <algorithm>*/
using namespace cub;
using namespace std;

template<typename KeyT>
void sort_keys(KeyT* d_vec, const unsigned int num_keys, CachingDeviceAllocator& g_allocator) {
 // Allocate device memory for input/output
  DoubleBuffer<KeyT> d_keys;
  d_keys.d_buffers[0] = d_vec;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(KeyT) * num_keys));

  // Allocate temporary storage
  size_t  temp_storage_bytes  = 0;
  void    *d_temp_storage     = NULL;
  CubDebugExit(DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_keys));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Sort
  CubDebugExit(DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_keys));

  // Copy results for verification. GPU-side part is done.
  if (d_keys.Current() != d_vec) {
    CubDebugExit(cudaMemcpy(d_vec, d_keys.Current(), sizeof(KeyT) * num_keys, cudaMemcpyDeviceToDevice));
  }

  if (d_temp_storage)
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

  if (d_keys.d_buffers[1])
    CubDebugExit( g_allocator.DeviceFree(d_keys.d_buffers[1]) );
}

///////////////////////////////////////////////////////////////////
////           FUNCTIONS TO GENERATE UINTS
///////////////////////////////////////////////////////////////////
typedef void (*ptrToUintGeneratingFunction)(uint*, uint, curandGenerator_t, CachingDeviceAllocator&);

void generateUniformUints(uint *d_vec, uint num_keys, curandGenerator_t generator, 
    CachingDeviceAllocator& g_allocator) {
  curandGenerate(generator, d_vec, num_keys);
}

void generateSortedIncreasingUints(uint* d_vec, uint num_keys, curandGenerator_t gen, 
    CachingDeviceAllocator& g_allocator) {
  curandGenerate(gen, d_vec,num_keys);
  sort_keys<uint>(d_vec, num_keys, g_allocator);
}

void generateSortedDecreasingUints(uint* d_vec, uint num_keys, curandGenerator_t gen, 
    CachingDeviceAllocator& g_allocator) {
  curandGenerate(gen, d_vec,num_keys);
  sort_keys<uint>(d_vec, num_keys, g_allocator);
}

#define NUMBEROFUINTDISTRIBUTIONS 3
ptrToUintGeneratingFunction arrayOfUintGenerators[NUMBEROFUINTDISTRIBUTIONS] = {&generateUniformUints, &generateSortedIncreasingUints, &generateSortedDecreasingUints};
const char* namesOfUintGeneratingFunctions[NUMBEROFUINTDISTRIBUTIONS]={"UNIFORM UINTS","SORTED INC UINTS", "SORTED DEC UINTS"};

///////////////////////////////////////////////////////////////////
////           FUNCTIONS TO GENERATE FLOATS
///////////////////////////////////////////////////////////////////
typedef void (*ptrToFloatGeneratingFunction)(float*, uint, curandGenerator_t, CachingDeviceAllocator&);

void generateUniformFloats(float *d_vec, uint num_keys, curandGenerator_t generator, 
    CachingDeviceAllocator& g_allocator){
  curandGenerateUniform(generator, d_vec,num_keys);
}

void generateSortedIncreasingFloats(float *d_vec, uint num_keys, curandGenerator_t generator, 
    CachingDeviceAllocator& g_allocator) {
  curandGenerateUniform(generator, d_vec,num_keys);
  sort_keys<float>(d_vec, num_keys, g_allocator);
}

void generateSortedDecreasingFloats(float *d_vec, uint num_keys, curandGenerator_t generator, 
    CachingDeviceAllocator& g_allocator) {
  curandGenerateUniform(generator, d_vec,num_keys);
  sort_keys<float>(d_vec, num_keys, g_allocator);
}

/*void generateBucketKillerFloats(float *d_vec, uint num_keys, curandGenerator_t generator){*/
  /*int i;*/
  /*float * d_generated = d_vec;*/
  /*curandGenerateUniform(generator, d_generated,num_keys);*/
  /*thrust::device_ptr<unsigned int> dev_ptr((uint *)d_generated);*/
  /*thrust::for_each( dev_ptr, dev_ptr + num_keys, makeSmallFloat());*/
  /*thrust::sort(dev_ptr,dev_ptr + num_keys);*/

  /*float* h_vec = (float*) malloc(num_keys * sizeof(float));*/
  /*cudaMemcpy(h_vec, d_generated, num_keys * sizeof(float), cudaMemcpyDeviceToHost);*/

  /*for(i = -126; i < 127; i++){*/
    /*h_vec[i + 126] = pow(2.0f,(float)i);*/
  /*}*/
  /*cudaMemcpy(d_generated, h_vec, num_keys * sizeof(float), cudaMemcpyHostToDevice);*/
  /*free(h_vec);*/
/*}*/

#define NUMBEROFFLOATDISTRIBUTIONS 3
ptrToFloatGeneratingFunction arrayOfFloatGenerators[NUMBEROFFLOATDISTRIBUTIONS] = {&generateUniformFloats, &generateSortedIncreasingFloats, &generateSortedDecreasingFloats};

const char* namesOfFloatGeneratingFunctions[NUMBEROFFLOATDISTRIBUTIONS] = {"UNIFORM FLOATS", "SORTED INC FLOATS", "SORTED DEC FLOATS"};

///////////////////////////////////////////////////////////////////
////           FUNCTIONS TO GENERATE DOUBLES
///////////////////////////////////////////////////////////////////

typedef void (*ptrToDoubleGeneratingFunction)(double*, uint, curandGenerator_t, CachingDeviceAllocator&);

void generateUniformDoubles(double *d_generated, uint num_keys, curandGenerator_t generator, 
    CachingDeviceAllocator& g_allocator) {
  curandGenerateUniformDouble(generator, d_generated,num_keys);
}

#define NUMBEROFDOUBLEDISTRIBUTIONS 1
ptrToDoubleGeneratingFunction arrayOfDoubleGenerators[NUMBEROFDOUBLEDISTRIBUTIONS] = {&generateUniformDoubles};
const char* namesOfDoubleGeneratingFunctions[NUMBEROFDOUBLEDISTRIBUTIONS]={"UNIFORM DOUBLES"};

template<typename T> void* returnGenFunctions(){
  if(typeid(T) == typeid(uint)){
    return arrayOfUintGenerators;
  }
  else if(typeid(T) == typeid(float)){
    return arrayOfFloatGenerators;
  }
  else{
    return arrayOfDoubleGenerators;
  }
}


template<typename T> const char** returnNamesOfGenerators(){
  if(typeid(T) == typeid(uint)){
    return &namesOfUintGeneratingFunctions[0];
  }
  else if(typeid(T) == typeid(float)){
    return &namesOfFloatGeneratingFunctions[0];
  }
  else {
    return &namesOfDoubleGeneratingFunctions[0];
  }
}

/*template void* returnGenFunctions<uint>();*/
/*template void* returnGenFunctions<float>();*/
/*template void* returnGenFunctions<double>();*/

/*template const char** returnNamesOfGenerators<uint>();*/
/*template const char** returnNamesOfGenerators<float>();*/
/*template const char** returnNamesOfGenerators<double>();*/

