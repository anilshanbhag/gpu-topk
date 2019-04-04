/* Copyright 2011 Russel Steinbach, Jeffrey Blanchard, Bradley Gordon,
 *   and Toluwaloju Alabi
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <iostream>
#include <cstdio>
#include <cuda.h>
using namespace std;

namespace PrintFunctions{
  template<typename T>
  void printArray(T *h_vec,uint size){
    printf("\n");
    for(int i = 0; i < size; i++){
      std::cout <<h_vec[i] << "\n";
    }
    printf("\n");
  }
  
  void printArray(char **h_vec,uint size){
    printf("\n");
    for(int i = 0; i < size; i++){
      printf("%d-%s\n",i, h_vec[i]);
    }
    printf("\n");
  }

  template<typename T>
  void printCudaArray(T *d_vec,uint size){
    T* h_vec = (T *) std::malloc(size * sizeof(T));
    cudaMemcpy(h_vec, d_vec, size*sizeof(T), cudaMemcpyDeviceToHost);
    printArray(h_vec, size);
    free(h_vec);
  }
}//end namespace 

