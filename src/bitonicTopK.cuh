#pragma once

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <algorithm>

#include "sharedmem.cuh"

using namespace std;
using namespace cub;

#define NUM_ELEM_PT 16
#define NUM_ELEM_BITSHIFT 4

//#define K 32
//#define KLog2 5

#define ORDERV(x,a,b) { bool swap = reverse ^ (x[a]<x[b]); \
      T auxa = x[a]; \
      if (swap) { x[a] = x[b]; x[b] = auxa; } }
      //T auxa = x[a]; T auxb = x[b]; \
      //x[a] = (swap)?auxb:auxa; x[b] = (swap)?auxa:auxb;}

#define B2V(x,a) { ORDERV(x,a,a+1) }
#define B4V(x,a) { for (int i4=0;i4<2;i4++) { ORDERV(x,a+i4,a+i4+2) } B2V(x,a) B2V(x,a+2) }
#define B8V(x,a) { for (int i8=0;i8<4;i8++) { ORDERV(x,a+i8,a+i8+4) } B4V(x,a) B4V(x,a+4) }
#define B16V(x,a) { for (int i16=0;i16<8;i16++) { ORDERV(x,a+i16,a+i16+8) } B8V(x,a) B8V(x,a+8) }
#define B32V(x,a) { for (int i32=0;i32<16;i32++) { ORDERV(x,a+i32,a+i32+16) } B16V(x,a) B16V(x,a+16) }
#define B64V(x,a) { for (int i64=0;i64<32;i64++) { ORDERV(x,a+i64,a+i64+32) } B32V(x,a) B32V(x,a+32) }

template<typename T>
__forceinline__
__device__  T get(T* sdata, int i) {
  return sdata[i + (i>>5)];
}

#define set(a,b,c) { int tempIndex = b; a[tempIndex + (tempIndex >> 5)] = c;  }

#define NUM_GROUPS (NUM_ELEM_PT/2)
#define NUM_GROUPS_BITSHIFT (NUM_ELEM_BITSHIFT-1)

#define RUN_64(X) { \
  inc >>= 5; \
  low = t & (inc - 1); \
  tCur = ((t - low) << 6) + low; \
  reverse = ((dir & tCur) == 0); \
  for (int j=0; j<NUM_GROUPS/(32 * X); j++) { \
    for (int i=0; i<64; i++) x[i] = get(sdata, tCur+i*inc); \
    B64V(x,0); \
    for (int i=0; i<64; i++) set(sdata, tCur+i*inc, x[i]); \
  } \
  inc >>= 1; \
}

#define RUN_32(X) { \
  inc >>= 4; \
  low = t & (inc - 1); \
  tCur = ((t - low) << 5) + low; \
  reverse = ((dir & tCur) == 0); \
  for (int j=0; j<NUM_GROUPS/(16 * X); j++) { \
    for (int i=0; i<32; i++) x[i] = get(sdata, tCur+i*inc); \
    B32V(x,0); \
    for (int i=0; i<32; i++) set(sdata, tCur+i*inc, x[i]); \
  } \
  inc >>= 1; \
}

#define RUN_16(X) { \
  inc >>= 3; \
  low = t & (inc - 1); \
  tCur = ((t - low) << 4) + low; \
  reverse = ((dir & tCur) == 0); \
  for (int j=0; j<NUM_GROUPS/(8 * X); j++) { \
    for (int i=0; i<16; i++) x[i] = get(sdata, tCur+i*inc); \
    B16V(x,0); \
    for (int i=0; i<16; i++) set(sdata, tCur+i*inc, x[i]); \
  } \
  inc >>= 1; \
}

#define RUN_8(X) { \
  inc >>= 2; \
  low = t & (inc - 1); \
  tCur = ((t - low) << 3) + low; \
  reverse = ((dir & tCur) == 0); \
  for (int j=0; j<NUM_GROUPS/(4 * X); j++) { \
    for (int i=0; i<8; i++) x[i] = get(sdata, tCur+i*inc); \
    B8V(x,0); \
    for (int i=0; i<8; i++) set(sdata, tCur+i*inc, x[i]); \
  } \
  inc >>= 1; \
}

#define RUN_4(X) { \
  inc >>= 1; \
  low = t & (inc - 1); \
  tCur = ((t - low) << 2) + low; \
  reverse = ((dir & tCur) == 0); \
  for (int j=0; j<NUM_GROUPS/(2 * X); j++) { \
    for (int i=0;i<4;i++) x[i] = get(sdata, 4*wg*j + tCur + i*inc); \
    B4V(x,0); \
    for (int i=0;i<4;i++) set(sdata, 4*wg*j + tCur + i*inc, x[i]); \
  } \
  inc >>= 1; \
}

#define RUN_2(X) { \
  low = t & (inc - 1); \
  tCur = ((t - low) << 1) + low; \
  reverse = ((dir & tCur) == 0); \
  for (int j=0; j<NUM_GROUPS/(X); j++) { \
    for (int i=0;i<2;i++) x[i] = get(sdata, 2*wg*j + tCur + i*inc); \
    B2V(x,0); \
    for (int i=0;i<2;i++) set(sdata, 2*wg*j + tCur + i*inc, x[i]); \
  } \
  inc >>= 1; \
}

#define REDUCE(X) { \
  tCur = ((t >> klog2) << (klog2 + 1)) + (t & (k-1)); \
  for(int j=0; j<NUM_GROUPS/(X); j++) { \
    x[j] = max(get(sdata, 2*wg*j + tCur), get(sdata, 2*wg*j + tCur + k)); \
  } \
  __syncthreads(); \
  for(int j=0; j<NUM_GROUPS/(X); j++) { \
    set(sdata, wg*j + t, x[j]); \
  } \
}

template<typename T>
__global__ void Bitonic_TopKLocalSortInPlace(T* __restrict__ in, T* __restrict__ out,
  const int k, const int klog2)
{
/*  const int k = K;*/
  /*const int klog2 = KLog2;*/

  // Shared mem size is determined by the host app at run time.
  // For n elements, we have n * 33/32 shared memory.
  // We use this to break bank conflicts.
  SharedMemory<T> smem;
  T* sdata = smem.getPointer();

  const int t = threadIdx.x; // index in workgroup
  const int wg = blockDim.x; // workgroup size = block size, power of 2
  const int gid = blockIdx.x;

  int length = min(NUM_GROUPS, k >> 1);
  int inc = length;
  inc >>= NUM_GROUPS_BITSHIFT;
  int low = t & (inc - 1);
  int dir = length << 1;
  bool reverse;

  T x[NUM_ELEM_PT];

  // Move IN, OUT to block start
  in += NUM_ELEM_PT * gid * wg;

  int tCur = t << NUM_ELEM_BITSHIFT;
  for (int i=0; i<NUM_ELEM_PT; i++) x[i] = in[tCur + i];

  for (int i=0; i<NUM_ELEM_PT; i+=2) {
    reverse = ((i >> 1) + 1)&1;
    B2V(x,i);
  }
  if (k > 2) {
#if NUM_ELEM_PT > 4
    for (int i=0; i<NUM_ELEM_PT; i+=4) {
      reverse = ((i >> 2) + 1)&1;
      B4V(x,i);
    }
    if (k > 4) {
#if NUM_ELEM_PT > 8
      for (int i=0; i<NUM_ELEM_PT; i+=8) {
        reverse = ((i >> 3) + 1)&1;
        B8V(x,i);
      }
      if (k > 8) {
#if NUM_ELEM_PT > 16
        for (int i=0; i<NUM_ELEM_PT; i+=16) {
          reverse = ((i >> 4) + 1)&1;
          B16V(x,i);
        }
        if (k > 16) {
#if NUM_ELEM_PT > 32
          for (int i=0; i<NUM_ELEM_PT; i+=32) {
            reverse = ((i >> 5) + 1)&1;
            B32V(x,i);
          }
          if (k > 32) {
            reverse = ((dir & tCur) == 0); B64V(x,0);
          }
#else
          reverse = ((dir & tCur) == 0); B32V(x,0);
#endif
        }
#else
        reverse = ((dir & tCur) == 0); B16V(x,0);
#endif
      }
#else
      reverse = ((dir & tCur) == 0); B8V(x,0);
#endif
    }
#else
    reverse = ((dir & tCur) == 0); B4V(x,0);
#endif
  }

  for (int i=0; i<NUM_ELEM_PT; i++) set(sdata, tCur+i, x[i]);

  __syncthreads();

  // Complete the remaining steps to create sorted sequences of length k.
  int mod;
  unsigned int mask;

  for (length=NUM_ELEM_PT; length<k; length<<=1)
  {
    dir = length << 1;
    // Loop on comparison distance (between keys)
    inc = length;
    mod = inc;
    mask = ~(NUM_ELEM_PT/(1) - 1);
    while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 0);

    if (mod & 1)
    {
      RUN_2(1)
      __syncthreads();
    }
    if (mod & 2)
    {
      RUN_4(1)
      __syncthreads();
    }
#if NUM_ELEM_PT > 8
    if (mod & 4)
    {
      RUN_8(1)
      __syncthreads();
    }
#if NUM_ELEM_PT > 16
    if (mod & 8)
    {
      RUN_16(1)
      __syncthreads();
    }
    while (inc > 8)
    {
      RUN_32(1)
      __syncthreads();
    }
#else
    while (inc > 4)
    {
      RUN_16(1)
      __syncthreads();
    }
#endif // NUM_ELEM_PT > 16
#else
    while (inc > 2)
    {
      RUN_8(1)
      __syncthreads();
    }
#endif // NUM_ELEM_PT > 8
  }

  // Step 2: Reduce the size by factor 2 by pairwise comparing adjacent sequences.
  REDUCE(1)
  __syncthreads();
  // End of Step 2;

  // Step 3: Construct sorted sequence of length k from bitonic sequence of length k.
  // We now have n/2 elements.
  length = k >> 1;
  dir = length << 1;
  // Loop on comparison distance (between keys)
  inc = length;
  mod = inc;
  mask = ~(NUM_ELEM_PT/(1) - 1);
  while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 0);

  if (mod & 1)
  {
    RUN_2(2)
    __syncthreads();
  }
#if NUM_ELEM_PT > 4
  if (mod & 2)
  {
    RUN_4(2)
    __syncthreads();
  }
#if NUM_ELEM_PT > 8
  if (mod & 4)
  {
    RUN_8(2)
    __syncthreads();
  }
  while (inc > 4)
  {
    if (t < (wg >> 1)) {
      RUN_16(1)
    } else {
      inc >>= 4;
    }
    __syncthreads();
  }
#else
  while (inc > 2)
  {
    RUN_8(2)
    __syncthreads();
  }
#endif // NUM_ELEM_PT > 16
#else
  while (inc > 1)
  {
    RUN_4(2)
    __syncthreads();
  }
#endif // NUM_ELEM_PT > 8

  // Step 4: Reduce size again by 2.
  REDUCE(2)
  __syncthreads();
  // End of Step 1;

  length = k >> 1;
  dir = length << 1;
  // Loop on comparison distance (between keys)
  inc = length;
  mod = inc;
  mask = ~(NUM_ELEM_PT/(2) - 1);
  while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 1);

#if NUM_ELEM_PT > 4
  if (mod & 1)
  {
    RUN_2(4)
    __syncthreads();
  }
#if NUM_ELEM_PT > 8
  if (mod & 2)
  {
    RUN_4(4)
    __syncthreads();
  }
  while (inc > 2)
  {
    if (t < (wg >> 1)) {
      RUN_8(2)
    } else {
      inc >>= 3;
    }
    __syncthreads();
  }
#else
  while (inc > 1)
  {
    RUN_4(4)
    __syncthreads();
  }
#endif // NUM_ELEM_PT > 16
#else
  while (inc > 0)
  {
    RUN_2(4)
    __syncthreads();
  }
#endif // NUM_ELEM_PT > 8 while (inc > 0)

  // Step 4: Reduce size again by 2.
  REDUCE(4)
  __syncthreads();
  // End of Step 1;

  length = k >> 1;
  dir = length << 1;
  // Loop on comparison distance (between keys)
  inc = length;
  mod = inc;
  mask = ~(NUM_ELEM_PT/(4) - 1);
  while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 2);

  if (mod & 1)
  {
    RUN_2(8)
    __syncthreads();
  }
  while (inc > 0)
  {
    if (t < (wg >> 1)) {
      RUN_4(4)
    } else {
      inc >>= 2;
    }
    __syncthreads();
  }

  out += (NUM_ELEM_PT/16) * gid * wg;
  tCur = ((t >> klog2) << (klog2+1)) + (t&(k-1));
  for (int j=0; j<NUM_GROUPS/8; j++) {
    T x0 = get(sdata, 2*wg*j + tCur);
    T x1 = get(sdata, 2*wg*j + tCur + k);
    out[wg*j + t] = max(x0, x1);
  }

/*  out += (NUM_ELEM_PT/8) * gid * wg;*/
  //tCur = ((t >> klog2) << (klog2+1)) + (t&(k-1));
  //for (int j=0; j<NUM_GROUPS/4; j++) {
    //T x0 = get(sdata, 2*wg*j + tCur);
    //T x1 = get(sdata, 2*wg*j + tCur + k);
    //out[wg*j + t] = max(x0, x1);
  /*}*/
}

template<typename T>
__global__ void Bitonic_TopKReduce(T* __restrict__ in, T* __restrict__ out,
  const int k, const int klog2)
{
/*  const int k = K;*/
  /*const int klog2 = KLog2;*/

  // Shared mem size is determined by the host app at run time.
  // For n elements, we have n * 33/32 shared memory.
  // We use this to break bank conflicts.
  SharedMemory<T> smem;
  T* sdata = smem.getPointer();

  const int t = threadIdx.x; // index in workgroup
  const int wg = blockDim.x; // workgroup size = block size, power of 2
  const int gid = blockIdx.x;

  int length = min(NUM_GROUPS, k >> 1);
  int inc = length;
  inc >>= NUM_GROUPS_BITSHIFT;
  int low = t & (inc - 1);
  int dir = length << 1;
  bool reverse;

  T x[NUM_ELEM_PT];

  // Move IN, OUT to block start
  in += NUM_ELEM_PT * gid * wg;

  int tCur = t << NUM_ELEM_BITSHIFT;
  for (int i=0; i<NUM_ELEM_PT; i++) x[i] = in[tCur + i];
  for (int i=0; i<NUM_ELEM_PT; i++) set(sdata, tCur+i, x[i]);

  __syncthreads();

  // Complete the remaining steps to create sorted sequences of length k.
  int mod;
  unsigned int mask;

    length = (k >> 1);
    dir = length << 1;
    // Loop on comparison distance (between keys)
    inc = length;
    mod = inc;
    mask = ~(NUM_ELEM_PT/(1) - 1);
    while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 0);

    if (mod & 1)
    {
      RUN_2(1)
      __syncthreads();
    }
    if (mod & 2)
    {
      RUN_4(1)
      __syncthreads();
    }
#if NUM_ELEM_PT > 8
    if (mod & 4)
    {
      RUN_8(1)
      __syncthreads();
    }
#if NUM_ELEM_PT > 16
    if (mod & 8)
    {
      RUN_16(1)
      __syncthreads();
    }
    while (inc > 8)
    {
      RUN_32(1)
      __syncthreads();
    }
#else
    while (inc > 4)
    {
      RUN_16(1)
      __syncthreads();
    }
#endif // NUM_ELEM_PT > 16
#else
    while (inc > 2)
    {
      RUN_8(1)
      __syncthreads();
    }
#endif // NUM_ELEM_PT > 8

  // Step 2: Reduce the size by factor 2 by pairwise comparing adjacent sequences.
  REDUCE(1)
  __syncthreads();
  // End of Step 2;

  // Step 3: Construct sorted sequence of length k from bitonic sequence of length k.
  // We now have n/2 elements.
  length = k >> 1;
  dir = length << 1;
  // Loop on comparison distance (between keys)
  inc = length;
  mod = inc;
  mask = ~(NUM_ELEM_PT/(1) - 1);
  while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 0);

  if (mod & 1)
  {
    RUN_2(2)
    __syncthreads();
  }
#if NUM_ELEM_PT > 4
  if (mod & 2)
  {
    RUN_4(2)
    __syncthreads();
  }
#if NUM_ELEM_PT > 8
  if (mod & 4)
  {
    RUN_8(2)
    __syncthreads();
  }
  while (inc > 4)
  {
    if (t < (wg >> 1)) {
      RUN_16(1)
    } else {
      inc >>= 4;
    }
    __syncthreads();
  }
#else
  while (inc > 2)
  {
    RUN_8(2)
    __syncthreads();
  }
#endif // NUM_ELEM_PT > 16
#else
  while (inc > 1)
  {
    RUN_4(2)
    __syncthreads();
  }
#endif // NUM_ELEM_PT > 8

  // Step 4: Reduce size again by 2.
  REDUCE(2)
  __syncthreads();
  // End of Step 1;

  length = k >> 1;
  dir = length << 1;
  // Loop on comparison distance (between keys)
  inc = length;
  mod = inc;
  mask = ~(NUM_ELEM_PT/(2) - 1);
  while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 1);

#if NUM_ELEM_PT > 4
  if (mod & 1)
  {
    RUN_2(4)
    __syncthreads();
  }
#if NUM_ELEM_PT > 8
  if (mod & 2)
  {
    RUN_4(4)
    __syncthreads();
  }
  while (inc > 2)
  {
    if (t < (wg >> 1)) {
      RUN_8(2)
    } else {
      inc >>= 3;
    }
    __syncthreads();
  }
#else
  while (inc > 1)
  {
    RUN_4(4)
    __syncthreads();
  }
#endif // NUM_ELEM_PT > 16
#else
  while (inc > 0)
  {
    RUN_2(4)
    __syncthreads();
  }
#endif // NUM_ELEM_PT > 8 while (inc > 0)

  // Step 4: Reduce size again by 2.
  REDUCE(4)
  __syncthreads();
  // End of Step 1;

  length = k >> 1;
  dir = length << 1;
  // Loop on comparison distance (between keys)
  inc = length;
  mod = inc;
  mask = ~(NUM_ELEM_PT/(4) - 1);
  while ((mod & mask) != 0) mod >>= (NUM_ELEM_BITSHIFT - 2);

  if (mod & 1)
  {
    RUN_2(8)
    __syncthreads();
  }
  while (inc > 0)
  {
    if (t < (wg >> 1)) {
      RUN_4(4)
    } else {
      inc >>= 2;
    }
    __syncthreads();
  }

  out += (NUM_ELEM_PT/16) * gid * wg;
  tCur = ((t >> klog2) << (klog2+1)) + (t&(k-1));
  for (int j=0; j<NUM_GROUPS/8; j++) {
    T x0 = get(sdata, 2*wg*j + tCur);
    T x1 = get(sdata, 2*wg*j + tCur + k);
    out[wg*j + t] = max(x0, x1);
  }
}

const int tab32[32] = {
     0,  9,  1, 10, 13, 21,  2, 29,
    11, 14, 16, 18, 22, 25,  3, 30,
     8, 12, 20, 28, 15, 17, 24,  7,
    19, 27, 23,  6, 26,  5,  4, 31};

int log2_32 (uint value)
{
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  return tab32[(uint)(value*0x07C4ACDD) >> 27];
}

template<typename KeyT>
cudaError_t bitonicTopK(KeyT *d_keys_in, unsigned int num_items, unsigned int k, KeyT *d_keys_out,
    CachingDeviceAllocator&  g_allocator) {
  if (k < 16) k = 16;

  int klog2 = log2_32(k);

  DoubleBuffer<KeyT> d_keys;
  d_keys.d_buffers[0] = d_keys_in;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(KeyT) * num_items));

  int current = 0;
  int numThreads = num_items;

  int wg_size = max(64,k);

  numThreads >>= 1; // Each thread processes 2 elements.
  numThreads >>= NUM_GROUPS_BITSHIFT;
  Bitonic_TopKLocalSortInPlace<KeyT><<<numThreads/wg_size, wg_size, ((2*NUM_GROUPS*wg_size*33)/32)*sizeof(KeyT)>>>(d_keys.Current(), d_keys.Alternate(), k, klog2);
  current = 1-current;

  // Toggle the buffer index in the double buffer
  d_keys.selector = d_keys.selector ^ 1;

  numThreads >>= (1 + NUM_GROUPS_BITSHIFT);

  while (numThreads >= wg_size)
  {
    Bitonic_TopKReduce<KeyT><<<numThreads/wg_size, wg_size, ((2*NUM_GROUPS*wg_size*33)/32)*sizeof(KeyT)>>>(d_keys.Current(), d_keys.Alternate(), k, klog2);

    // Toggle the buffer index in the double buffer
    d_keys.selector = d_keys.selector ^ 1;

    numThreads >>= (1 + NUM_GROUPS_BITSHIFT);
  }

  KeyT* res_vec = (KeyT*) malloc(sizeof(KeyT) * 2 * numThreads * NUM_GROUPS);
  cudaMemcpy(res_vec, d_keys.Current(), 2 * numThreads * NUM_GROUPS * sizeof(KeyT), cudaMemcpyDeviceToHost);
  std::sort(res_vec, res_vec + 2*numThreads*NUM_GROUPS, std::greater<KeyT>());
  cudaMemcpy(d_keys_out, res_vec, k * sizeof(KeyT), cudaMemcpyHostToDevice);

  if (d_keys.d_buffers[1])
    CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));

  return cudaSuccess;
}


