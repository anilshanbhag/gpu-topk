#include <cuda.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>

using namespace std;
using namespace cub;

template<typename KeyT>
cudaError_t sortTopK(KeyT *d_keys_in, unsigned int num_items, unsigned int k, KeyT *d_keys_out,
    CachingDeviceAllocator&  g_allocator) {
  // Allocate device memory for input/output
  DoubleBuffer<KeyT> d_keys;
  d_keys.d_buffers[0] = d_keys_in;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(KeyT) * num_items));

  // Allocate temporary storage
  size_t  temp_storage_bytes  = 0;
  void    *d_temp_storage     = NULL;
  CubDebugExit(DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, d_keys, num_items));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Sort
  CubDebugExit(DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, d_keys, num_items));

  // Copy results for verification. GPU-side part is done.
  CubDebugExit(cudaMemcpy(d_keys_out, d_keys.Current(), sizeof(KeyT) * k, cudaMemcpyDeviceToDevice));

  // Cleanup
  if (d_keys.d_buffers[1])
    CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));
  if (d_temp_storage)
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

  return cudaSuccess;
}

