#include <cuda.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>

using namespace cub;
using namespace std;

/**
 * Computes the histogram over the digit values of an array of keys that MUST have a length of an integer multiple of (KPT * blockDim.x).
 * The padding to the integer multiple can be done by adding 0's at the end and subtracting the number of padded 0's from the final result's 0 bin.
 * The 2^NUM_BITS possible counts (0..2^NUM_BITSNUM_BITS-1) will be placed in global_histo.
 * @param keys            [IN]  The keys for which to compute the histogram
 * @param digit           [IN]
 * @param global_histo        [OUT] The array of element counts, MUST be 256 in size.
 * @param per_block_histo     [OUT]
 */
template<
  typename KeyT,    // Data type of the keys within device memory. Data will be twiddled (if necessary) to unsigned type
  typename IndexT,  // Data type used for key's offsets and counters (limits number of supported keys, uint = 2^32)
  int NUM_BITS,     // Number of bits being sorted at a time
  int KPT,          // Number of keys per thread
  int TPB,          // Number of threads per block
  int PRE_SORT_RUNS_LENGTH // For values greater than 1, this causes to sort a thread's keys by runs of a given length to improve run-length encoded updates to shared memory.
>
__global__ void rdxsrt_histogram(KeyT *__restrict__ keys, const uint digit, IndexT *global_histo)
{
  /*** TYPEDEFs***/
  typedef Traits<KeyT>                        KeyTraits;
  typedef typename KeyTraits::UnsignedBits    UnsignedBits;
  /*typedef LoadUnit<IndexT, RDXSRT_LOAD_STRIDE_WARP, KPT, TPB> KeyLoader;*/

  /*** DECLARATIONS ***/
  UnsignedBits tloc_keys[KPT];
  uint tloc_masked[KPT];
  __shared__ uint shared_bins[0x01<<NUM_BITS];

  /*** INIT SHARED HISTO ***/
  if(threadIdx.x < 32){
    #pragma unroll
    for(int i=0;i<(0x01<<NUM_BITS);i+=32){
      shared_bins[i+threadIdx.x] = 0;
    }
  }
  __syncthreads();

  /*** GET KEYS & PREPARE KEYS FOR HISTO ***/
  // Bucket index used to determine the memory offset of the bucket's global histogram
  const uint bucket_idx = 0;
  // This thread block's keys memory offset, pointing to the index of its first key
  const IndexT block_offset = (blockDim.x * blockIdx.x * KPT);

  // Load keys
  // KeyLoader(block_offset, threadIdx.x).template LoadStrided<UnsignedBits, KeyT, 0, KPT>(keys, tloc_keys);
  #pragma unroll
  for (int i=0; i<KPT; i++) {
    tloc_keys[i] = reinterpret_cast<UnsignedBits*>(keys)[block_offset + threadIdx.x + blockDim.x * i];
  }

#if true || USE_RLE_HISTO
  // Mask
  #pragma unroll
  for (int i=0; i<KPT; i++) {
    tloc_keys[i] = KeyTraits::TwiddleIn(tloc_keys[i]);
    tloc_masked[i] = (tloc_keys[i]>>((sizeof(KeyT)*8)-(NUM_BITS*(digit+1))))&((0x01<<NUM_BITS)-1);
  }

#if 0
  /*** SORT RUNS ***/
  if(PRE_SORT_RUNS_LENGTH>1){
    SortingNetwork<uint>::sort_runs<PRE_SORT_RUNS_LENGTH>(tloc_masked);
  }
#endif

  /*** COMPUTE HISTO ***/
  uint rle = 1;
  #pragma unroll
  for(int i=1; i<KPT; i++){
    if(tloc_masked[i] == tloc_masked[i-1])
      rle++;
    else{
      atomicAdd(&shared_bins[tloc_masked[i-1]], rle);
      rle=1;
    }
  }
  atomicAdd(&shared_bins[tloc_masked[KPT-1]], rle);
#else
  #pragma unroll
  for(int i=0; i<KPT; i++){
    tloc_masked[i] = (tloc_keys[i]>>((sizeof(KeyT)*8)-(NUM_BITS*(digit+1))))&((0x01<<NUM_BITS)-1);
    atomicAdd(&shared_bins[tloc_masked[i]], 1);
  }
#endif

  // Make sure we've got the counts from all threads
  __syncthreads();

  /*** Write shared histo to global histo ***/
  if(threadIdx.x < 32){
    for(int i=0;i<(0x01<<NUM_BITS);i+=32){
      atomicAdd(&global_histo[(0x01<<NUM_BITS)*bucket_idx+i+threadIdx.x], shared_bins[i+threadIdx.x]);
      // per_block_histo[blockIdx.x*(0x01<<NUM_BITS)+i+threadIdx.x] = shared_bins[i+threadIdx.x];
    }
  }
}

template<
  typename KeyT,    // Data type of the keys within device memory. Data will be twiddled (if necessary) to unsigned type
  typename IndexT,  // Data type used for key's offsets and counters (limits number of supported keys, uint = 2^32)
  int NUM_BITS,     // Number of bits being sorted at a time
  int KPT,          // Number of keys per thread
  int TPB,          // Number of threads per block
  int PRE_SORT_RUNS_LENGTH // For values greater than 1, this causes to sort a thread's keys by runs of a given length to improve run-length encoded updates to shared memory.
>
__global__ void rdxsrt_histogram_with_guards(KeyT *__restrict__ keys, const uint digit, IndexT *global_histo, const IndexT total_keys, const int block_index_offset)
{
  /*** TYPEDEFs***/
  typedef Traits<KeyT>                        KeyTraits;
  typedef typename KeyTraits::UnsignedBits    UnsignedBits;
  /*typedef LoadUnit<IndexT, RDXSRT_LOAD_STRIDE_WARP, KPT, TPB> KeyLoader;*/

  /*** DECLARATIONS ***/
  UnsignedBits tloc_keys[KPT];
  uint tloc_masked[KPT];
  __shared__ uint shared_bins[(0x01<<NUM_BITS) + 1];

  /*** INIT SHARED HISTO ***/
  if (threadIdx.x < 32) {
    #pragma unroll
    for(int i=0;i<(0x01<<NUM_BITS);i+=32){
      shared_bins[i+threadIdx.x] = 0;
    }
  }
  __syncthreads();

  /*** GET KEYS & PREPARE KEYS FOR HISTO ***/
  // Bucket index used to determine the memory offset of the bucket's global histogram
  const uint bucket_idx = 0;
  // This thread block's keys memory offset, pointing to the index of its first key
  const IndexT block_offset = (blockDim.x * (block_index_offset + blockIdx.x) * KPT);

  // Maximum number of keys the block may fetch
  const IndexT block_max_num_keys = total_keys - block_offset;
  // KeyLoader(block_offset, threadIdx.x).template LoadStridedWithGuards<UnsignedBits, KeyT, 0, KPT>(keys, tloc_keys, block_max_num_keys);
  #pragma unroll
  for (int i=0; i<KPT; i++) {
    if ((threadIdx.x + blockDim.x * i) < block_max_num_keys) {
      tloc_keys[i] = reinterpret_cast<UnsignedBits*>(keys)[block_offset + threadIdx.x + blockDim.x * i];
    }
  }

  #pragma unroll
  for(int i=0; i<KPT; i++){
    // if(KeyLoader(block_offset, threadIdx.x).ThreadIndexInBounds(block_max_num_keys, i)){
    if ((threadIdx.x + blockDim.x * i) < block_max_num_keys) {
      tloc_keys[i] = KeyTraits::TwiddleIn(tloc_keys[i]);
      tloc_masked[i] = (tloc_keys[i]>>((sizeof(KeyT)*8)-(NUM_BITS*(digit+1))))&((0x01<<NUM_BITS)-1);
      atomicAdd(&shared_bins[tloc_masked[i]], 1);
    }
  }

  // Make sure we've got the counts from all threads
  __syncthreads();

  /*** Write shared histo to global histo ***/
  if(threadIdx.x < 32){
    for(int i=0;i<(0x01<<NUM_BITS);i+=32){
      atomicAdd(&global_histo[(0x01<<NUM_BITS)*bucket_idx+i+threadIdx.x], shared_bins[i+threadIdx.x]);
      // per_block_histo[(block_index_offset + blockIdx.x)*(0x01<<NUM_BITS)+i+threadIdx.x] = shared_bins[i+threadIdx.x];
    }
  }
}

/**
 * Makes a single pass over the input array to find entries whose digit is equal to selected digit value and greater than
 * digit value. Entries equal to digit value are written to keys_buffer for future processing, entries greater
 * are written to output array.
 * @param d_keys_in        [IN] The keys for which to compute the histogram
 * @param digit            [IN] Digit index (0 => highest digit, 3 => lowest digit for 32-bit)
 * @param digit_val        [IN] Digit value.
 * @param num_items        [IN] Number of entries.
 * @param d_keys_buffer    [OUT] Entries with x[digit] > digit_val.
 * @param d_keys_out       [OUT] Entries with x[digit] > digit_val.
 * @param d_index_buffer   [OUT] Index into d_keys_buffer.
 * @param d_index_out      [OUT] Index into d_keys_out.
 */
template<
  typename KeyT,    // Data type of the keys within device memory. Data will be twiddled (if necessary) to unsigned type
  typename IndexT,  // Data type used for key's offsets and counters (limits number of supported keys, uint = 2^32)
  int NUM_BITS,     // Number of bits being sorted at a time
  int KPT,          // Number of keys per thread
  int TPB           // Number of threads per block
>
__global__ void select_kth_bucket(KeyT* d_keys_in, const uint digit, const uint digit_val, uint num_items,
    KeyT* d_keys_buffer, KeyT* d_keys_out, uint* d_index_buffer, uint* d_index_out)
{
  typedef Traits<KeyT>                        KeyTraits;
  typedef typename KeyTraits::UnsignedBits    UnsignedBits;

  // Specialize BlockLoad for a 1D block of TPB threads owning KPT integer items each
  typedef cub::BlockLoad<UnsignedBits, TPB, KPT, BLOCK_LOAD_TRANSPOSE> BlockLoadT;

  // Specialize BlockScan type for our thread block
  typedef BlockScan<int, TPB, BLOCK_SCAN_RAKING> BlockScanT;

  const int tile_size = TPB * KPT;
  int tile_idx = blockIdx.x;    // Current tile index
  int tile_offset = tile_idx * tile_size;

  // Allocate shared memory for BlockLoad
  __shared__ union TempStorage
  {
    typename BlockLoadT::TempStorage    load_items;
    typename BlockScanT::TempStorage    scan;
    int offset[1];
    UnsignedBits raw_exchange[2 * TPB * KPT];
  } temp_storage;

  // Load a segment of consecutive items that are blocked across threads
  UnsignedBits key_entries[KPT];
  /*float payload_entries[KPT];*/
  int selection_flags[KPT];
  int selection_indices[KPT];

  int num_tiles = (num_items + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (tile_idx == num_tiles - 1) {
    num_tile_items = num_items - tile_offset;
    is_last_tile = true;
  }

  // Load keys
  if (is_last_tile)
    BlockLoadT(temp_storage.load_items).Load(reinterpret_cast<UnsignedBits*>(d_keys_in) + tile_offset, key_entries, num_tile_items);
  else
    BlockLoadT(temp_storage.load_items).Load(reinterpret_cast<UnsignedBits*>(d_keys_in) + tile_offset, key_entries);


#if 0
  if (is_last_tile)
    BlockLoadT(temp_storage.load_items).Load(payload + tile_offset, payload_entries, num_tile_items);
  else
    BlockLoadT(temp_storage.load_items).Load(payload + tile_offset, payload_entries);
#endif

  __syncthreads();

  /*** Step 1: Find keys with digit value to selected digit value ***/
  #pragma unroll
  for (int ITEM = 0; ITEM < KPT; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    selection_flags[ITEM] = 0;

    if (!is_last_tile || (int(threadIdx.x * KPT) + ITEM < num_tile_items)) {
      UnsignedBits key = KeyTraits::TwiddleIn(key_entries[ITEM]);
      uint masked_key = (key>>((sizeof(KeyT)*8)-(NUM_BITS*(digit+1))))&((0x01<<NUM_BITS)-1);
      selection_flags[ITEM] = (masked_key > digit_val);
    }
  }

  __syncthreads();

  // Compute exclusive prefix sum
  int num_selected;
  BlockScanT(temp_storage.scan).ExclusiveSum(selection_flags, selection_indices, num_selected);

  __syncthreads();

  if (num_selected > 0) {
    int index_out;
    if (threadIdx.x == 0) {
      // Find index into keys_out array
      index_out = atomicAdd(d_index_out, num_selected);
      temp_storage.offset[0] = index_out;
    }

    __syncthreads();

    index_out = temp_storage.offset[0];

    __syncthreads();

    // Compact and scatter items
    #pragma unroll
    for (int ITEM = 0; ITEM < KPT; ++ITEM)
    {
      int local_scatter_offset = selection_indices[ITEM];
      if (selection_flags[ITEM])
      {
        temp_storage.raw_exchange[local_scatter_offset] = key_entries[ITEM];
        /*temp_storage.raw_exchange[tile_size + local_scatter_offset] = payload_entries[ITEM];*/
      }
    }

    __syncthreads();

    // Write out matched entries to output array
    for (int item = threadIdx.x; item < num_selected; item += TPB)
    {
      reinterpret_cast<UnsignedBits*>(d_keys_out)[index_out + item] = temp_storage.raw_exchange[item];
    }

    __syncthreads();

#if 0
    for (int item = threadIdx.x; item < num_selected; item += TPB)
    {
      payload_out[num_selections_prefix + item] = temp_storage.raw_exchange[tile_size + item];
    }
#endif
  }

  /*** Step 2: Find entries that have digit equal to digit value ***/
  #pragma unroll
  for (int ITEM = 0; ITEM < KPT; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    selection_flags[ITEM] = 0;

    if (!is_last_tile || (int(threadIdx.x * KPT) + ITEM < num_tile_items)) {
      UnsignedBits key = KeyTraits::TwiddleIn(key_entries[ITEM]);
      uint masked_key = (key>>((sizeof(KeyT)*8)-(NUM_BITS*(digit+1))))&((0x01<<NUM_BITS)-1);
      selection_flags[ITEM] = (masked_key == digit_val);
    }
  }

  __syncthreads();

  // Compute exclusive prefix sum
  BlockScanT(temp_storage.scan).ExclusiveSum(selection_flags, selection_indices, num_selected);

  __syncthreads();

  if (num_selected > 0) {
    int index_buffer;
    if (threadIdx.x == 0) {
      index_buffer = atomicAdd(d_index_buffer, num_selected);
      temp_storage.offset[0] = index_buffer;
    }

    __syncthreads();

    index_buffer = temp_storage.offset[0];

    __syncthreads();

    // Compact and scatter items
    #pragma unroll
    for (int ITEM = 0; ITEM < KPT; ++ITEM)
    {
      int local_scatter_offset = selection_indices[ITEM];
      if (selection_flags[ITEM])
      {
        temp_storage.raw_exchange[local_scatter_offset] = key_entries[ITEM];
        /*temp_storage.raw_exchange[tile_size + local_scatter_offset] = payload_entries[ITEM];*/
      }
    }

    __syncthreads();

    // Write out output entries
    for (int item = threadIdx.x; item < num_selected; item += TPB)
    {
      reinterpret_cast<UnsignedBits*>(d_keys_buffer)[index_buffer + item] = temp_storage.raw_exchange[item];
    }

    __syncthreads();
  }
}

#define KPT 16
#define TPB 384
#define DIGIT_BITS 8
template<typename KeyT>
cudaError_t radixSelectTopK(KeyT *d_keys_in, uint num_items, uint k, KeyT *d_keys_out,
    CachingDeviceAllocator&  g_allocator) {
  cudaError error = cudaSuccess;

  DoubleBuffer<KeyT> d_keys;
  d_keys.d_buffers[0] = d_keys_in;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(KeyT) * num_items));

  uint* d_histogram;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_histogram, sizeof(uint) * num_items));

  // We allocate two indices, one that maintains index into output array (this goes till K)
  // second maintains index into the output buffer containing reduced set of top-k candidates.
  uint* d_index_out;
  uint* d_index_buffer;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_index_out, sizeof(uint)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_index_buffer, sizeof(uint)));

  // Set the index into output array to 0.
  cudaMemset(d_index_out, 0, 4);

  uint* h_histogram = new uint[256];

  uint KPB = KPT * TPB;

  for (uint digit = 0; digit < 4; digit++) {
    uint num_blocks = num_items / KPB;// Pass-0 rough processing blocks (floor on purpose)
    uint processed_elements = num_blocks * KPB;// Pass-0 number of rough processed elements
    uint remaining_elements = num_items - processed_elements;// Do the remaining elements with a check in the inner loop
    uint remainder_blocks = (KPB-1+remaining_elements) / KPB;// Number of blocks required for remaining elements (typically 0 or 1)

    // Zero out the histogram
    cudaMemset(d_histogram, 0, 256 * 4);

    if (num_blocks > 0)
      rdxsrt_histogram<KeyT, uint, DIGIT_BITS, KPT, TPB, 9><<<num_blocks, TPB, 0>>>(d_keys.Current(), digit, d_histogram);
    if (remaining_elements > 0)
      rdxsrt_histogram_with_guards<KeyT, uint, DIGIT_BITS, KPT, TPB, 9><<<remainder_blocks, TPB, 0>>>(d_keys.Current(), digit, d_histogram, num_items, num_blocks);

    cudaMemcpy(h_histogram, d_histogram, 256 * sizeof(uint), cudaMemcpyDeviceToHost);

    // Check for failure to launch
    CubDebugExit(error = cudaPeekAtLastError());

    cudaMemcpy(h_histogram, d_histogram, 256 * sizeof(uint), cudaMemcpyDeviceToHost);
    uint rolling_sum = 0;
    uint digit_val;
    for (int i=255; i>=0; i--) {
      if ((rolling_sum + h_histogram[i]) > k) {
        digit_val = i;
        k -= rolling_sum;
        break;
      }
      rolling_sum += h_histogram[i];
    }

    cudaMemset(d_index_buffer, 0, 4);

    select_kth_bucket<KeyT, uint, DIGIT_BITS, KPT, TPB><<<num_blocks + remainder_blocks, TPB>>>(d_keys.Current(), digit, digit_val, num_items, d_keys.Alternate(), d_keys_out, d_index_buffer, d_index_out);

    CubDebugExit(error = cudaPeekAtLastError());

    uint h_index_out;
    uint h_index_buffer;

    cudaMemcpy(&h_index_out, d_index_out, sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_index_buffer, d_index_buffer, sizeof(uint), cudaMemcpyDeviceToHost);

    // Update number of items to reflect reduced number of elements.
    num_items = h_index_buffer;

    if (k == 0) break;
    else if (k != 0 && digit == 3) {
      // We are at last digit and k != 4 implies that kth value has repetition.
      // Copy any of the repeated values to out array to complete the array.
      cudaMemcpy(d_keys_out + h_index_out, d_keys.Alternate(), k * sizeof(KeyT), cudaMemcpyDeviceToDevice);
      k -= k;
    }

    // Toggle the buffer index in the double buffer
    d_keys.selector = d_keys.selector ^ 1;
  }

  // Cleanup
  if (d_keys.d_buffers[1])
    CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));
  if (d_histogram)
    CubDebugExit(g_allocator.DeviceFree(d_histogram));
  if (d_index_buffer)
    CubDebugExit(g_allocator.DeviceFree(d_index_buffer));
  if (d_index_out)
    CubDebugExit(g_allocator.DeviceFree(d_index_out));

  return error;
}

