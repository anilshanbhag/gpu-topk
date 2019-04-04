template<typename KeyT>
struct SortingNetwork
{
  enum { NETWORK_MAX_NUM_ITEMS = 18 };

private:
  static __device__ __forceinline__ void compare_and_swap(KeyT *keys, int low_idx, int high_idx)
  {
    if(keys[low_idx]>keys[high_idx]){
      KeyT tmp = keys[low_idx];
      keys[low_idx] = keys[high_idx];
      keys[high_idx] = tmp;
    }
  }

  template<int NUM_ITEMS, int DUMMY>
  struct SortingNetworkAgent
  {
    static __device__ __forceinline__ void sort(KeyT *keys){
#if __cplusplus >= 199711
    static_assert(NUM_ITEMS<=NETWORK_MAX_NUM_ITEMS, "Sorting network of given size not supported.");
#endif
    }
  };

  /*** SORTING NETWORK 1 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<1, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys){}
  };

  /*** SORTING NETWORK 2 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<2, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys){
      compare_and_swap(keys, 0, 1);
    }
  };

  /*** SORTING NETWORK 3 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<3, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys)
    {
      compare_and_swap(keys, 1, 2);
      compare_and_swap(keys, 0, 2);
      compare_and_swap(keys, 0, 1);
    }
  };

  /*** SORTING NETWORK 4 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<4, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys)
    {
      compare_and_swap(keys, 0, 1);
      compare_and_swap(keys, 2, 3);
      compare_and_swap(keys, 0, 2);
      compare_and_swap(keys, 1, 3);
      compare_and_swap(keys, 1, 2);
    }
  };

  /*** SORTING NETWORK 5 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<5, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys)
    {
      compare_and_swap(keys, 0, 1);
      compare_and_swap(keys, 3, 4);
      compare_and_swap(keys, 2, 4);
      compare_and_swap(keys, 2, 3);
      compare_and_swap(keys, 0, 3);
      compare_and_swap(keys, 0, 2);
      compare_and_swap(keys, 1, 4);
      compare_and_swap(keys, 1, 3);
      compare_and_swap(keys, 1, 2);
    }
  };

  /*** SORTING NETWORK 6 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<6, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys)
    {
      compare_and_swap(keys, 1, 2);
      compare_and_swap(keys, 0, 2);
      compare_and_swap(keys, 0, 1);
      compare_and_swap(keys, 4, 5);
      compare_and_swap(keys, 3, 5);
      compare_and_swap(keys, 3, 4);
      compare_and_swap(keys, 0, 3);
      compare_and_swap(keys, 1, 4);
      compare_and_swap(keys, 2, 5);
      compare_and_swap(keys, 2, 4);
      compare_and_swap(keys, 1, 3);
      compare_and_swap(keys, 2, 3);
    }
  };

  /*** SORTING NETWORK 7 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<7, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys)
    {
      compare_and_swap(keys, 1, 2);
      compare_and_swap(keys, 0, 2);
      compare_and_swap(keys, 0, 1);
      compare_and_swap(keys, 3, 4);
      compare_and_swap(keys, 5, 6);
      compare_and_swap(keys, 3, 5);
      compare_and_swap(keys, 4, 6);
      compare_and_swap(keys, 4, 5);
      compare_and_swap(keys, 0, 4);
      compare_and_swap(keys, 0, 3);
      compare_and_swap(keys, 1, 5);
      compare_and_swap(keys, 2, 6);
      compare_and_swap(keys, 2, 5);
      compare_and_swap(keys, 1, 3);
      compare_and_swap(keys, 2, 4);
      compare_and_swap(keys, 2, 3);
    }
  };

  /*** SORTING NETWORK 8 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<8, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys)
    {
      compare_and_swap(keys, 0, 1);
      compare_and_swap(keys, 2, 3);
      compare_and_swap(keys, 0, 2);
      compare_and_swap(keys, 1, 3);
      compare_and_swap(keys, 1, 2);
      compare_and_swap(keys, 4, 5);
      compare_and_swap(keys, 6, 7);
      compare_and_swap(keys, 4, 6);
      compare_and_swap(keys, 5, 7);
      compare_and_swap(keys, 5, 6);
      compare_and_swap(keys, 0, 4);
      compare_and_swap(keys, 1, 5);
      compare_and_swap(keys, 1, 4);
      compare_and_swap(keys, 2, 6);
      compare_and_swap(keys, 3, 7);
      compare_and_swap(keys, 3, 6);
      compare_and_swap(keys, 2, 4);
      compare_and_swap(keys, 3, 5);
      compare_and_swap(keys, 3, 4);
    }
  };

  /*** SORTING NETWORK 9 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<9, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys)
    {
      compare_and_swap(keys, 1, 2);
      compare_and_swap(keys, 3, 4);
      compare_and_swap(keys, 5, 6);
      compare_and_swap(keys, 7, 8);
      compare_and_swap(keys, 0, 1);
      compare_and_swap(keys, 2, 4);
      compare_and_swap(keys, 6, 8);
      compare_and_swap(keys, 5, 7);
      compare_and_swap(keys, 0, 3);
      compare_and_swap(keys, 4, 8);
      compare_and_swap(keys, 1, 2);
      compare_and_swap(keys, 6, 7);
      compare_and_swap(keys, 3, 7);
      compare_and_swap(keys, 1, 6);
      compare_and_swap(keys, 0, 5);
      compare_and_swap(keys, 2, 4);
      compare_and_swap(keys, 4, 7);
      compare_and_swap(keys, 2, 6);
      compare_and_swap(keys, 3, 5);
      compare_and_swap(keys, 1, 3);
      compare_and_swap(keys, 4, 6);
      compare_and_swap(keys, 7, 8);
      compare_and_swap(keys, 2, 5);
      compare_and_swap(keys, 2, 3);
      compare_and_swap(keys, 4, 5);
    }
  };

  /*** SORTING NETWORK 10 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<10, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys)
    {
      compare_and_swap(keys, 4, 9);
      compare_and_swap(keys, 3, 8);
      compare_and_swap(keys, 2, 7);
      compare_and_swap(keys, 1, 6);
      compare_and_swap(keys, 0, 5);
      compare_and_swap(keys, 1, 4);
      compare_and_swap(keys, 6, 9);
      compare_and_swap(keys, 0, 3);
      compare_and_swap(keys, 5, 8);
      compare_and_swap(keys, 0, 2);
      compare_and_swap(keys, 3, 6);
      compare_and_swap(keys, 7, 9);
      compare_and_swap(keys, 0, 1);
      compare_and_swap(keys, 2, 4);
      compare_and_swap(keys, 5, 7);
      compare_and_swap(keys, 8, 9);
      compare_and_swap(keys, 1, 2);
      compare_and_swap(keys, 4, 6);
      compare_and_swap(keys, 7, 8);
      compare_and_swap(keys, 3, 5);
      compare_and_swap(keys, 2, 5);
      compare_and_swap(keys, 6, 8);
      compare_and_swap(keys, 1, 3);
      compare_and_swap(keys, 4, 7);
      compare_and_swap(keys, 2, 3);
      compare_and_swap(keys, 6, 7);
      compare_and_swap(keys, 3, 4);
      compare_and_swap(keys, 5, 6);
      compare_and_swap(keys, 4, 5);
    }
  };


  /*** SORTING NETWORK 11 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<11, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys)
    {
      compare_and_swap(keys, 0, 1);
      compare_and_swap(keys, 2, 3);
      compare_and_swap(keys, 4, 5);
      compare_and_swap(keys, 6, 7);
      compare_and_swap(keys, 8, 9);
      compare_and_swap(keys, 1, 3);
      compare_and_swap(keys, 5, 7);
      compare_and_swap(keys, 0, 2);
      compare_and_swap(keys, 4, 6);
      compare_and_swap(keys, 8, 10);
      compare_and_swap(keys, 1, 2);
      compare_and_swap(keys, 5, 6);
      compare_and_swap(keys, 9, 10);
      compare_and_swap(keys, 1, 5);
      compare_and_swap(keys, 6, 10);
      compare_and_swap(keys, 5, 9);
      compare_and_swap(keys, 2, 6);
      compare_and_swap(keys, 1, 5);
      compare_and_swap(keys, 6, 10);
      compare_and_swap(keys, 0, 4);
      compare_and_swap(keys, 3, 7);
      compare_and_swap(keys, 4, 8);
      compare_and_swap(keys, 0, 4);
      compare_and_swap(keys, 1, 4);
      compare_and_swap(keys, 7, 10);
      compare_and_swap(keys, 3, 8);
      compare_and_swap(keys, 2, 3);
      compare_and_swap(keys, 8, 9);
      compare_and_swap(keys, 2, 4);
      compare_and_swap(keys, 7, 9);
      compare_and_swap(keys, 3, 5);
      compare_and_swap(keys, 6, 8);
      compare_and_swap(keys, 3, 4);
    }
  };

  /*** SORTING NETWORK 12 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<12, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys)
    {
      compare_and_swap(keys, 0, 1);
      compare_and_swap(keys, 2, 3);
      compare_and_swap(keys, 4, 5);
      compare_and_swap(keys, 6, 7);
      compare_and_swap(keys, 8, 9);
      compare_and_swap(keys, 10, 11);
      compare_and_swap(keys, 1, 3);
      compare_and_swap(keys, 5, 7);
      compare_and_swap(keys, 9, 11);
      compare_and_swap(keys, 0, 2);
      compare_and_swap(keys, 4, 6);
      compare_and_swap(keys, 8, 10);
      compare_and_swap(keys, 1, 2);
      compare_and_swap(keys, 5, 6);
      compare_and_swap(keys, 9, 10);
      compare_and_swap(keys, 1, 5);
      compare_and_swap(keys, 6, 10);
      compare_and_swap(keys, 5, 9);
      compare_and_swap(keys, 2, 6);
      compare_and_swap(keys, 1, 5);
      compare_and_swap(keys, 6, 10);
      compare_and_swap(keys, 0, 4);
      compare_and_swap(keys, 7, 11);
      compare_and_swap(keys, 3, 7);
      compare_and_swap(keys, 4, 8);
      compare_and_swap(keys, 0, 4);
      compare_and_swap(keys, 7, 11);
      compare_and_swap(keys, 1, 4);
      compare_and_swap(keys, 7, 10);
      compare_and_swap(keys, 3, 8);
      compare_and_swap(keys, 2, 3);
      compare_and_swap(keys, 8, 9);
      compare_and_swap(keys, 2, 4);
      compare_and_swap(keys, 7, 9);
      compare_and_swap(keys, 3, 5);
      compare_and_swap(keys, 6, 8);
      compare_and_swap(keys, 3, 4);
      compare_and_swap(keys, 5, 6);
      compare_and_swap(keys, 7, 8);
    }
  };

  /*** SORTING NETWORK 13 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<13, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys)
    {
      compare_and_swap(keys, 1, 7);
      compare_and_swap(keys, 9, 11);
      compare_and_swap(keys, 3, 4);
      compare_and_swap(keys, 5, 8);
      compare_and_swap(keys, 0, 12);
      compare_and_swap(keys, 2, 6);
      compare_and_swap(keys, 0, 1);
      compare_and_swap(keys, 2, 3);
      compare_and_swap(keys, 4, 6);
      compare_and_swap(keys, 8, 11);
      compare_and_swap(keys, 7, 12);
      compare_and_swap(keys, 5, 9);
      compare_and_swap(keys, 0, 2);
      compare_and_swap(keys, 3, 7);
      compare_and_swap(keys, 10, 11);
      compare_and_swap(keys, 1, 4);
      compare_and_swap(keys, 6, 12);
      compare_and_swap(keys, 7, 8);
      compare_and_swap(keys, 11, 12);
      compare_and_swap(keys, 4, 9);
      compare_and_swap(keys, 6, 10);
      compare_and_swap(keys, 3, 4);
      compare_and_swap(keys, 5, 6);
      compare_and_swap(keys, 8, 9);
      compare_and_swap(keys, 10, 11);
      compare_and_swap(keys, 1, 7);
      compare_and_swap(keys, 2, 6);
      compare_and_swap(keys, 9, 11);
      compare_and_swap(keys, 1, 3);
      compare_and_swap(keys, 4, 7);
      compare_and_swap(keys, 8, 10);
      compare_and_swap(keys, 0, 5);
      compare_and_swap(keys, 2, 5);
      compare_and_swap(keys, 6, 8);
      compare_and_swap(keys, 9, 10);
      compare_and_swap(keys, 1, 2);
      compare_and_swap(keys, 3, 5);
      compare_and_swap(keys, 7, 8);
      compare_and_swap(keys, 4, 6);
      compare_and_swap(keys, 2, 3);
      compare_and_swap(keys, 4, 5);
      compare_and_swap(keys, 6, 7);
      compare_and_swap(keys, 8, 9);
      compare_and_swap(keys, 3, 4);
      compare_and_swap(keys, 5, 6);
    }
  };

  /*** SORTING NETWORK 14 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<14, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys)
    {
      compare_and_swap(keys, 0, 1);
      compare_and_swap(keys, 2, 3);
      compare_and_swap(keys, 4, 5);
      compare_and_swap(keys, 6, 7);
      compare_and_swap(keys, 8, 9);
      compare_and_swap(keys, 10, 11);
      compare_and_swap(keys, 12, 13);
      compare_and_swap(keys, 0, 2);
      compare_and_swap(keys, 4, 6);
      compare_and_swap(keys, 8, 10);
      compare_and_swap(keys, 1, 3);
      compare_and_swap(keys, 5, 7);
      compare_and_swap(keys, 9, 11);
      compare_and_swap(keys, 0, 4);
      compare_and_swap(keys, 8, 12);
      compare_and_swap(keys, 1, 5);
      compare_and_swap(keys, 9, 13);
      compare_and_swap(keys, 2, 6);
      compare_and_swap(keys, 3, 7);
      compare_and_swap(keys, 0, 8);
      compare_and_swap(keys, 1, 9);
      compare_and_swap(keys, 2, 10);
      compare_and_swap(keys, 3, 11);
      compare_and_swap(keys, 4, 12);
      compare_and_swap(keys, 5, 13);
      compare_and_swap(keys, 5, 10);
      compare_and_swap(keys, 6, 9);
      compare_and_swap(keys, 3, 12);
      compare_and_swap(keys, 7, 11);
      compare_and_swap(keys, 1, 2);
      compare_and_swap(keys, 4, 8);
      compare_and_swap(keys, 1, 4);
      compare_and_swap(keys, 7, 13);
      compare_and_swap(keys, 2, 8);
      compare_and_swap(keys, 2, 4);
      compare_and_swap(keys, 5, 6);
      compare_and_swap(keys, 9, 10);
      compare_and_swap(keys, 11, 13);
      compare_and_swap(keys, 3, 8);
      compare_and_swap(keys, 7, 12);
      compare_and_swap(keys, 6, 8);
      compare_and_swap(keys, 10, 12);
      compare_and_swap(keys, 3, 5);
      compare_and_swap(keys, 7, 9);
      compare_and_swap(keys, 3, 4);
      compare_and_swap(keys, 5, 6);
      compare_and_swap(keys, 7, 8);
      compare_and_swap(keys, 9, 10);
      compare_and_swap(keys, 11, 12);
      compare_and_swap(keys, 6, 7);
      compare_and_swap(keys, 8, 9);
    }
  };

  /*** SORTING NETWORK 15 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<15, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys)
    {
      compare_and_swap(keys, 0, 1);
      compare_and_swap(keys, 2, 3);
      compare_and_swap(keys, 4, 5);
      compare_and_swap(keys, 6, 7);
      compare_and_swap(keys, 8, 9);
      compare_and_swap(keys, 10, 11);
      compare_and_swap(keys, 12, 13);
      compare_and_swap(keys, 0, 2);
      compare_and_swap(keys, 4, 6);
      compare_and_swap(keys, 8, 10);
      compare_and_swap(keys, 12, 14);
      compare_and_swap(keys, 1, 3);
      compare_and_swap(keys, 5, 7);
      compare_and_swap(keys, 9, 11);
      compare_and_swap(keys, 0, 4);
      compare_and_swap(keys, 8, 12);
      compare_and_swap(keys, 1, 5);
      compare_and_swap(keys, 9, 13);
      compare_and_swap(keys, 2, 6);
      compare_and_swap(keys, 10, 14);
      compare_and_swap(keys, 3, 7);
      compare_and_swap(keys, 0, 8);
      compare_and_swap(keys, 1, 9);
      compare_and_swap(keys, 2, 10);
      compare_and_swap(keys, 3, 11);
      compare_and_swap(keys, 4, 12);
      compare_and_swap(keys, 5, 13);
      compare_and_swap(keys, 6, 14);
      compare_and_swap(keys, 5, 10);
      compare_and_swap(keys, 6, 9);
      compare_and_swap(keys, 3, 12);
      compare_and_swap(keys, 13, 14);
      compare_and_swap(keys, 7, 11);
      compare_and_swap(keys, 1, 2);
      compare_and_swap(keys, 4, 8);
      compare_and_swap(keys, 1, 4);
      compare_and_swap(keys, 7, 13);
      compare_and_swap(keys, 2, 8);
      compare_and_swap(keys, 11, 14);
      compare_and_swap(keys, 2, 4);
      compare_and_swap(keys, 5, 6);
      compare_and_swap(keys, 9, 10);
      compare_and_swap(keys, 11, 13);
      compare_and_swap(keys, 3, 8);
      compare_and_swap(keys, 7, 12);
      compare_and_swap(keys, 6, 8);
      compare_and_swap(keys, 10, 12);
      compare_and_swap(keys, 3, 5);
      compare_and_swap(keys, 7, 9);
      compare_and_swap(keys, 3, 4);
      compare_and_swap(keys, 5, 6);
      compare_and_swap(keys, 7, 8);
      compare_and_swap(keys, 9, 10);
      compare_and_swap(keys, 11, 12);
      compare_and_swap(keys, 6, 7);
      compare_and_swap(keys, 8, 9);
    }
  };

  /*** SORTING NETWORK 16 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<16, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys)
    {
      compare_and_swap(keys, 0, 1);
      compare_and_swap(keys, 2, 3);
      compare_and_swap(keys, 4, 5);
      compare_and_swap(keys, 6, 7);
      compare_and_swap(keys, 8, 9);
      compare_and_swap(keys, 10, 11);
      compare_and_swap(keys, 12, 13);
      compare_and_swap(keys, 14, 15);
      compare_and_swap(keys, 0, 2);
      compare_and_swap(keys, 4, 6);
      compare_and_swap(keys, 8, 10);
      compare_and_swap(keys, 12, 14);
      compare_and_swap(keys, 1, 3);
      compare_and_swap(keys, 5, 7);
      compare_and_swap(keys, 9, 11);
      compare_and_swap(keys, 13, 15);
      compare_and_swap(keys, 0, 4);
      compare_and_swap(keys, 8, 12);
      compare_and_swap(keys, 1, 5);
      compare_and_swap(keys, 9, 13);
      compare_and_swap(keys, 2, 6);
      compare_and_swap(keys, 10, 14);
      compare_and_swap(keys, 3, 7);
      compare_and_swap(keys, 11, 15);
      compare_and_swap(keys, 0, 8);
      compare_and_swap(keys, 1, 9);
      compare_and_swap(keys, 2, 10);
      compare_and_swap(keys, 3, 11);
      compare_and_swap(keys, 4, 12);
      compare_and_swap(keys, 5, 13);
      compare_and_swap(keys, 6, 14);
      compare_and_swap(keys, 7, 15);
      compare_and_swap(keys, 5, 10);
      compare_and_swap(keys, 6, 9);
      compare_and_swap(keys, 3, 12);
      compare_and_swap(keys, 13, 14);
      compare_and_swap(keys, 7, 11);
      compare_and_swap(keys, 1, 2);
      compare_and_swap(keys, 4, 8);
      compare_and_swap(keys, 1, 4);
      compare_and_swap(keys, 7, 13);
      compare_and_swap(keys, 2, 8);
      compare_and_swap(keys, 11, 14);
      compare_and_swap(keys, 2, 4);
      compare_and_swap(keys, 5, 6);
      compare_and_swap(keys, 9, 10);
      compare_and_swap(keys, 11, 13);
      compare_and_swap(keys, 3, 8);
      compare_and_swap(keys, 7, 12);
      compare_and_swap(keys, 6, 8);
      compare_and_swap(keys, 10, 12);
      compare_and_swap(keys, 3, 5);
      compare_and_swap(keys, 7, 9);
      compare_and_swap(keys, 3, 4);
      compare_and_swap(keys, 5, 6);
      compare_and_swap(keys, 7, 8);
      compare_and_swap(keys, 9, 10);
      compare_and_swap(keys, 11, 12);
      compare_and_swap(keys, 6, 7);
      compare_and_swap(keys, 8, 9);
    }
  };

  /*** SORTING NETWORK 17 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<17, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys)
    {
      compare_and_swap(keys, 0, 1);
      compare_and_swap(keys, 2, 3);
      compare_and_swap(keys, 0, 2);
      compare_and_swap(keys, 1, 3);
      compare_and_swap(keys, 1, 2);
      compare_and_swap(keys, 4, 5);
      compare_and_swap(keys, 6, 7);
      compare_and_swap(keys, 4, 6);
      compare_and_swap(keys, 5, 7);
      compare_and_swap(keys, 5, 6);
      compare_and_swap(keys, 0, 4);
      compare_and_swap(keys, 1, 5);
      compare_and_swap(keys, 1, 4);
      compare_and_swap(keys, 2, 6);
      compare_and_swap(keys, 3, 7);
      compare_and_swap(keys, 3, 6);
      compare_and_swap(keys, 2, 4);
      compare_and_swap(keys, 3, 5);
      compare_and_swap(keys, 3, 4);
      compare_and_swap(keys, 8, 9);
      compare_and_swap(keys, 10, 11);
      compare_and_swap(keys, 8, 10);
      compare_and_swap(keys, 9, 11);
      compare_and_swap(keys, 9, 10);
      compare_and_swap(keys, 12, 13);
      compare_and_swap(keys, 15, 16);
      compare_and_swap(keys, 14, 16);
      compare_and_swap(keys, 14, 15);
      compare_and_swap(keys, 12, 15);
      compare_and_swap(keys, 12, 14);
      compare_and_swap(keys, 13, 16);
      compare_and_swap(keys, 13, 15);
      compare_and_swap(keys, 13, 14);
      compare_and_swap(keys, 8, 13);
      compare_and_swap(keys, 8, 12);
      compare_and_swap(keys, 9, 14);
      compare_and_swap(keys, 9, 13);
      compare_and_swap(keys, 9, 12);
      compare_and_swap(keys, 10, 15);
      compare_and_swap(keys, 11, 16);
      compare_and_swap(keys, 11, 15);
      compare_and_swap(keys, 10, 13);
      compare_and_swap(keys, 10, 12);
      compare_and_swap(keys, 11, 14);
      compare_and_swap(keys, 11, 13);
      compare_and_swap(keys, 11, 12);
      compare_and_swap(keys, 0, 9);
      compare_and_swap(keys, 0, 8);
      compare_and_swap(keys, 1, 10);
      compare_and_swap(keys, 1, 9);
      compare_and_swap(keys, 1, 8);
      compare_and_swap(keys, 2, 11);
      compare_and_swap(keys, 3, 12);
      compare_and_swap(keys, 3, 11);
      compare_and_swap(keys, 2, 9);
      compare_and_swap(keys, 2, 8);
      compare_and_swap(keys, 3, 10);
      compare_and_swap(keys, 3, 9);
      compare_and_swap(keys, 3, 8);
      compare_and_swap(keys, 4, 13);
      compare_and_swap(keys, 5, 14);
      compare_and_swap(keys, 5, 13);
      compare_and_swap(keys, 6, 15);
      compare_and_swap(keys, 7, 16);
      compare_and_swap(keys, 7, 15);
      compare_and_swap(keys, 6, 13);
      compare_and_swap(keys, 7, 14);
      compare_and_swap(keys, 7, 13);
      compare_and_swap(keys, 4, 9);
      compare_and_swap(keys, 4, 8);
      compare_and_swap(keys, 5, 10);
      compare_and_swap(keys, 5, 9);
      compare_and_swap(keys, 5, 8);
      compare_and_swap(keys, 6, 11);
      compare_and_swap(keys, 7, 12);
      compare_and_swap(keys, 7, 11);
      compare_and_swap(keys, 6, 9);
      compare_and_swap(keys, 6, 8);
      compare_and_swap(keys, 7, 10);
      compare_and_swap(keys, 7, 9);
      compare_and_swap(keys, 7, 8);
    }
  };

  /*** SORTING NETWORK 18 ***/
  template<int DUMMY>
  struct SortingNetworkAgent<18, DUMMY>
  {
    static __device__ __forceinline__ void sort(KeyT *keys)
    {
      compare_and_swap(keys, 0, 1);
      compare_and_swap(keys, 2, 3);
      compare_and_swap(keys, 0, 2);
      compare_and_swap(keys, 1, 3);
      compare_and_swap(keys, 1, 2);
      compare_and_swap(keys, 4, 5);
      compare_and_swap(keys, 7, 8);
      compare_and_swap(keys, 6, 8);
      compare_and_swap(keys, 6, 7);
      compare_and_swap(keys, 4, 7);
      compare_and_swap(keys, 4, 6);
      compare_and_swap(keys, 5, 8);
      compare_and_swap(keys, 5, 7);
      compare_and_swap(keys, 5, 6);
      compare_and_swap(keys, 0, 5);
      compare_and_swap(keys, 0, 4);
      compare_and_swap(keys, 1, 6);
      compare_and_swap(keys, 1, 5);
      compare_and_swap(keys, 1, 4);
      compare_and_swap(keys, 2, 7);
      compare_and_swap(keys, 3, 8);
      compare_and_swap(keys, 3, 7);
      compare_and_swap(keys, 2, 5);
      compare_and_swap(keys, 2, 4);
      compare_and_swap(keys, 3, 6);
      compare_and_swap(keys, 3, 5);
      compare_and_swap(keys, 3, 4);
      compare_and_swap(keys, 9, 10);
      compare_and_swap(keys, 11, 12);
      compare_and_swap(keys, 9, 11);
      compare_and_swap(keys, 10, 12);
      compare_and_swap(keys, 10, 11);
      compare_and_swap(keys, 13, 14);
      compare_and_swap(keys, 16, 17);
      compare_and_swap(keys, 15, 17);
      compare_and_swap(keys, 15, 16);
      compare_and_swap(keys, 13, 16);
      compare_and_swap(keys, 13, 15);
      compare_and_swap(keys, 14, 17);
      compare_and_swap(keys, 14, 16);
      compare_and_swap(keys, 14, 15);
      compare_and_swap(keys, 9, 14);
      compare_and_swap(keys, 9, 13);
      compare_and_swap(keys, 10, 15);
      compare_and_swap(keys, 10, 14);
      compare_and_swap(keys, 10, 13);
      compare_and_swap(keys, 11, 16);
      compare_and_swap(keys, 12, 17);
      compare_and_swap(keys, 12, 16);
      compare_and_swap(keys, 11, 14);
      compare_and_swap(keys, 11, 13);
      compare_and_swap(keys, 12, 15);
      compare_and_swap(keys, 12, 14);
      compare_and_swap(keys, 12, 13);
      compare_and_swap(keys, 0, 9);
      compare_and_swap(keys, 1, 10);
      compare_and_swap(keys, 1, 9);
      compare_and_swap(keys, 2, 11);
      compare_and_swap(keys, 3, 12);
      compare_and_swap(keys, 3, 11);
      compare_and_swap(keys, 2, 9);
      compare_and_swap(keys, 3, 10);
      compare_and_swap(keys, 3, 9);
      compare_and_swap(keys, 4, 13);
      compare_and_swap(keys, 5, 14);
      compare_and_swap(keys, 5, 13);
      compare_and_swap(keys, 6, 15);
      compare_and_swap(keys, 7, 16);
      compare_and_swap(keys, 8, 17);
      compare_and_swap(keys, 8, 16);
      compare_and_swap(keys, 7, 15);
      compare_and_swap(keys, 8, 15);
      compare_and_swap(keys, 6, 13);
      compare_and_swap(keys, 7, 14);
      compare_and_swap(keys, 8, 14);
      compare_and_swap(keys, 7, 13);
      compare_and_swap(keys, 8, 13);
      compare_and_swap(keys, 4, 9);
      compare_and_swap(keys, 5, 10);
      compare_and_swap(keys, 5, 9);
      compare_and_swap(keys, 6, 11);
      compare_and_swap(keys, 7, 12);
      compare_and_swap(keys, 8, 12);
      compare_and_swap(keys, 7, 11);
      compare_and_swap(keys, 8, 11);
      compare_and_swap(keys, 6, 9);
      compare_and_swap(keys, 7, 10);
      compare_and_swap(keys, 8, 10);
      compare_and_swap(keys, 7, 9);
      compare_and_swap(keys, 8, 9);
    }
  };

  template<
    int RUN_LENGTH,
    int NUM_RUN_ITEMS,  // Number of items to sort in this run (starting at keys)
    bool IS_LAST_RUN  // If this run is the last run to be sorted
  >
  struct SortRunAgent
  {
    static __device__ __forceinline__ void sort_runs(KeyT *keys)
    {
      SortingNetwork<KeyT>::sort<RUN_LENGTH>(keys);
      SortRunAgent<RUN_LENGTH, NUM_RUN_ITEMS-RUN_LENGTH, (NUM_RUN_ITEMS-RUN_LENGTH<=RUN_LENGTH)>::sort_runs(&keys[RUN_LENGTH]);
    }
  };

  template<
    int RUN_LENGTH,
    int NUM_RUN_ITEMS
  >
  struct SortRunAgent<RUN_LENGTH, NUM_RUN_ITEMS, true>
  {
    static __device__ __forceinline__ void sort_runs(KeyT *keys)
    {
      SortingNetwork<KeyT>::sort<NUM_RUN_ITEMS>(keys);
    }
  };

public:

  template<int NUM_ITEMS>
  static __device__ __forceinline__ void sort(KeyT (&keys)[NUM_ITEMS])
  {
    SortingNetworkAgent<NUM_ITEMS, 0>::sort(keys);
  }

  template<int NUM_ITEMS>
  static __device__ __forceinline__ void sort(KeyT *keys)
  {
    SortingNetworkAgent<NUM_ITEMS, 0>::sort(keys);
  }

  template<
    int RUN_LENGTH,
    int NUM_ITEMS
  >
  static __device__ __forceinline__ void sort_runs(KeyT (&keys)[NUM_ITEMS])
  {
    SortRunAgent<RUN_LENGTH, NUM_ITEMS, (NUM_ITEMS>RUN_LENGTH)>::sort_runs(keys);
  }

  template<
    int RUN_LENGTH,
    int NUM_ITEMS
  >
  static __device__ __forceinline__ void sort_runs(KeyT *keys)
  {
    SortRunAgent<RUN_LENGTH, NUM_ITEMS, (NUM_ITEMS>RUN_LENGTH)>::sort_runs(keys);
  }
}
