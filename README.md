GPU-TopK
========

GPU-TopK implements efficient top-k runtimes for GPUs. The specific problem solved is given a array of entries (key-only or key-value), find the top-k entries based on value of key. 

The package implements the following routines:

* Bitonic Top-K: reduction algorithm based on bitonic sort
* Radix Select Top-K: reduction of radix sort to compute top-k
* Sort Top-K: sorts the entire array and selects the top-k entries

For full details of the algorithms, see our [paper](http://anilshanbhag.in/static/papers/gputopk_sigmod18.pdf)

```
@inproceedings{shanbhag2018efficient,
  title={Efficient Top-K query processing on massively parallel hardware},
  author={Shanbhag, Anil and Pirk, Holger and Madden, Samuel},
  booktitle={Proceedings of the 2018 International Conference on Management of Data},
  pages={1557--1570},
  year={2018},
  organization={ACM}
}
```

Usage
----

The individual implementations can used directly as standalone header files. For example, to use `RadixSelectTopK`, 

```
#include "radixSelectTopK.cuh"
...
float* d_keys_in; // device pointer to the array
uint num_items;   // number of entries in the array
uint k;           // 
float* d_keys_out;// device pointer to the result array (needs to be of size atleast k)
CachingDeviceAllocator&  g_allocator // Cub memory allocator 
radixSelectTopK<float>(d_keys_in, num_items, k, d_keys_out, g_allocator);
```

We have implemented a testutil called `compareTopKAlgorithms` that can be used to benchmark the different algorithms. The testutil lets you test performance of the algorithms across standard data types and certain pre-defined distributions. To run the testutil:

```
# Edit Makefile to select the right Gencode for your GPU
# For example: for V100 GPU set GENCODE_FLAGS to use GENCODE_SM70

make compareTopKAlgorithms
./compareTopKAlgorithms
```

Here is an example tracelog:
```
$ ./compareTopKAlgorithms
Please enter the type of value you want to test:
1-float
2-double
3-uint
1
Please enter distribution type: 0
Please enter K: 32
Please enter number of tests to run per K: 3
Please enter start power (dataset size starts at 2^start)(max val: 29): 29
Please enter stop power (dataset size stops at 2^stop)(max val: 29): 29
NOW STARTING A NEW K

The distribution is: UNIFORM FLOATS
Running test 1 of 3 for size: 536870912 and k: 32
TESTING: 0 Sort
TESTING: 2 Bitonic TopK
TESTING: 1 Radix Select
Running test 2 of 3 for size: 536870912 and k: 32
TESTING: 2 Bitonic TopK
TESTING: 0 Sort
TESTING: 1 Radix Select
Running test 3 of 3 for size: 536870912 and k: 32
TESTING: 0 Sort
TESTING: 1 Radix Select
TESTING: 2 Bitonic TopK


Sort                 averaged: 219.273071 ms
Radix Select         averaged: 132.391724 ms
Bitonic TopK         averaged: 134.959854 ms
Sort                 minimum: 215.583801 ms
Radix Select         minimum: 63.751999 ms
Bitonic TopK         minimum: 28.718592 ms
Sort won 0 times
Radix Select won 1 times
Bitonic TopK won 2 times
```

For benchmarking, it is advisable to run the suite more than once inorder to have GPU warmed up. To see the full set of distributions implemented, check test/generateProblems.cuh. 

Known Issues
-----------
1. Currently works for key-only, we will add key-value soon
2. Works for data set size upto 2^29. This is due to inherent limitations of maximum array size on GPUs.
3. Tested to work well on Nvidia Maxwell architecture and upwards (may not work on K80 GPU - you are welcome to submit a patch). 
4. BitonicTopK works only for K<=256, if you are testing K > 256, make sure to comment BitonicTopK
