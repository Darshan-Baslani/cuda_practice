#pragma once

#include "kernels/1_naive.cuh"
#include "kernels/2_coalesced.cuh"
#include "kernels/3_shared_mem.cuh"
#include "kernels/4_1d_block_tiling.cuh"
#include "kernels/5_2d_block_tiling.cuh"
#include "kernels/6_vectorized_loads.cuh"
#include "kernels/7_warp_tiling.cuh"