/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Split into multiple files to compile in paralell

#include "selective_scan/selective_scan_bwd_kernel.cuh"


template void selective_scan_bwd_cuda<at::Half, float, at::Half>(SSMParamsBwd & params, cudaStream_t stream);

template void selective_scan_bwd_cuda<at::Half, float, float>(SSMParamsBwd & params, cudaStream_t stream);
