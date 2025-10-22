cmake_minimum_required(VERSION 3.18)
project(cuda_mlp_binary LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

add_executable(mlp_cuda src/mlp_cuda.cu)
set_target_properties(mlp_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
