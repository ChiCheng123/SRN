# NMS

This module is from https://github.com/huaifeng1993/NMS, modified for SRN+torch1.4.0.

Huge thanks to https://github.com/yhenon/pytorch-retinanet/issues/66#issuecomment-486671704.

## FAQ

### Cannot locate `nvcc`, `include` or `lib64`

1. Make sure you have both installed `CUDA` and `nvcc`. Please search for how to install `CUDA` and `nvcc`, or verify the installation.
1. Change the code in `locate_cuda()` in `setup3.py` according to you case. NOTE: `nvcc` should be located in `$CUDA/bin/nvcc`, `include` should be `$CUDA/include`, `lib64` should be `$CUDA/lib64`.