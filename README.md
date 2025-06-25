
# Guide for Using a PyTorch Model on QNX RTOS Based on ONNX Runtime

This repository describes the steps to run a pre-trained PyTorch model on QNX Real-Time OS using a port of ONNX Runtime. ONNX Runtime is a tool that allows running inference of models created on any ML framework. We are not limited to PyTorch only!

This example utilizes custom segmentation model!

## Installation

1. Install Ubuntu >= 22.04.6 LTS via WSL (Windows Subsystem for Linux)
2. Install QNX SDP 7.1 on Ubuntu

### Custom Build of ONNX Runtime for QNX SDP 7.1
3. Download the [repo](https://github.com/jeffswt/onnxruntime-qnx/tree/master) and follow the instructions
    a) execute the installation commands and specify the path to qnx710  
    b) create a folder ort_qnx and specify it  
    c) in `build_qnx.sh` change inside the function `_build_boost_1_80_0_qnx710()`
    ```bash
    wget https://boostorg.jfrog.io/artifactory/main/release/1.80.0/source/boost_1_80_0.tar.gz -O $__boost_archive
    ```
    to the following line
    ```bash
    wget https://sourceforge.net/projects/boost/files/boost/1.80.0/boost_1_80_0.tar.gz/download -O $__boost_archive
    ```
4. Now you can run
```shell
./build_qnx.sh
```

## Cross-Compiling inference.cpp for AARCH64le Architecture

By executing the following command we compile the script and get an exe file that can be run on a QNX system

```shell
qcc -Vgcc_ntoaarch64le -o src/inference src/inference.cpp src/lodepng.cpp -I/home/user/ort_qnx/onnxruntime/include -L/home/user/ort_qnx/onnxruntime/cmake/build -lonnxruntime -Wl,-rpath,/home/user/ort_qnx/onnxruntime/cmake/build
```

Note: specify the correct path to the ONNX header for correct compilation
```cpp
#include "~/onnxruntime/include/onnxruntime/core/session/onnxruntime_cxx_api.h" // onnx runtime
```

## Compilation for Linux

Before compiling, install ONNX Runtime C++ for Linux x64 (CPU) and export the path. Example compilation:

```shell
export LD_LIBRARY_PATH=/mnt/c/Users/user/Downloads/onnxruntime-linux-x64-1.22.0/lib
```

```shell
g++ -o src/inference src/inference.cpp src/lodepng.cpp -I/mnt/c/Users/user/Downloads/onnxruntime-linux-x64-1.22.0/include     -L/mnt/c/Users/user/Downloads/onnxruntime-linux-x64-1.22.0/lib     -lonnxruntime  -Wl,-rpath,/mnt/c/Users/Downloads/onnxruntime-linux-x64-1.22.0/lib
```

Example usage: <input_folder> <model_path> [threshold], where default threshold = 0.5f

```shell
./src/inference ./processed model_187_0.9326.onnx 0.6
```

Note: model will be looked in ./models/

## Copying Required Libraries to SD Card

1. `~/qnx710/target/qnx7/aarch64le/usr/lib/`
    - libc++.so.1
    - libstdc++.so.6
    - libiconv.so.1
2. `~/qnx710/target/qnx7/aarch64le/lib`
    - libm.so.3.sym
    - libc.so.5.sym
    - libgcc_s.so.1
    - libcatalog.so.1
3. `~/ort_qnx/onnxruntime/cmake/build`
    - libonnxruntime.so.1.14.1 or
    - libonnxruntime.so

Additionally copy the resulting .exe file and .ONNX model to the SD card.

## Running on QNX

### Execute commands in the terminal via serial port

1. `mount -t dos /dev/hd0t12 /mnt`
2. `cp -i *.so.* /tmp && cp -i *.onnx /tmp && cp -i inference /tmp`
    `cp -i /mnt/*.so.* /mnt/*.onnx /mnt/inference /tmp`
    `cp -Mqnx -rv /processed /tmp`
3. `export LD_LIBRARY_PATH=/tmp`
4. `cd /tmp`
5. `chmod +x inference`
6. `./inference`

## For the future: do cross-compilation with transfer over SSH

Success!
