# Install OpenCV 2.4.9

## Extract

There is already a zip file containing OpenCV 2.4.9 in `Thirdpart`. Extract/unzip it to the Thirdpart directory.

```bash
unzip opencv-2.4.9.zip
```

## Patch

The following changes are required:

1. In `opencv-2.4.9/CMakeLists.txt`, comment line 77 so that it reads:
```
# include(cmake/OpenCVDetectCXXCompiler.cmake)
```

2. In `opencv-2.4.9/modules/contrib/src/chamfermatching.cpp`, comment or delete lines 969, 972, 1016, 1019, 1111 and 1130. 

## Build and Install

Minimum build, in `Release` mode, using `libjpeg` and `libpng` system libraries:
```bash
cd opencv-2.4.9
mkdir build && cd build
cmake   -DWITH_CUDA=off -DWITH_OPENCL=off -DWITH_OPENGL=off \
        -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/opt/opencv-2.4.9 \
        -DBUILD_JPEG=off -DBUILD_JASPER=off -DBUILD_OPENEXR=off \
        -DBUILD_PNG=off -DBUILD_TIFF=off -DBUILD_ZLIB=off \
        -DWITH_FFMPEG=off \
        -DCMAKE_CXX_FLAGS="-std=c++11" \
        -DCMAKE_CXX_COMPILER=/usr/bin/g++-8 \
        -DCMAKE_C_COMPILER=/usr/bin/gcc-8 \
        -DENABLE_PRECOMPILED_HEADERS=off \
        ..
make -j
sudo make install
```

Note that `make -j` may fail if you have insufficient RAM + swapspace. If this happens, try building with less cores (e.g. `make -j2` for 2 cores), increase swapspace, or increase RAM allowance (e.g. if using a virtual machine).
