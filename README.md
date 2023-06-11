# Map2DFusion
------------------------------------------------------------------------------

![](./map2dfusion.gif)

## Information about Map2DFusion
This is an open-source implementation of paper:
Map2DFusion: Real-time Incremental UAV Image Mosaicing based on Monocular SLAM.

Website : http://zhaoyong.adv-ci.com/map2dfusion/

Video   : https://www.youtube.com/watch?v=-kSTDvGZ-YQ

PDF     : http://zhaoyong.adv-ci.com/Data/map2dfusion/map2dfusion.pdf   

If you use this project for research, please cite our paper:

```
@CONFERENCE{zhaoyong2016Map2DFusion, 
	author={S. {Bu} and Y. {Zhao} and G. {Wan} and Z. {Liu}}, 
	booktitle={2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
	title={Map2DFusion: Real-time incremental UAV image mosaicing based on monocular SLAM}, 
	year={2016}, 
	volume={}, 
	number={}, 
	pages={4564-4571}, 
	doi={10.1109/IROS.2016.7759672}, 
	ISSN={2153-0866}, 
	month={Oct}
}
```

___
## Build

This has been adapted from [git@gitee.com:pi-lab/Map2DFusion.git](git@gitee.com:pi-lab/Map2DFusion.git) for Ubuntu 20.04.

### Dependencies

Note that some of these packages may already be installed, which can be checked for with `apt search <package_name>`.

If we want to compile with CUDA later on, then CUDA_PATH needs to be defined.

#### Build tools

```bash
sudo apt install build-essential g++ cmake git
```

#### OpenGL tools
```bash
sudo apt install freeglut3-dev libxmu-dev libxi-dev
```

#### Qt4
```bash
sudo add-apt-repository ppa:rock-core/qt4
sudo apt install libqt4-dev libqt4-opengl-dev libqt4-sql-sqlite
```

Note for Ubuntu 18.04, no need to add the repository.

#### POCO
```bash
sudo apt install libpoco-dev
```

#### C++/C compilers (others may work but untested)
```bash
sudo apt install gcc-8 g++-8
```

#### OpenCV 2.4.9
See the OpenCV Install Instructions at [Thirdpart/opencv_install.md](Thirdpart/opencv_install.md).


### Map2DFusion

Currently Map2DFusion must be build in `Debug` mode:

```bash
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/usr/bin/g++-8 \
      -DCMAKE_C_COMPILER=/usr/bin/gcc-8 \
      -DOpenCV_DIR=/opt/opencv-2.4.9/share/OpenCV \
      -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-g" -DCMAKE_C_FLAGS="-g" \
      ..
make -j
```

Note that you may get the following linker warning due to the use of the old `lua-5.1.5` library, which can be ignored:
```
/usr/bin/ld: CMakeFiles/map2dfusion.dir/Thirdpart/lua-5.1.5/src/loslib.c.o: in function `os_tmpname':
/home/william/src/Map2DFusion/Thirdpart/lua-5.1.5/src/loslib.c:60: warning: the use of `tmpnam' is dangerous, better use `mkstemp'
```

___
## Usage (Dynamic Object Removal)

### Setup

In the file `Default.cfg` inside the root directory of Map2DFusion the parameter `Map2D.Type` has to be set to `5`.

The following parameters can be configured:

|  Parameter              |                                                                      Description                                                                           |
|  :---                   |                                                                  :-----------------                                                                        |
|  Map2D.Type             |  **Type 0:** Map2D, **Type 1:** Map2DCPU, **Type 2:** Map2DGPU, **Type 3:** MultiBandMap2DCPU, **Type 4:** Map2DRender, **Type 5:** MultiBandMap2DCPUSem.  |
|  Map2D.SemDir           |  Semantic segmentation directory. In the case of the proposed datasets can be `sem` or `sem_dilated`.                                                      |
|  Map2D.SemLabels        |  Location of the `labels.txt` file.                                                                                                                        |
|  Map2D.ShowSemBlending  |  Flag for changing between the visualization of the semantic segmentation orthomosaic creation or the RGB orthomosaic creation.                            |

### Download Datasets

Download from [https://polybox.ethz.ch/index.php/s/7GgeCEqXuemsaBb](https://polybox.ethz.ch/index.php/s/7GgeCEqXuemsaBb).

The remaining instructions will assume they are downloaded in `~/data`

### Run

The `map2dfusion` executable will be in your `build` folder after compilation.

```bash
cd build
./map2dfusion conf=../Default.cfg DataPath=$HOME/data/phantom3-village-kfs
```

___
## Usage (Original)

### Setup

In the file `Default.cfg` inside the root directory of Map2DFusion the parameter `Map2D.Type` has to be set to `3`.

### Download Datasets

Download from [https://drive.google.com/file/d/1qKEDOre4XzhPGiec3bmuKEZh5cieqlLZ/view](https://drive.google.com/file/d/1qKEDOre4XzhPGiec3bmuKEZh5cieqlLZ/view) and [https://github.com/zdzhaoyong/phantom3-village-kfs](https://github.com/zdzhaoyong/phantom3-village-kfs).

The remaining instructions will assume they are downloaded in `~/data`

### Run

The `map2dfusion` executable will be in your `build` folder after compilation.

```bash
cd build
./map2dfusion conf=../Default.cfg DataPath=$HOME/data/phantom3-village-kfs
```
