# ___2019 - 08 - 12 Caffe___
***

# Install python caffe
## links
  - [Caffe Installation](http://caffe.berkeleyvision.org/installation.html)
## git clone
  ```sh
  git clone https://github.com/BVLC/caffe --depth=10
  cd caffe/
  git pull --depth=100000
  ```
## Makefile.config
  ```sh
  $ diff Makefile.config Makefile.config.example
  5c5
  < USE_CUDNN := 1
  ---
  > # USE_CUDNN := 1
  39,40c39,40
  < CUDA_ARCH := # -gencode arch=compute_20,code=sm_20 \
  < 		# -gencode arch=compute_20,code=sm_21 \
  ---
  > CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
  > 		-gencode arch=compute_20,code=sm_21 \
  53c53
  < BLAS := open
  ---
  > BLAS := atlas
  71,72c71,72
  < # PYTHON_INCLUDE := /usr/include/python2.7 \
  < # 		/usr/lib/python2.7/dist-packages/numpy/core/include
  ---
  > PYTHON_INCLUDE := /usr/include/python2.7 \
  > 		/usr/lib/python2.7/dist-packages/numpy/core/include
  75,78c75,78
  < ANACONDA_HOME := /opt/anaconda3
  < PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
  < 		$(ANACONDA_HOME)/include/python3.7m \
  < 		$(ANACONDA_HOME)/lib/python3.7/site-packages/numpy/core/include
  ---
  > # ANACONDA_HOME := $(HOME)/anaconda
  > # PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
  > 		# $(ANACONDA_HOME)/include/python2.7 \
  > 		# $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include
  97,98c97,98
  < INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
  < LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial
  ---
  > INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
  > LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
  ```
## Makefile
  ```sh
  $ git diff Makefile
  diff --git a/Makefile b/Makefile
  index b7660e85..9331d8aa 100644
  --- a/Makefile
  +++ b/Makefile
  @@ -198,14 +198,14 @@ ifeq ($(USE_HDF5), 1)
          LIBRARIES += hdf5_hl hdf5
   endif
   ifeq ($(USE_OPENCV), 1)
  -       LIBRARIES += opencv_core opencv_highgui opencv_imgproc
  +       LIBRARIES += opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs

          ifeq ($(OPENCV_VERSION), 3)
  -               LIBRARIES += opencv_imgcodecs
  +               LIBRARIES += opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs
          endif

   endif
  -PYTHON_LIBRARIES ?= boost_python python2.7
  +PYTHON_LIBRARIES ?= boost_python-py36
   WARNINGS := -Wall -Wno-sign-compare

   ##############################
  ```
## apt install
  ```sh
  sudo apt install \
          protobuf-c-compiler protobuf-compiler libprotobuf-dev libboost-all-dev \
          libgflags-dev libgoogle-glog-dev libleveldb-dev liblmdb-dev libopencv-dev \
          libsnappy-dev libhdf5-serial-dev libopenblas-dev libatlas-base-dev
  ```
## make
  ```sh
  cp Makefile.config.example Makefile.config

  make all
  make test
  make runtest
  make pytest  # --> caffe/python/caffe/_caffe.so
  export PYTHONPATH=/home/leondgarse/workspace/caffe/python:$PYTHONPATH
  ```
## Q / A
  - **Q: Import error: undefined symbol**
    ```py
    import caffe
    undefined symbol: _ZN5boost6python6detail11init_moduleER11PyModuleDefPFvvE
    ```
    A: 原因是 boost_python 的版本不匹配，Makefile 中指定的是 2.7，应使用 python 3.6
    ```sh
    # ls /usr/lib/x86_64-linux-gnu/libboost_python* -l
    # /usr/lib/x86_64-linux-gnu/libboost_python-py36.so -> libboost_python3-py36.so

    # vi Makefile +208
    PYTHON_LIBRARIES ?= boost_python-py36
    ```
    重新编译 so 文件
    ```sh
    make pytest
    ```
  - **Q: Build error: undefined reference to cv::imread**
    ```sh
    build_release/lib/libcaffe.so: undefined reference to cv::imread(cv::String const&, int)
    ```
    A: 在Makefile中添加代码：
    ```sh
    # vi Makefile +201
    ifeq ($(USE_OPENCV), 2)
        LIBRARIES += opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs
    endif  
    ```
  - **Q: include error: hdf5.h: No such file or directory**
    ```sh
    ./include/caffe/util/hdf5.hpp:6:18: fatal error: hdf5.h: No such file or directory
    ```
    A: PYTHON_INCLUDE / PYTHON_LIB 中指定的路径不对
    ```sh
    # vi Makefile.config +97
    < INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
    < LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial
    ```
  - **Q: Build error: Unsupported gpu architecture 'compute_20'**
    ```sh
    NVCC src/caffe/layers/reduction_layer.cu
    nvcc fatal   : Unsupported gpu architecture 'compute_20'
    Makefile:588: recipe for target '.build_release/cuda/src/caffe/layers/reduction_layer.o' failed
    make: *** [.build_release/cuda/src/caffe/layers/reduction_layer.o] Error 1
    ```
    A: 修改 CUDA architecture setting
    ```sh
    # vi Makefile.config +39
    CUDA_ARCH := # -gencode arch=compute_20,code=sm_20 \
    		# -gencode arch=compute_20,code=sm_21 \
    ```
  - **Q: build error: undefined reference to caffe::cudnn::dataType<float>::one**
    ```sh
    .build_release/lib/libcaffe.so: undefined reference to caffe::cudnn::dataType<float>::one
    collect2: error: ld returned 1 exit status
    make: *** [.build_release/tools/upgrade_net_proto_text.bin] Error 1
    ```
    A: 使用 openblas 替换 atlas blas
    ```sh
    # vi Makefile.config +53
    BLAS := open
    ```
