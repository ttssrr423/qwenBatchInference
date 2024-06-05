rm -r ./build
mkdir build
cd build
cmake ..  -DCUTLASS_NVCC_ARCHS="70;75;80" -DMY_PYTHON_VERSION=3.9 
make -j 2
cd package && python setup.py install
cd ..
cd ..
