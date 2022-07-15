rm -rf build
mkdir build
cd build

# Make (or rather a Makefile) is a buildsystem - it drives the compiler and other build tools to build your code. 
# CMake is a generator of buildsystems
cmake ..
# nproc - print the number of processing units available
make -j$(nproc)

./helloSLAM
./useHello