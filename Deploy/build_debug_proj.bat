del build -y
mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="%~dp0libtorch-win-shared-with-deps-debug-nightly/libtorch;%~dp0opencv344/opencv/build" ..