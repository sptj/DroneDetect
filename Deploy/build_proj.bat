del build -y
mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="%~dp0libtorch-win-shared-with-deps-cu100/libtorch;%~dp0opencv344/opencv/build" ..