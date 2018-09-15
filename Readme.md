Для построения Docker образа со всеми необходимым библиотеками и построения программы, необходимо вызвать: \
docker build -t videoreader .  Для запуска программы вызывать: \
docker run videoreader

Чтобы построить ffmpeg + CUVID под Windows нужно:
1) Иметь msys2
2) Скачать nv-codec-headers
3) Добавить pkg-config из msys2 в PATH
4) Добавить путь до nv-codec-headers .pc файл в $PKG_CONFIG_PATH (PKG_CONFIG_PATH=path)
5) Следовать инструкции: https://stackoverflow.com/questions/41358478/is-it-possible-to-build-ffmpeg-x64-on-windows
Но с измененным конфигом:
./configure --toolchain=msvc --arch=x86_64 --enable-yasm --enable-asm --enable-shared --enable-w32threads --disable-programs --disable-doc --disable-static --prefix=/d/ffmpeg/ --enable-cuda --enable-cuvid --enable-nvenc --extra-cflags=-I/D:/Soft/v9.0/include --extra-ldflags=-LIBPATH:/D:/Soft/v9.0/lib/x64
6) make, make install
7) В случае ошибок конфигурации чекать на наличие Warning ffbuild/log файл в папке с ffmpeg бинарниками

Pytorch build: https://github.com/pytorch/pytorch
set "VS150COMNTOOLS=C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build"
set CMAKE_GENERATOR=Visual Studio 15 2017 Win64
set DISTUTILS_USE_SDK=1
REM The following two lines are needed for Python 2.7, but the support for it is very experimental.
set MSSdk=1
set FORCE_PY27_BUILD=1
REM As for CUDA 8, VS2015 Update 3 is also required to build PyTorch. Use the following two lines.
set "PREBUILD_COMMAND=%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat"
set PREBUILD_COMMAND_ARGS=x64

call "%VS150COMNTOOLS%\vcvarsall.bat" x64 -vcvars_ver=14.11
python setup.py install