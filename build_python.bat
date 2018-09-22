set "VS150COMNTOOLS=D:\Soft\VisualStudio2017\VC\Auxiliary\Build"
call "%VS150COMNTOOLS%\vcvarsall.bat" x64 -vcvars_ver=14.11
set path=%path%;C:\Users\Home\Desktop\VideoReader\external\ffmpeg\bin
python setup.py install
python Test.py
pause