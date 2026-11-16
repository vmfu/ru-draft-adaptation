@echo off
call G:\VS2022BT\VC\Auxiliary\Build\vcvars64.bat
set CUDA_PATH=G:\Program Files\CUDA\v12.8
set PATH=%CUDA_PATH%\bin;%PATH%
cd /d F:\Projects\lm-studio-serv\_ru_draft\llama.cpp-src
F:\Projects\lm-studio-serv\_ru_draft\_tools\cmake-4.1.2-windows-x86_64\bin\cmake.exe -B build_local -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON ^
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF ^
  -DLLAMA_CURL=OFF
if errorlevel 1 exit /b 1
ninja -C build_local llama-imatrix
exit /b %ERRORLEVEL%
