@echo off
REM Host test runner (MSVC) for gpu_driven/core, which includes no Godot headers.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if not exist "%TEMP%\voxel_gpu_driven_tests" mkdir "%TEMP%\voxel_gpu_driven_tests"
pushd "%TEMP%\voxel_gpu_driven_tests"
cl /nologo /std:c++17 /EHsc /Zi /Od /W4 /fsanitize=address ^
   "%~dp0test_gpu_driven_core.cpp" "%~dp0..\core\gpu_vertex_pack.cpp" "%~dp0..\core\gpu_range_allocator.cpp" ^
   /Fe:test_gpu_driven.exe || goto :fail
"%TEMP%\voxel_gpu_driven_tests\test_gpu_driven.exe" || goto :fail
popd
echo ALL TESTS PASSED
exit /b 0
:fail
popd
echo TESTS FAILED
exit /b 1
