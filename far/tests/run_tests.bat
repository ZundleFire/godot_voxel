@echo off
REM Host test runner (MSVC) for far/core, which includes no Godot headers.
REM ponytail: no build system, two cl invocations; add one if tests multiply.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if not exist "%TEMP%\voxel_far_tests" mkdir "%TEMP%\voxel_far_tests"
pushd "%TEMP%\voxel_far_tests"
cl /nologo /std:c++17 /EHsc /Zi /Od /fsanitize=address ^
   "%~dp0test_far_core.cpp" "%~dp0..\core\far_column_data.cpp" "%~dp0..\core\far_extract.cpp" "%~dp0..\core\far_mesher.cpp" ^
   /Fe:test_far.exe || goto :fail
cl /nologo /std:c++17 /EHsc /Zi /Od /fsanitize=address ^
   "%~dp0test_far_lod_tree.cpp" "%~dp0..\core\far_column_data.cpp" "%~dp0..\core\far_lod_tree.cpp" "%~dp0..\core\far_sphere.cpp" ^
   /Fe:test_lod.exe || goto :fail
cl /nologo /std:c++17 /EHsc /Zi /Od /fsanitize=address ^
   "%~dp0test_far_sphere.cpp" "%~dp0..\core\far_sphere.cpp" ^
   /Fe:test_sphere.exe || goto :fail
"%TEMP%\voxel_far_tests\test_sphere.exe" || goto :fail
"%TEMP%\voxel_far_tests\test_far.exe" || goto :fail
"%TEMP%\voxel_far_tests\test_lod.exe" || goto :fail
popd
echo ALL TESTS PASSED
exit /b 0
:fail
popd
echo TESTS FAILED
exit /b 1
