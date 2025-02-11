@echo off
setlocal enabledelayedexpansion

REM =====================================================
REM Change to the script’s directory
REM =====================================================
cd /D "%~dp0"

REM Ensure system32 is in PATH
set PATH=%PATH%;%SystemRoot%\system32

REM =====================================================
REM Normalize current directory by stripping any double quotes.
REM =====================================================
set "CURRDIR=%CD:"=%"
echo Debug: Normalized Current Directory is [%CURRDIR%]

REM =====================================================
REM Check for spaces in the installation path.
REM =====================================================
set "NO_SPACES=%CURRDIR: =%"
if not "%CURRDIR%"=="%NO_SPACES%" (
    echo ERROR: This script requires an installation path with no spaces.
    goto end
)

REM =====================================================
REM Warn if special characters are present in the path.
REM =====================================================
set "SPCHARMESSAGE=WARNING: Special characters detected in the installation path! This may cause installation issues."
echo %CURRDIR% | findstr /R /C:"[!#\$%&()*+,;<=>?@\[\]^`{|}~]" >nul && (
    call :PrintBigMessage %SPCHARMESSAGE%
)
set SPCHARMESSAGE=

REM =====================================================
REM Set temporary directories (used during install/setup)
REM =====================================================
set TMP=%CURRDIR%\installer_files
set TEMP=%CURRDIR%\installer_files

REM =====================================================
REM Deactivate any active Conda environments to avoid conflicts.
REM =====================================================
(call conda deactivate && call conda deactivate && call conda deactivate) 2>nul

REM =====================================================
REM Configuration Variables.
REM =====================================================
set INSTALL_DIR=%CURRDIR%\installer_files
set CONDA_ROOT_PREFIX=%CURRDIR%\installer_files\conda
set INSTALL_ENV_DIR=%CURRDIR%\installer_files\env
set MINICONDA_DOWNLOAD_URL=https://repo.anaconda.com/miniconda/Miniconda3-py311_24.11.1-0-Windows-x86_64.exe
set MINICONDA_CHECKSUM=43dcbcc315ff91edf959e002cd2f1ede38c64b999fefcc951bccf2ed69c9e8bb
set conda_exists=F

REM =====================================================
REM Check if Conda is already installed.
REM =====================================================
call "%CONDA_ROOT_PREFIX%\_conda.exe" --version >nul 2>&1
if "%ERRORLEVEL%" EQU "0" set conda_exists=T

REM =====================================================
REM If not, download and install Miniconda.
REM =====================================================
if "%conda_exists%"=="F" (
    echo Downloading Miniconda from %MINICONDA_DOWNLOAD_URL% to %INSTALL_DIR%\miniconda_installer.exe
    mkdir "%INSTALL_DIR%"
    call curl -Lk "%MINICONDA_DOWNLOAD_URL%" > "%INSTALL_DIR%\miniconda_installer.exe" || (
        echo Miniconda download failed.
        goto end
    )
    REM Verify installer checksum
    for /f %%a in ('CertUtil -hashfile "%INSTALL_DIR%\miniconda_installer.exe" SHA256 ^| find /i /v " " ^| find /i "%MINICONDA_CHECKSUM%"') do (
        set "output=%%a"
    )
    if not defined output (
        for /f %%a in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "if((Get-FileHash \"%INSTALL_DIR%\miniconda_installer.exe\" -Algorithm SHA256).Hash -eq ''%MINICONDA_CHECKSUM%''){echo true}"') do (
            set "output=%%a"
        )
    )
    if not defined output (
        echo Checksum verification failed.
        del "%INSTALL_DIR%\miniconda_installer.exe"
        goto end
    ) else (
        echo Checksum verification passed.
    )
    echo Installing Miniconda to %CONDA_ROOT_PREFIX%
    start /wait "" "%INSTALL_DIR%\miniconda_installer.exe" /InstallationType=JustMe /NoShortcuts=1 /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=%CONDA_ROOT_PREFIX%
    call "%CONDA_ROOT_PREFIX%\_conda.exe" --version || (
        echo Miniconda installation failed.
        goto end
    )
    del "%INSTALL_DIR%\miniconda_installer.exe"
)

REM =====================================================
REM Create the Conda environment for SpongeQuant if needed.
REM =====================================================
if not exist "%INSTALL_ENV_DIR%" (
    echo Creating Conda environment in %INSTALL_ENV_DIR%
    call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y --prefix "%INSTALL_ENV_DIR%" python=3.11 || (
        echo Conda environment creation failed.
        goto end
    )
)
if not exist "%INSTALL_ENV_DIR%\python.exe" (
    echo Conda environment is empty.
    goto end
)

REM =====================================================
REM Set Environment Isolation Variables.
REM =====================================================
set PYTHONNOUSERSITE=1
set PYTHONPATH=
set PYTHONHOME=
set "CUDA_PATH=%INSTALL_ENV_DIR%"
set "CUDA_HOME=%CUDA_PATH%"

REM =====================================================
REM Activate the Conda Environment.
REM =====================================================
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || (
    echo Failed to activate Conda environment.
    goto end
)

REM =====================================================
REM Check for a suitable CMake.
REM Priority: Check portable version first; if not, check PATH.
REM =====================================================
set "USE_CMAKE="
if exist "%INSTALL_DIR%\cmake\bin\cmake.exe" (
    echo [INFO] Found portable CMake in installer_files.
    set "USE_CMAKE=%INSTALL_DIR%\cmake\bin\cmake.exe"
) else (
    for /f "delims=" %%V in ('where cmake 2^>nul') do (
       set "TEMP_CMAKE=%%V"
       goto :gotCMAKE
    )
    :gotCMAKE
    if defined TEMP_CMAKE (
       for /f "tokens=2 delims= " %%A in ('"%TEMP_CMAKE% --version"') do set "CMAKE_VER=%%A"
       echo [DEBUG] Found CMake version: %CMAKE_VER%
       echo %CMAKE_VER% | findstr /c:"3.31.5" >nul
       if not errorlevel 1 (
           echo [INFO] Found acceptable CMake in PATH.
           set "USE_CMAKE=%TEMP_CMAKE%"
       )
    )
)
if not defined USE_CMAKE (
    echo [INFO] CMake not found or version not acceptable. Attempting to download portable CMake...
    set "CMAKE_VERSION=3.31.5"
    set "CMAKE_URL=https://github.com/Kitware/CMake/releases/download/v3.31.5/cmake-3.31.5-windows-x86_64.zip"
    set "CMAKE_ZIP=%INSTALL_DIR%\cmake.zip"
    if exist "!CMAKE_ZIP!" del /F /Q "!CMAKE_ZIP!"
    echo [DEBUG] CMAKE_URL=!CMAKE_URL!
    echo [DEBUG] CMAKE_ZIP=!CMAKE_ZIP!
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { (New-Object System.Net.WebClient).DownloadFile(\"!CMAKE_URL!\",\"!CMAKE_ZIP!\"); Write-Host 'Download succeeded.' } catch { Write-Host 'Download failed: ' + $_.Exception.Message }"
    timeout /t 5 >nul
    echo [DEBUG] Directory listing for installer_files:
    dir "%INSTALL_DIR%"
    if not exist "!CMAKE_ZIP!" (
        echo [ERROR] Failed to download CMake.
        goto end
    )
    set "CMAKE_DIR=%INSTALL_DIR%\cmake"
    mkdir "%CMAKE_DIR%"

    REM -- Use native tar to extract the ZIP archive --
    echo [DEBUG] Executing: tar -xf "!CMAKE_ZIP!" -C "%INSTALL_DIR%\cmake"
    tar -xf "!CMAKE_ZIP!" -C "%INSTALL_DIR%\cmake" 2>&1
    echo [DEBUG] tar exit code: %ERRORLEVEL%

    REM -- If tar extraction failed, fallback to PowerShell Expand-Archive --
    if errorlevel 1 (
         echo [WARNING] tar extraction failed. Trying PowerShell Expand-Archive...
         powershell -NoProfile -Command "Expand-Archive -LiteralPath '!CMAKE_ZIP!' -DestinationPath '%INSTALL_DIR%\cmake' -Force"
         echo [DEBUG] Listing contents of "%INSTALL_DIR%\cmake" after Expand-Archive:
         dir "%INSTALL_DIR%\cmake" /s
    )
    REM -- Instead of scanning recursively, check the expected location:
    set "EXPECTED_CMAKE=%INSTALL_DIR%\cmake\cmake-3.31.5-windows-x86_64\bin\cmake.exe"
    echo [DEBUG] Checking for expected cmake.exe at: !EXPECTED_CMAKE!
    if exist "!EXPECTED_CMAKE!" (
         set "USE_CMAKE=!EXPECTED_CMAKE!"
         echo [DEBUG] Found cmake.exe at: !USE_CMAKE!
    ) else (
         echo [ERROR] Extraction failed: expected cmake.exe not found at !EXPECTED_CMAKE!.
         goto end
    )
    echo [INFO] Downloaded portable CMake and updated PATH.
)
echo [DEBUG] Using CMake: !USE_CMAKE!
if defined USE_CMAKE (
    set "PATH=%~dp0installer_files\cmake\bin;%PATH%"
)

REM =====================================================
REM Determine GPU availability flag.
REM =====================================================
set GPU_FOUND=0
where nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    set GPU_FOUND=1
)

REM =====================================================
REM Install Python dependencies based on GPU availability.
REM =====================================================
where nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    echo [INFO] NVIDIA GPU detected: installing GPU CUDA dependencies...
    call pip install -r src\requirements.gpu-cuda.txt
) else (
    echo [INFO] No NVIDIA GPU detected: installing CPU-only dependencies...
    call pip install -r src\requirements.cpu.txt
)

REM =====================================================
REM Clone and build external repositories.
REM -----------------------------------------------------
REM Build llama.cpp (always needed) using CMake (default generator)
REM =====================================================
if not exist "llama_cpp" (
    echo [INFO] Cloning llama.cpp repository...
    git clone https://github.com/ggerganov/llama.cpp.git llama_cpp
)
pushd llama_cpp
if not exist "build" (
    echo [INFO] Building llama.cpp binaries using CMake...
    mkdir build
    pushd build
    "%USE_CMAKE%" .. 
    "%USE_CMAKE%" --build . --config Release
    popd
) else (
    echo [INFO] llama.cpp binaries already built.
)
popd

REM =====================================================
REM Clone and install AutoAWQ (only if GPU detected)
REM =====================================================
if "%GPU_FOUND%"=="1" (
    if not exist "AutoAWQ" (
        echo [INFO] Cloning AutoAWQ repository...
        git clone https://github.com/casper-hansen/AutoAWQ.git AutoAWQ
        pushd AutoAWQ
        git checkout v0.2.4
        call pip install -e .
        popd
    ) else (
        echo [INFO] AutoAWQ repository already present.
    )
) else (
    echo [INFO] No GPU detected; skipping AutoAWQ installation.
)

REM =====================================================
REM Clone and install exllamav2 (only if GPU detected)
REM =====================================================
if "%GPU_FOUND%"=="1" (
    if not exist "exllamav2" (
        echo [INFO] Cloning exllamav2 repository...
        git clone https://github.com/turboderp-org/exllamav2.git exllamav2
        pushd exllamav2
        call pip install -e .
        popd
    ) else (
        echo [INFO] exllamav2 repository already present.
    )
) else (
    echo [INFO] No GPU detected; skipping exllamav2 installation.
)

REM =====================================================
REM Run the SpongeQuant Application.
REM (Assuming the main script is at src\app.py)
REM =====================================================
call python src\app.py %*

:end
pause
exit /b

:PrintBigMessage
echo.
echo *******************************************************************
for %%M in (%*) do echo * %%~M
echo *******************************************************************
echo.
exit /b