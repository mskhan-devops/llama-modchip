param(
    [switch]$Cuda,
    [switch]$Clean,
    [switch]$EnableAvx512,
    [switch]$DisableAvx512,
    [ValidateSet("all", "qwen38", "dsv4", "nemotron")]
    [string]$Target = "all",
    [string]$Arch = "89",
    [string]$SrcRoot = ""
)

# Builds the unified llama-modchip v3.1.0 engine on Windows (MinGW-w64 CPU or MSVC + CUDA).
# Supports Qwen3.8-Flash-Next, DeepSeek-V4-Flash, and Nemotron-3.5-30B with zero-collision custom types:
#   - GGML_TYPE_Q2_0       = 42 (upstream 18 B/64 untouched)
#   - GGML_TYPE_F8_E4M3    = 43 (upstream untouched)
#   - GGML_TYPE_FRACTAL_64 = 44 (30 B/64)
#   - GGML_TYPE_VQ3_64     = 45 (24 B/64)

$ErrorActionPreference = "Stop"

$WindowsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Split-Path -Parent $WindowsDir
if (-not $SrcRoot) { $SrcRoot = Join-Path $RepoRoot "workspace" }

$BaseCommit = "46eb1742db6d2caec1bfa848c5702b1bdcc815e5"
$env:HF_UI_VERSION = "b12604"
$PatchFile  = Join-Path $RepoRoot "patches\llama-modchip-v3.2.patch"
$LlamaDir   = Join-Path $SrcRoot "llama.cpp-src"
$OutDir     = if ($Cuda) { Join-Path $WindowsDir "bin-modchip-cuda" } else { Join-Path $WindowsDir "bin-modchip" }
$BuildDir   = if ($Cuda) { Join-Path $LlamaDir "build-cuda" } else { Join-Path $LlamaDir "build-mingw" }

function Fail([string]$Msg) { Write-Host "[!] $Msg" -ForegroundColor Red; exit 1 }

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  llama-modchip v3.1.0 Unified Windows Engine Build (Qwen3.8 + DeepSeek-V4 + Nemotron)" -ForegroundColor Green
Write-Host ("  Target : {0} | Backend: {1}" -f $Target, $(if ($Cuda) { "CUDA sm_$Arch + CPU (MSVC)" } else { "CPU Native SIMD (MinGW-w64)" })) -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan

$git = (Get-Command git.exe -ErrorAction SilentlyContinue).Source
if (-not $git) { Fail "git.exe not in PATH." }
if (-not (Test-Path $PatchFile)) { Fail "Patch missing: $PatchFile" }

New-Item -ItemType Directory -Path $SrcRoot -Force | Out-Null

if (-not (Test-Path (Join-Path $LlamaDir ".git"))) {
    Write-Host "[*] Cloning llama.cpp ..." -ForegroundColor Cyan
    & $git clone --depth 1 https://github.com/ggml-org/llama.cpp $LlamaDir
    if ($LASTEXITCODE -ne 0) { Fail "git clone failed." }
}
Push-Location $LlamaDir
try {
    $head = (& $git rev-parse HEAD).Trim()
    if ($head -ne $BaseCommit) {
        Write-Host "[*] Fetching pinned commit $BaseCommit ..." -ForegroundColor Cyan
        & $git fetch --depth 1 origin $BaseCommit
        if ($LASTEXITCODE -ne 0) { Fail "git fetch of the pinned commit failed (network?)." }
        & $git checkout -q FETCH_HEAD
    }
    Write-Host "[*] Resetting working tree ..." -ForegroundColor Cyan
    & $git checkout -q -- .
    foreach ($f in "ggml\src\vq3_codebook.h", "common\rc_predictor.h", "common\rc_stockfish.h", "common\rc_trie_v2.h", "common\rc_arbiter_l1.h", "common\rc_routecrack2.h", "tools\server\rc_harp_radix.h") {
        Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $LlamaDir $f)
    }

    Write-Host ("[*] Applying {0} ..." -f (Split-Path -Leaf $PatchFile)) -ForegroundColor Cyan
    & $git apply $PatchFile
    if ($LASTEXITCODE -ne 0) { Fail "Patch does not apply. Expected llama.cpp at $BaseCommit." }

    if ($Clean -and (Test-Path $BuildDir)) { Remove-Item $BuildDir -Recurse -Force }

    $jobs = [math]::Max(1, (Get-CimInstance Win32_Processor | Select-Object -First 1).NumberOfCores)

    if ($Cuda) {
        $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (-not (Test-Path $vswhere)) { Fail "vswhere.exe not found: install Visual Studio Build Tools with the C++ workload." }
        $vsPath = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
        if (-not $vsPath) { Fail "No MSVC C++ toolset found." }
        $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
        if (-not (Test-Path $vcvars)) { Fail "vcvars64.bat not found under $vsPath" }

        $cudaRoot = $env:CUDA_PATH
        if (-not $cudaRoot -or -not (Test-Path (Join-Path $cudaRoot "bin\nvcc.exe"))) {
            $cudaRoot = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
                        Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
        }
        if (-not $cudaRoot) { Fail "CUDA Toolkit not found (winget install Nvidia.CUDA)." }

        # Note: IsProcessorFeaturePresent(49) only tests AVX512F (true on Skylake-X/Cascade Lake which lack VBMI).
        # Default MSVC+CUDA CPU fallback to portable AVX2+FMA+F16C unless -EnableAvx512 is explicitly requested on Zen 4/5 or Ice Lake+.
        $useAvx512 = ($EnableAvx512 -and (-not $DisableAvx512))
        $isaFlags = if ($useAvx512) {
            Write-Host "[*] -EnableAvx512 requested — enabling MSVC AVX-512 + VBMI + VNNI kernels (AMD Zen 4/5 or Intel Ice Lake+)." -ForegroundColor Green
            "-DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"
        } else {
            Write-Host "[*] Building MSVC CPU kernels for AVX2 + FMA + F16C (safe on all modern x86_64 CPUs; pass -EnableAvx512 on Zen 4/5 or Ice Lake+)." -ForegroundColor Yellow
            "-DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"
        }

        $cmd = @"
call "$vcvars" >nul 2>&1
set "PATH=$cudaRoot\bin;%PATH%"
set "HF_UI_VERSION=b12604"
cd /d "$LlamaDir"
cmake -B "$BuildDir" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_NUMBER=12604 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$Arch -DGGML_NATIVE=OFF $isaFlags -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF "-DCMAKE_EXE_LINKER_FLAGS=/STACK:16777216"
if errorlevel 1 exit /b 1
cmake --build "$BuildDir" -j $jobs --target llama-server llama-cli llama-bench llama-perplexity llama-quantize
"@
        $bat = Join-Path $env:TEMP ("mcwin_build_cuda_" + [System.IO.Path]::GetRandomFileName() + ".bat")
        try {
            Set-Content -LiteralPath $bat -Value $cmd -Encoding ASCII
            Write-Host "[*] Building with MSVC + CUDA sm_$Arch ..." -ForegroundColor Cyan
            & cmd /c $bat
            if ($LASTEXITCODE -ne 0) { Fail "CUDA build failed." }
        } finally {
            Remove-Item -LiteralPath $bat -Force -ErrorAction SilentlyContinue
        }
    } else {
        $cmake = (Get-Command cmake.exe -ErrorAction SilentlyContinue).Source
        $ninja = (Get-Command ninja.exe -ErrorAction SilentlyContinue).Source
        $gcc   = (Get-Command g++.exe   -ErrorAction SilentlyContinue).Source
        if (-not $cmake -or -not $ninja -or -not $gcc) { Fail "cmake.exe / ninja.exe / g++.exe not in PATH (install MinGW-w64 WinLibs + CMake + Ninja)." }
        $winFlags = "-D_WIN32_WINNT=0x0A00 -DNTDDI_VERSION=0x0A000000"
        Write-Host "[*] Configuring ..." -ForegroundColor Cyan
        & cmake -B $BuildDir -G Ninja `
            -DCMAKE_BUILD_TYPE=Release `
            -DLLAMA_BUILD_NUMBER=12604 `
            -DGGML_NATIVE=ON `
            -DGGML_OPENMP=OFF `
            -DLLAMA_CURL=OFF `
            -DLLAMA_BUILD_TESTS=OFF `
            -DBUILD_SHARED_LIBS=OFF `
            "-DCMAKE_EXE_LINKER_FLAGS=-static -Wl,--stack,16777216" `
            "-DCMAKE_C_FLAGS=$winFlags" `
            "-DCMAKE_CXX_FLAGS=$winFlags"
        if ($LASTEXITCODE -ne 0) { Fail "cmake configure failed." }
        Write-Host "[*] Building (a few minutes) ..." -ForegroundColor Cyan
        & cmake --build $BuildDir -j $jobs --target llama-server llama-cli llama-bench llama-perplexity llama-quantize
        if ($LASTEXITCODE -ne 0) { Fail "Build failed." }
    }

    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
    foreach ($b in "llama-server.exe", "llama-cli.exe", "llama-bench.exe", "llama-perplexity.exe", "llama-quantize.exe") {
        $p = Join-Path $BuildDir "bin\$b"
        if (Test-Path $p) { Copy-Item $p $OutDir -Force }
    }
    Get-ChildItem (Join-Path $BuildDir "bin") -Filter "*.dll" -ErrorAction SilentlyContinue | Copy-Item -Destination $OutDir -Force

    Write-Host ("[OK] Engine written to {0}" -f $OutDir) -ForegroundColor Green
} finally {
    Pop-Location
}
