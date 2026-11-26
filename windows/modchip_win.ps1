param(
    [string]$Mode = "Menu",
    [int]$Profile = 0,
    [int]$Port = 8080,
    [int]$Ctx = 0,
    [int]$CpuMoe = -1,
    [int]$MoeTopK = 6,
    [double]$MoeMinW = 0.0
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BinDir    = Join-Path $ScriptDir "bin"
$ModelDir  = Join-Path $ScriptDir "models"

if (-not (Test-Path $BinDir))   { New-Item -ItemType Directory -Path $BinDir   | Out-Null }
if (-not (Test-Path $ModelDir)) { New-Item -ItemType Directory -Path $ModelDir | Out-Null }

function Write-Banner {
    Clear-Host
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "  llama-modchip v3.1.0 -- Windows Hybrid GPU + CPU MoE Edition" -ForegroundColor Green
    Write-Host "  Optimized for NVIDIA RTX / AMD Vulkan (8-16 GB VRAM) + CPU (16-128 GB RAM)" -ForegroundColor White
    Write-Host "================================================================================" -ForegroundColor Cyan
}

function Get-HardwareInfo {
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $os  = Get-CimInstance Win32_OperatingSystem
    $cpuName = ($cpu.Name -replace '\s+', ' ').Trim()
    $physCores = [int]$cpu.NumberOfCores
    $logCores  = [int]$cpu.NumberOfLogicalProcessors
    if ($physCores -le 0) { $physCores = 8 }

    $totalRamGB = [math]::Round($os.TotalVisibleMemorySize / 1048576, 1)
    $freeRamGB  = [math]::Round($os.FreePhysicalMemory / 1048576, 1)

    $gpuName = "No NVIDIA GPU detected (Vulkan/CPU fallback)"
    $vramTotalMB = 0
    $vramFreeMB  = 0
    $hasNvidia   = $false

    $nvidiaSmi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi -and (Test-Path "C:\Windows\System32\nvidia-smi.exe")) {
        $nvidiaSmi = "C:\Windows\System32\nvidia-smi.exe"
    }

    if ($nvidiaSmi) {
        try {
            $smiOut = & $nvidiaSmi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1
            if ($smiOut) {
                $parts = $smiOut.Split(",")
                $gpuName     = $parts[0].Trim()
                $vramTotalMB = [int]$parts[1].Trim()
                $vramFreeMB  = [int]$parts[2].Trim()
                $hasNvidia   = $true
            }
        } catch {}
    }

    $driveLetter = (Split-Path -Qualifier $ScriptDir).TrimEnd(':')
    $disk = Get-PSDrive -Name $driveLetter -ErrorAction SilentlyContinue
    $freeDiskGB = if ($disk) { [math]::Round($disk.Free / 1GB, 1) } else { 100.0 }

    return [PSCustomObject]@{
        CpuName     = $cpuName
        PhysCores   = $physCores
        LogCores    = $logCores
        TotalRamGB  = $totalRamGB
        FreeRamGB   = $freeRamGB
        HasNvidia   = $hasNvidia
        GpuName     = $gpuName
        VramTotalMB = $vramTotalMB
        VramFreeMB  = $vramFreeMB
        FreeDiskGB  = $freeDiskGB
    }
}

function Show-HardwareSummary($hw) {
    Write-Host ""
    Write-Host "  [DETECTED HARDWARE PROFILE]" -ForegroundColor Yellow
    Write-Host ("  - CPU    : {0} ({1} physical cores / {2} threads)" -f $hw.CpuName, $hw.PhysCores, $hw.LogCores)
    if ($hw.HasNvidia) {
        $vTot = [math]::Round($hw.VramTotalMB / 1024, 1)
        $vFree = [math]::Round($hw.VramFreeMB / 1024, 1)
        Write-Host ("  - GPU    : {0} ({1} GB VRAM total, {2} GB VRAM free)" -f $hw.GpuName, $vTot, $vFree) -ForegroundColor Green
    } else {
        Write-Host ("  - GPU    : {0}" -f $hw.GpuName) -ForegroundColor Yellow
    }
    $ramColor = if ($hw.FreeRamGB -lt 8.5) { "Yellow" } else { "Green" }
    Write-Host ("  - RAM    : {0} GB total ({1} GB free right now)" -f $hw.TotalRamGB, $hw.FreeRamGB) -ForegroundColor $ramColor
    Write-Host ("  - Disk   : {0} GB free on {1}" -f $hw.FreeDiskGB, (Split-Path -Qualifier $ScriptDir))
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor DarkGray
}

function Invoke-ResumableDownload([string]$Url, [string]$DestPath, [string]$Label) {
    Write-Host ("`n[*] Downloading: {0}" -f $Label) -ForegroundColor Cyan
    Write-Host ("    URL : {0}" -f $Url) -ForegroundColor DarkGray
    Write-Host ("    File: {0}" -f $DestPath) -ForegroundColor DarkGray
    Write-Host "    (If interrupted, re-run the script to resume automatically.)" -ForegroundColor Yellow

    $curlExe = "C:\Windows\System32\curl.exe"
    if (Test-Path $curlExe) {
        & $curlExe -L --fail --retry 5 --retry-delay 3 -C - -o $DestPath $Url
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed via curl.exe (exit code $LASTEXITCODE)."
        }
    } else {
        Invoke-WebRequest -Uri $Url -OutFile $DestPath -UseBasicParsing
    }
}

function Expand-ZipFast([string]$ZipPath, [string]$DestDir) {
    Write-Host ("[*] Extracting archive {0} ..." -f (Split-Path -Leaf $ZipPath)) -ForegroundColor Cyan
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        foreach ($entry in $archive.Entries) {
            if ([string]::IsNullOrEmpty($entry.Name)) { continue }
            $targetFile = Join-Path $DestDir $entry.Name
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $targetFile, $true)
        }
    } finally {
        $archive.Dispose()
    }
}

function Ensure-LlamaBinaries($hw) {
    $serverExe = Join-Path $BinDir "llama-server.exe"
    $cudaDll   = Join-Path $BinDir "ggml-cuda.dll"
    if ((Test-Path $serverExe) -and ((-not $hw.HasNvidia) -or (Test-Path $cudaDll))) {
        return $serverExe
    }

    Write-Host "`n[*] First run: downloading official ggml-org/llama.cpp GPU engine binaries..." -ForegroundColor Green
    $headers = @{ "User-Agent" = "llama-modchip-windows/3.1.0" }
    $releases = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5" -Headers $headers

    $chosenRel = $null
    $binAsset  = $null
    $rtAsset   = $null

    foreach ($rel in $releases) {
        if ($hw.HasNvidia) {
            $b = $rel.assets | Where-Object { $_.name -match "^llama-b\d+-bin-win-cuda-12\.\d+-x64\.zip$" } | Select-Object -First 1
            $r = $rel.assets | Where-Object { $_.name -match "^cudart-llama-bin-win-cuda-12\.\d+-x64\.zip$" } | Select-Object -First 1
            if ($b -and $r) {
                $chosenRel = $rel
                $binAsset  = $b
                $rtAsset   = $r
                break
            }
        } else {
            $b = $rel.assets | Where-Object { $_.name -match "^llama-b\d+-bin-win-vulkan-x64\.zip$" } | Select-Object -First 1
            if ($b) {
                $chosenRel = $rel
                $binAsset  = $b
                break
            }
        }
    }

    if (-not $binAsset) {
        throw "Could not locate Windows binary assets in recent ggml-org/llama.cpp releases."
    }

    Write-Host ("[*] Selected engine release: llama.cpp {0}" -f $chosenRel.tag_name) -ForegroundColor Green

    if ($rtAsset) {
        $rtZip = Join-Path $BinDir $rtAsset.name
        if (-not (Test-Path $rtZip)) {
            Invoke-ResumableDownload -Url $rtAsset.browser_download_url -DestPath $rtZip -Label ("CUDA 12.4 Runtime DLLs ({0} MB)" -f [math]::Round($rtAsset.size / 1MB))
        }
        Expand-ZipFast -ZipPath $rtZip -DestDir $BinDir
    }

    $binZip = Join-Path $BinDir $binAsset.name
    if (-not (Test-Path $binZip)) {
        Invoke-ResumableDownload -Url $binAsset.browser_download_url -DestPath $binZip -Label ("llama.cpp CUDA/GPU Engine ({0} MB)" -f [math]::Round($binAsset.size / 1MB))
    }
    Expand-ZipFast -ZipPath $binZip -DestDir $BinDir

    if (-not (Test-Path $serverExe)) {
        throw "llama-server.exe not found after extraction in $BinDir"
    }
    Write-Host "[OK] GPU engine installed successfully into bin\!" -ForegroundColor Green
    return $serverExe
}

$ModelCatalog = @(
    [PSCustomObject]@{
        Id          = 1
        Title       = "[RECOMMENDED 16GB RAM] Qwen3-30B-A3B (Q3_K_M, 13.7 GiB) -- 30B Sparse MoE"
        Subtitle    = "Hybrid GPU+CPU split: ~9.2 GB in VRAM (11GB+ GPU) + ~4.5 GB in System RAM."
        FileName    = "Qwen3-30B-A3B-Q3_K_M.gguf"
        SizeBytes   = 14711847488
        SizeGB      = 13.7
        Url         = "https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q3_K_M.gguf"
        Kind        = "qwen3_moe"
        TotalMoe    = 48
        MoeLayerMB  = 265
        BaseVramMB  = 2100
        DefaultCtx  = 32768
    },
    [PSCustomObject]@{
        Id          = 2
        Title       = "[LONG-CONTEXT 128K-256K] NVIDIA-Nemotron-3.5-30B-A3B (Q4_0, 17.6 GiB) -- 30B Mamba-2 + MoE"
        Subtitle    = "Hybrid GPU+CPU split: ~9.6 GB in VRAM (11GB+ GPU) + ~8.0 GB in System RAM."
        FileName    = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
        SizeBytes   = 18898091584
        SizeGB      = 17.6
        Url         = "https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/resolve/8cd082170ed313484be1cf44f256a5caa2cfdad3/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
        Kind        = "nemotron_moe"
        TotalMoe    = 23
        MoeLayerMB  = 585
        BaseVramMB  = 4750
        DefaultCtx  = 131072
    },
    [PSCustomObject]@{
        Id          = 3
        Title       = "[100% GPU >=10GB VRAM] Qwen3-14B (Q4_K_M, 8.4 GiB) -- 14B Dense (0 GB System RAM)"
        Subtitle    = "Fits 100% in VRAM on >=10-11 GB GPUs with high throughput and zero RAM pressure."
        FileName    = "Qwen3-14B-Q4_K_M.gguf"
        SizeBytes   = 9001752960
        SizeGB      = 8.4
        Url         = "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf"
        Kind        = "dense_gpu"
        TotalMoe    = 0
        MoeLayerMB  = 0
        BaseVramMB  = 9200
        DefaultCtx  = 32768
    },
    [PSCustomObject]@{
        Id          = 4
        Title       = "[100% GPU TURBO 128K] NVIDIA-Nemotron-H-8B-Reasoning (Q8_0, 8.0 GiB) -- 100+ tok/s!"
        Subtitle    = "Native 8-bit precision, 128K context in ~210 MB KV, 100% VRAM resident on >=10 GB GPUs."
        FileName    = "nvidia_Nemotron-H-8B-Reasoning-128K-Q8_0.gguf"
        SizeBytes   = 8619887584
        SizeGB      = 8.0
        Url         = "https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF/resolve/main/nvidia_Nemotron-H-8B-Reasoning-128K-Q8_0.gguf"
        Kind        = "dense_gpu"
        TotalMoe    = 0
        MoeLayerMB  = 0
        BaseVramMB  = 8600
        DefaultCtx  = 131072
    }
)

function Ensure-ModelFile($modelObj, $hw) {
    $dest = Join-Path $ModelDir $modelObj.FileName
    if (Test-Path $dest) {
        $len = (Get-Item $dest).Length
        if ($len -ge ($modelObj.SizeBytes * 0.98)) {
            Write-Host ("[OK] Model already downloaded: {0} ({1} GiB)" -f $modelObj.FileName, [math]::Round($len / 1GB, 2)) -ForegroundColor Green
            return $dest
        }
    }

    $needGB = [math]::Round($modelObj.SizeGB + 1.0, 1)
    if ($hw.FreeDiskGB -lt $needGB) {
        throw ("Insufficient disk space: need {0} GB free, available {1} GB." -f $needGB, $hw.FreeDiskGB)
    }

    Invoke-ResumableDownload -Url $modelObj.Url -DestPath $dest -Label ("{0} ({1} GiB)" -f $modelObj.FileName, $modelObj.SizeGB)
    return $dest
}

function Compute-OptimalCpuMoe($modelObj, $hw, [int]$ExplicitCpuMoe) {
    if ($ExplicitCpuMoe -ge 0) { return $ExplicitCpuMoe }
    if ($modelObj.TotalMoe -le 0) { return 0 }
    if (-not $hw.HasNvidia) { return $modelObj.TotalMoe }

    # Re-query live free VRAM right before launching
    $freshHw = Get-HardwareInfo
    $freeVram = $freshHw.VramFreeMB
    if ($freeVram -le 0) { $freeVram = $hw.VramTotalMB - 900 }

    # Keep ~550 MB WDDM headroom so Windows never spills VRAM allocations into shared GTT memory
    $safetyMarginMB = 550
    $usableForMoeMB = $freeVram - $modelObj.BaseVramMB - $safetyMarginMB
    if ($usableForMoeMB -lt 0) { $usableForMoeMB = 0 }

    $gpuMoeLayers = [math]::Floor($usableForMoeMB / $modelObj.MoeLayerMB)
    if ($gpuMoeLayers -gt $modelObj.TotalMoe) { $gpuMoeLayers = $modelObj.TotalMoe }
    if ($gpuMoeLayers -lt 0) { $gpuMoeLayers = 0 }

    $cpuMoeLayers = $modelObj.TotalMoe - [int]$gpuMoeLayers
    return $cpuMoeLayers
}

function Run-QuickBenchmark($serverExe, $modelPath, $modelObj, $hw) {
    $benchExe = Join-Path $BinDir "llama-bench.exe"
    if (-not (Test-Path $benchExe)) {
        Write-Host "[!] llama-bench.exe not found in bin\." -ForegroundColor Red
        return
    }
    $nCpuMoe = Compute-OptimalCpuMoe -modelObj $modelObj -hw $hw -ExplicitCpuMoe $CpuMoe
    $benchThreads = [math]::Max(1, $hw.PhysCores - 1)
    Write-Host "`n================================================================================" -ForegroundColor Cyan
    Write-Host "  HARDWARE BENCHMARK (Prefill pp512 + Decode tg128)" -ForegroundColor Yellow
    Write-Host ("  Model    : {0}" -f (Split-Path -Leaf $modelPath))
    Write-Host ("  Split    : -ngl 99 --n-cpu-moe {0} | CPU Threads: -t {1}" -f $nCpuMoe, $benchThreads)
    Write-Host "================================================================================" -ForegroundColor Cyan

    $benchArgs = @(
        "-m", $modelPath,
        "-ngl", "99",
        "-t", "$benchThreads",
        "-fa", "1",
        "-ctk", "q8_0",
        "-ctv", "q4_0",
        "-p", "512",
        "-n", "128",
        "-r", "2"
    )
    if ($nCpuMoe -gt 0) {
        $benchArgs += @("-ncmoe", "$nCpuMoe")
    }
    & $benchExe @benchArgs
}

# ==============================================================================
# MAIN FLOW
# ==============================================================================
Write-Banner
$hw = Get-HardwareInfo
Show-HardwareSummary $hw
$serverExe = Ensure-LlamaBinaries $hw

if ($Mode -eq "Bench") {
    $existing = Get-ChildItem -Path $ModelDir -Filter "*.gguf" | Sort-Object Length -Descending | Select-Object -First 1
    if (-not $existing) {
        Write-Host "`n[*] No model downloaded yet. Fetching recommended profile [1] for benchmark..." -ForegroundColor Yellow
        $modelPath = Ensure-ModelFile -modelObj $ModelCatalog[0] -hw $hw
        $mObj = $ModelCatalog[0]
    } else {
        $modelPath = $existing.FullName
        $mObj = $ModelCatalog | Where-Object { $_.FileName -eq $existing.Name } | Select-Object -First 1
        if (-not $mObj) { $mObj = $ModelCatalog[0] }
    }
    Run-QuickBenchmark -serverExe $serverExe -modelPath $modelPath -modelObj $mObj -hw $hw
    exit 0
}

if ($Profile -le 0) {
    Write-Host "`n  SELECT MODEL PROFILE TO LAUNCH (Press Enter for [1] Recommended):" -ForegroundColor Cyan
    Write-Host ""
    foreach ($m in $ModelCatalog) {
        $installedPath = Join-Path $ModelDir $m.FileName
        $badge = if (Test-Path $installedPath) { "[DOWNLOADED]" } else { "[Auto-Download]" }
        $color = if ($m.Id -eq 1) { "Green" } else { "White" }
        Write-Host ("  [{0}] {1} {2}" -f $m.Id, $m.Title, $badge) -ForegroundColor $color
        Write-Host ("      -> {0}" -f $m.Subtitle) -ForegroundColor DarkGray
        Write-Host ""
    }
    Write-Host "  [5] Launch another .gguf file located in models\" -ForegroundColor White
    Write-Host "  [6] Run hardware speed benchmark (llama-bench) on installed model" -ForegroundColor Yellow
    Write-Host ""
    $ans = Read-Host "  Select [1-6] (Default = 1)"
    if ([string]::IsNullOrWhiteSpace($ans)) { $Profile = 1 } else { $Profile = [int]$ans }
}

if ($Profile -eq 6) {
    $existing = Get-ChildItem -Path $ModelDir -Filter "*.gguf" | Sort-Object Length -Descending | Select-Object -First 1
    if (-not $existing) {
        $modelPath = Ensure-ModelFile -modelObj $ModelCatalog[0] -hw $hw
        $mObj = $ModelCatalog[0]
    } else {
        $modelPath = $existing.FullName
        $mObj = $ModelCatalog | Where-Object { $_.FileName -eq $existing.Name } | Select-Object -First 1
        if (-not $mObj) { $mObj = $ModelCatalog[0] }
    }
    Run-QuickBenchmark -serverExe $serverExe -modelPath $modelPath -modelObj $mObj -hw $hw
    exit 0
}

$selectedModel = $null
$modelPath     = $null

if ($Profile -eq 5) {
    $customFiles = @(Get-ChildItem -Path $ModelDir -Filter "*.gguf")
    if ($customFiles.Count -eq 0) {
        throw "No .gguf files found in $ModelDir. Copy your GGUF there or select [1-4]."
    }
    Write-Host "`n  GGUF files found in models\:" -ForegroundColor Cyan
    for ($i = 0; $i -lt $customFiles.Count; $i++) {
        $sz = [math]::Round($customFiles[$i].Length / 1GB, 2)
        Write-Host ("    [{0}] {1} ({2} GiB)" -f ($i + 1), $customFiles[$i].Name, $sz)
    }
    $cIdx = Read-Host "  Which file would you like to launch? (Default = 1)"
    $pick = if ([string]::IsNullOrWhiteSpace($cIdx)) { 0 } else { ([int]$cIdx) - 1 }
    $modelPath = $customFiles[$pick].FullName
    $fileGB = $customFiles[$pick].Length / 1GB
    $selectedModel = [PSCustomObject]@{
        FileName   = $customFiles[$pick].Name
        TotalMoe   = if ($fileGB -gt 11.0) { 23 } else { 0 }
        MoeLayerMB = 500
        BaseVramMB = 4200
        DefaultCtx = 65536
    }
} else {
    $selectedModel = $ModelCatalog | Where-Object { $_.Id -eq $Profile } | Select-Object -First 1
    if (-not $selectedModel) { $selectedModel = $ModelCatalog[0] }
    $modelPath = Ensure-ModelFile -modelObj $selectedModel -hw $hw
}

$nCpuMoe = Compute-OptimalCpuMoe -modelObj $selectedModel -hw $hw -ExplicitCpuMoe $CpuMoe
$ctxSize = if ($Ctx -gt 0) { $Ctx } else { $selectedModel.DefaultCtx }
$runThreads = [math]::Max(1, $hw.PhysCores - 1)

# Pre-launch free RAM guard for 30B hybrid MoE models on 16 GB Windows systems
$freshHw = Get-HardwareInfo
$estCpuRamGB = [math]::Round(($nCpuMoe * $selectedModel.MoeLayerMB) / 1024.0 + 0.8, 1)
if (($nCpuMoe -gt 0) -and ($freshHw.FreeRamGB -lt ($estCpuRamGB + 0.5))) {
    Write-Host ""
    Write-Host ("  [!] RAM WARNING: {0} CPU MoE layers require ~{1} GB RAM," -f $nCpuMoe, $estCpuRamGB) -ForegroundColor Yellow
    Write-Host ("      but Windows currently has only {0} GB free (of {1} GB total)." -f $freshHw.FreeRamGB, $freshHw.TotalRamGB) -ForegroundColor Yellow
    Write-Host "      Tip: close heavy browsers/Electron apps first to prevent SSD paging." -ForegroundColor Yellow
    if ($ctxSize -gt 65536) {
        $ctxSize = 65536
        Write-Host "      -> Automatically reduced context to 65,536 tokens (64K) to preserve RAM." -ForegroundColor Cyan
    }
    Read-Host "      Press ENTER when ready to launch the engine" | Out-Null
}

$helpOut = & $serverExe --help 2>&1 | Out-String
$loadModeArgs = @()
if ($helpOut -match "--load-mode") {
    $loadModeArgs = @("--load-mode", "none")
} elseif ($helpOut -match "--no-mmap") {
    $loadModeArgs = @("--no-mmap")
}

$specLabel = ""
if ($helpOut -match "routecrack2") {
    $specArgs = @("--spec-type", "routecrack2")
    $specLabel = "RouteCrack 2.0 (832B L1 Micro-Arbiter)"
} else {
    $specArgs = @("--spec-type", "ngram-simple")
    $specLabel = "Draftless ngram-simple (n=3, m=5)"
    if ($helpOut -match "--spec-ngram-simple-size-n") {
        $specArgs += @("--spec-ngram-simple-size-n", "3", "--spec-ngram-simple-size-m", "5")
    } elseif ($helpOut -match "--spec-ngram-size-n" -and $helpOut -notmatch "--spec-ngram-size-n[^\r\n]*removed") {
        $specArgs += @("--spec-ngram-size-n", "3", "--spec-ngram-size-m", "5")
    }
}

$serverArgs = @(
    "-m", $modelPath,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "-c", "$ctxSize",
    "-ngl", "99"
) + $loadModeArgs + @(
    "-ctk", "q8_0",
    "-ctv", "q4_0",
    "-fa", "on",
    "-t", "$runThreads",
    "-tb", "$runThreads",
    "--poll", "0",
    "--poll-batch", "0",
    "-b", "2048",
    "-ub", "512",
    "--cache-reuse", "256"
) + $specArgs + @(
    "--reasoning-format", "deepseek"
)

if ($nCpuMoe -gt 0) {
    $serverArgs += @("--n-cpu-moe", "$nCpuMoe")
}

# Isolate RouteCrack tables: never load Nemotron 131K tables when serving non-Nemotron models
if ($selectedModel.FileName -notmatch '(?i)nemotron') {
    $env:RC_TABLES_DIR = "/nonexistent"
}

# Configure live L2-quadrature MoE control (per-user $LOCALAPPDATA\llama-modchip\dsv4_moe_cfg.bin via DSV4_MOE_CFG_PATH; NEVER export DSV4_MOE_TOPK)
Remove-Item Env:DSV4_MOE_TOPK -ErrorAction SilentlyContinue
if ($MoeTopK -lt 6 -or $MoeMinW -gt 0.0) {
    $env:DSV4_MOE_MIN_W = "$MoeMinW"
    $env:LLAMA_MOE_TOP_K = "$MoeTopK"
}
try {
    $cfgBase = if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA "llama-modchip" } else { Join-Path $ScriptDir "run" }
    New-Item -ItemType Directory -Path $cfgBase -Force | Out-Null
    $cfgPath = Join-Path $cfgBase "dsv4_moe_cfg.bin"
    $env:DSV4_MOE_CFG_PATH = $cfgPath
    $ms = New-Object System.IO.MemoryStream
    $bw = New-Object System.IO.BinaryWriter($ms)
    $bw.Write([int32]$MoeTopK)
    $bw.Write([int32]$MoeTopK)
    $bw.Write([int32]([math]::Round($MoeMinW * 10000)))
    $bw.Write([int32]3)
    $bw.Write([int32]43)
    $bw.Write([int32]1) # L2 quadrature norm-preserving rescaling
    $bw.Write([int32]0)
    $bw.Write([int32]0)
    $bw.Flush()
    $bytes = $ms.ToArray()
    $bw.Close()
    # In-place FileShare::ReadWrite update so live updates succeed even while llama-server.exe holds MapViewOfFile
    $fs = New-Object System.IO.FileStream($cfgPath, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::ReadWrite)
    $fs.Write($bytes, 0, $bytes.Length)
    $fs.Flush()
    $fs.Close()
} catch {}

$gpuMoeCount = $selectedModel.TotalMoe - $nCpuMoe
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "  LAUNCHING HYBRID GPU + CPU ENGINE (llama-modchip v3.1.0 Windows)" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ("  - Active Model       : {0}" -f $selectedModel.FileName) -ForegroundColor White
Write-Host ("  - Active Context     : {0} tokens (KV Cache Q8_0/Q4_0 + Flash-Attn)" -f $ctxSize) -ForegroundColor White
if ($selectedModel.TotalMoe -gt 0) {
    $lmLabel = if ($loadModeArgs.Count -gt 0) { $loadModeArgs -join " " } else { "mmap" }
    Write-Host ("  - Tensor-Role Split  : Backbone + {0}/{1} MoE layers on GPU ({2})" -f $gpuMoeCount, $selectedModel.TotalMoe, $hw.GpuName) -ForegroundColor Cyan
    Write-Host ("                         {0}/{1} MoE layers in CPU RAM (~{2} GB via {3})" -f $nCpuMoe, $selectedModel.TotalMoe, $estCpuRamGB, $lmLabel) -ForegroundColor Cyan
} else {
    Write-Host ("  - GPU Offload        : 100% on {0} (-ngl 99, 0 GB System RAM)" -f $hw.GpuName) -ForegroundColor Cyan
}
Write-Host ("  - CPU Threads        : -t {0} -tb {0} --poll 0 --poll-batch 0 (non-convoy physical cores)" -f $runThreads) -ForegroundColor White
Write-Host ("  - Speculative Engine : {0}" -f $specLabel) -ForegroundColor White
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host ("  -> Web Chat UI       : http://127.0.0.1:{0}  (Opens in browser automatically once ready)" -f $Port) -ForegroundColor Yellow
Write-Host ("  -> Local OpenAI API  : http://127.0.0.1:{0}/v1 (For VS Code Continue / Cline / Cursor)" -f $Port) -ForegroundColor Yellow
Write-Host "  (Press CTRL+C in this window to stop the server)" -ForegroundColor DarkGray
Write-Host "================================================================================`n" -ForegroundColor Green

# Background helper that waits for /health and opens the local web chat
$healthUrl = "http://127.0.0.1:$Port/health"
$chatUrl   = "http://127.0.0.1:$Port"
$null = Start-Job -ScriptBlock {
    param($hUrl, $cUrl)
    for ($i = 0; $i -lt 180; $i++) {
        Start-Sleep -Seconds 1
        try {
            $r = Invoke-WebRequest -Uri $hUrl -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
            if ($r.StatusCode -eq 200) {
                Start-Process $cUrl
                break
            }
        } catch {}
    }
} -ArgumentList $healthUrl, $chatUrl

& $serverExe @serverArgs
