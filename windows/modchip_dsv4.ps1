param(
    [string]$Model   = "",
    [string]$Server  = "",
    [int]$Port       = 8080,
    [int]$Ctx        = 32768,
    [int]$Threads    = 0,
    [string]$KvType = "q8_0",
    [int]$MoeTopK    = 6,
    [double]$MoeMinW = 0.0,
    [int]$NGpuLayers = 0,
    [int]$NCpuMoe    = -1,
    [int]$NPredict   = -1,
    [string]$SpecType = "routecrack2",
    [switch]$Fast,
    [switch]$NoSpec,
    [switch]$Thinking,
    [switch]$NoBrowser,
    [switch]$OpenBrowser,
    [switch]$Hybrid,
    [switch]$Bench
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
$ModelDir  = Join-Path $ScriptDir "models"
$BinDir    = Join-Path $ScriptDir "bin"
$ModBinDir = Join-Path $ScriptDir "bin-modchip"
$CudaBinDir = Join-Path $ScriptDir "bin-modchip-cuda"

function Write-Head([string]$Text) {
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Green
    Write-Host "================================================================================" -ForegroundColor Cyan
}

function Find-Server {
    if ($Server -and (Test-Path $Server)) { return (Resolve-Path $Server).Path }
    $candidates = if ($Hybrid) {
        @(
            (Join-Path $CudaBinDir "llama-server.exe"),
            (Join-Path $ModBinDir "llama-server.exe")
        )
    } else {
        @(
            (Join-Path $ModBinDir "llama-server.exe"),
            (Join-Path $CudaBinDir "llama-server.exe"),
            (Join-Path $BinDir "llama-server.exe")
        )
    }
    foreach ($c in $candidates) {
        if (Test-Path $c) { return (Resolve-Path $c).Path }
    }
    throw "llama-server.exe not found. Put the patched build into $ModBinDir or run: powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1"
}

function Find-Model {
    if ($Model) {
        if (-not (Test-Path $Model)) { throw "Model file not found: $Model" }
        return (Resolve-Path $Model).Path
    }
    if (-not (Test-Path $ModelDir)) { throw "Model folder not found: $ModelDir" }
    # Prefer safety-aligned split GGUFs first (MaxQ-v4 -> MaxQ-Fractal -> VQ3 -> Fractal), then opt-in Abliterated
    $shard1Files = @(Get-ChildItem -LiteralPath $ModelDir -Filter "*.gguf" -File |
                     Where-Object { $_.Name -match "-00001-of-\d+\.gguf$" })
    foreach ($pat in @("Qwen3.8", "MaxQ-v4", "MaxQ-Fractal", "VQ3", "Fractal", "Abliterated")) {
        $match = $shard1Files | Where-Object {
            if ($pat -eq "Abliterated") { $_.Name -match "Abliterated" }
            else { ($_.Name -notmatch "Abliterated") -and ($_.Name -match $pat) }
        } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($match) { return $match.FullName }
    }
    $shard1 = $shard1Files | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($shard1) { return $shard1.FullName }
    $any = Get-ChildItem -LiteralPath $ModelDir -Filter "*.gguf" -File | Sort-Object Length -Descending | Select-Object -First 1
    if ($any) { return $any.FullName }
    throw "No .gguf model found in $ModelDir"
}

function Get-Hw {
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $os  = Get-CimInstance Win32_OperatingSystem
    $ramTotalGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    $ramFreeGB  = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
    $gpuName = "none"
    $vramFreeMB = 0
    $smi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
    if ($smi) {
        try {
            $line = & $smi.Source --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1
            if ($line) {
                $p = $line.Split(",")
                $gpuName = $p[0].Trim()
                $vramFreeMB = [int]$p[2].Trim()
            }
        } catch {}
    }
    [PSCustomObject]@{
        CpuName    = ($cpu.Name -replace '\s+', ' ').Trim()
        PhysCores  = [int]$cpu.NumberOfCores
        LogCores   = [int]$cpu.NumberOfLogicalProcessors
        RamTotalGB = $ramTotalGB
        RamFreeGB  = $ramFreeGB
        GpuName    = $gpuName
        VramFreeMB = $vramFreeMB
    }
}

# Writes the 32-byte live MoE tunable block that the patched ggml-cpu reads
# from per-user DSV4_MOE_CFG_PATH ($LOCALAPPDATA\llama-modchip\dsv4_moe_cfg.bin; int32 x 8: topk, topk, minw*10000, 3, 43, l2, 0, 0)
function Set-MoeConfig([int]$TopK, [double]$MinW) {
    try {
        $bytes = New-Object byte[] 32
        [BitConverter]::GetBytes([int32]$TopK).CopyTo($bytes, 0)
        [BitConverter]::GetBytes([int32]$TopK).CopyTo($bytes, 4)
        [BitConverter]::GetBytes([int32][math]::Round($MinW * 10000)).CopyTo($bytes, 8)
        [BitConverter]::GetBytes([int32]3).CopyTo($bytes, 12)
        [BitConverter]::GetBytes([int32]43).CopyTo($bytes, 16)
        [BitConverter]::GetBytes([int32]1).CopyTo($bytes, 20)
        [BitConverter]::GetBytes([int32]0).CopyTo($bytes, 24)
        [BitConverter]::GetBytes([int32]0).CopyTo($bytes, 28)
        $cfgBase = if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA "llama-modchip" } else { Join-Path $ScriptDir "run" }
        New-Item -ItemType Directory -Path $cfgBase -Force | Out-Null
        $path = Join-Path $cfgBase "dsv4_moe_cfg.bin"
        $fs = New-Object System.IO.FileStream($path, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::ReadWrite)
        try {
            # Write 32 bytes in-place without SetLength(0) so live updates succeed while llama-server.exe holds MapViewOfFile
            $fs.Write($bytes, 0, 32)
            $fs.Flush()
        } finally { $fs.Close() }
        return $path
    } catch { return $null }
}

$hw = Get-Hw
if ($Hybrid -and $NGpuLayers -le 0) { $NGpuLayers = 99 }
if ($Fast -and $MoeTopK -eq 6) { $MoeTopK = 4 }

$exe = Find-Server
$modelPath = Find-Model
$modelInfo = Get-Item -LiteralPath $modelPath
$modelGB = [math]::Round($modelInfo.Length / 1GB, 2)
$patched = $exe -like "*bin-modchip*" -or $exe -like "*build-mingw*"

$isNemotron = $modelInfo.Name -match '(?i)nemotron'
$isQwen38   = $modelInfo.Name -match '(?i)qwen3\.?8'
$isQwen32GB = $isQwen38 -and ($modelInfo.Name -match '(?i)32gb')
$modelAlias = if ($isQwen38) { "qwen3.8-flash-next" } elseif ($isNemotron) { "nemotron-3.5-30b-a3b" } else { "deepseek-v4-flash" }
$kvTypeV    = if ($isNemotron -and $KvType -eq "q8_0") { "q4_0" } else { $KvType }
if (-not $isNemotron) { $env:RC_TABLES_DIR = "/nonexistent" }

Write-Head ("llama-modchip v3.1.0 -- {0} on Windows" -f $(if ($isQwen38) { "Qwen3.8-Flash-Next MaxQ" } elseif ($isNemotron) { "NVIDIA-Nemotron-3.5-30B-A3B" } else { "DeepSeek-V4-Flash 284B" }))

if ($Threads -le 0) { $Threads = [math]::Max(1, $hw.PhysCores - 1) }

# Total size and shard count scoped strictly to the selected model prefix (handles >=10 shards safely)
$totalBytes = $modelInfo.Length
$shardCount = 1
if ($modelInfo.Name -match "^(.+)-\d{5}-of-(\d{5})\.gguf$") {
    $modelPrefix = $Matches[1]
    $n = [int]$Matches[2]
    $totalBytes = 0
    $shardCount = 0
    for ($i = 1; $i -le $n; $i++) {
        $f = Join-Path $ModelDir ("{0}-{1:D5}-of-{2:D5}.gguf" -f $modelPrefix, $i, $n)
        if (Test-Path $f) {
            $totalBytes += (Get-Item $f).Length
            $shardCount++
        }
    }
}
$totalGB = [math]::Round($totalBytes / 1GB, 2)

Write-Host ("  Engine    : {0}" -f $exe)
Write-Host ("  Patched   : {0}" -f $(if ($patched) { "yes (FRACTAL_64=44 + VQ3_64=45 + RouteCrack 2.0 + MoE pruning)" } else { "NO - stock build cannot read FRACTAL_64 (44) / VQ3_64 (45) tensors" }))
Write-Host ("  Model     : {0}" -f $modelInfo.Name)
Write-Host ("  Model size: {0} GB total ({1} shards)" -f $totalGB, $shardCount)
Write-Host ("  CPU       : {0} ({1} physical cores / {2} threads)" -f $hw.CpuName, $hw.PhysCores, $hw.LogCores)
Write-Host ("  GPU       : {0} ({1} MB VRAM free)" -f $hw.GpuName, $hw.VramFreeMB)
Write-Host ("  RAM       : {0} GB total, {1} GB free" -f $hw.RamTotalGB, $hw.RamFreeGB)

if (-not $patched) {
    Write-Host ""
    Write-Host "  [!] This is the stock llama.cpp build: it cannot read FRACTAL_64 (44) or VQ3_64 (45) tensors." -ForegroundColor Yellow
    Write-Host "      Run: powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1" -ForegroundColor Yellow
    Write-Host "      or pass -Server <path to patched llama-server.exe>." -ForegroundColor Yellow
}

# RAM guard: the weights are memory mapped, but KV cache + compute buffers are real allocations
$needFree = [math]::Round($totalGB * 0.06 + 2.0, 1)
if ($hw.RamFreeGB -lt $needFree) {
    Write-Host ""
    Write-Host ("  [!] Low free RAM: {0} GB free, keep at least ~{1} GB for KV cache and compute buffers." -f $hw.RamFreeGB, $needFree) -ForegroundColor Yellow
    Write-Host "      Close browsers/Electron apps first. Windows will page the model out to SSD otherwise." -ForegroundColor Yellow
}

$cfgPath = Set-MoeConfig -TopK $MoeTopK -MinW $MoeMinW
if ($cfgPath) { $env:DSV4_MOE_CFG_PATH = $cfgPath }

# Modchip env knobs consumed by the patched engine.
# NOTE: DSV4_MOE_TOPK / LLAMA_MOE_TOP_K must NOT be exported: during model load the patch
# derives the ffn_gate_tid2eid tensor shape from that value, and this GGUF is built for Top-K 6.
# Any other value aborts the load ("expected 4, 129280, got 6, 129280"). Top-K pruning is applied
# live through per-user $env:DSV4_MOE_CFG_PATH instead, which the compute path reads on every graph.
$env:LLAMA_KV_L3_WINDOW = "2048"
$env:LLAMA_ATTN_WINDOW   = "0"      # 0 = full exact attention (SWA off)
$env:LLAMA_ATTN_SINKS    = "16"
Remove-Item Env:DSV4_MOE_TOPK -ErrorAction SilentlyContinue
Remove-Item Env:DSV4_MOE_MIN_W -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_MOE_TOP_K -ErrorAction SilentlyContinue

$ubatchSize = if ($isQwen32GB) { "512" } else { "2048" }
$srvArgs = @(
    "-m", $modelPath,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--alias", $modelAlias,
    "-c", "$Ctx",
    "-np", "1",
    "-t", "$Threads",
    "-tb", "$Threads",
    "--poll", "0",
    "--poll-batch", "0",
    "--repack",
    "-b", "2048",
    "-ub", $ubatchSize,
    "-fa", "on",
    "--cache-type-k", $KvType,
    "--cache-type-v", $kvTypeV,
    "--cache-reuse", "256",
    "--jinja",
    "--metrics",
    "--reasoning-format", "deepseek"
)
if ($isQwen38) {
    $srvArgs += @("--ctx-checkpoints", $(if ($isQwen32GB) { "0" } else { "1" }))
    if ($isQwen32GB) {
        $srvArgs += @("--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--presence-penalty", "0.2", "--repeat-penalty", "1.05")
    }
} elseif ($isNemotron) {
    $srvArgs += @("--ctx-checkpoints", "64", "--checkpoint-min-step", "128")
} else {
    # Anti-loop sampling for ultra-low-bpw 284B builds.
    # Do NOT set --dry-sequence-breaker none: with no breakers DRY penalises tokens that repeat
    # inside the chat template itself and the output turns to nonsense.
    $srvArgs += @(
        "--repeat-penalty", "1.1",
        "--repeat-last-n", "256",
        "--dry-multiplier", "0.8",
        "--dry-base", "1.75",
        "--dry-allowed-length", "3",
        "--dry-penalty-last-n", "1024"
    )
}
if ($Thinking) { $srvArgs += @("--reasoning-budget", "-1") } else { $srvArgs += @("--reasoning-budget", "0") }
if ($NPredict -ge 0) { $srvArgs += @("-n", "$NPredict") }

if (-not $NoSpec -and $SpecType -ne "none") {
    if ($SpecType -eq "ngram-simple") {
        $srvArgs += @(
            "--spec-type", "ngram-simple",
            "--spec-ngram-simple-size-n", "4",
            "--spec-ngram-simple-size-m", "7",
            "--spec-ngram-simple-min-hits", "1"
        )
    } else {
        $srvArgs += @("--spec-type", $SpecType)
    }
}

# Hybrid GPU + CPU split.
# FRACTAL_64 (44) and VQ3_64 (45) tensors execute on CPU kernels on Windows/CUDA; pin those
# tensors to CPU explicitly when offloading standard ggml layers to GPU.
$totalLayers = 43          # deepseek4.block_count of DeepSeek-V4-Flash
$moeBytesPerLayerMB = 1390 # gate_exps + up_exps + down_exps of one layer ~1.39 GB
$gpuDenseMB = 1800         # embeddings + output + dense/attention parts of all layers
if ($NGpuLayers -gt 0) {
    if ($NCpuMoe -lt 0) {
        $usable = $hw.VramFreeMB - 800
        $moeOnGpu = [math]::Max(0, [math]::Floor(($usable - $gpuDenseMB) / $moeBytesPerLayerMB))
        if ($moeOnGpu -gt $totalLayers) { $moeOnGpu = $totalLayers }
        $NCpuMoe = $totalLayers - [int]$moeOnGpu
    }
    $srvArgs += @("-ngl", "$NGpuLayers", "--n-cpu-moe", "$NCpuMoe")
    $srvArgs += @("-ot", "blk\.\d+\.(attn_.*|indexer.*|ffn_(gate|up|down)_shexp)\.weight=CPU")
}

if ($Bench) {
    $benchExe = Join-Path (Split-Path -Parent $exe) "llama-bench.exe"
    if (-not (Test-Path $benchExe)) { throw "llama-bench.exe not found next to $exe" }
    Write-Head "llama-bench: prefill pp512 + decode tg128"
    & $benchExe -m $modelPath -t $Threads -fa 1 -ctk $KvType -ctv $kvTypeV -p 512 -n 128 -r 2
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host ("  Context   : {0} tokens (KV {1}/{2}, flash-attn, -ub {3})" -f $Ctx, $KvType, $kvTypeV, $ubatchSize)
if ($isQwen38) {
    Write-Host "  MoE Top-K : 10 of 512 routed experts + 1 shared expert"
} elseif ($isNemotron) {
    Write-Host "  MoE Top-K : 6 of 128 routed experts + 1 shared expert (double-width)"
} else {
    Write-Host ("  MoE Top-K : {0} of 6 routed experts (min weight {1})" -f $MoeTopK, $MoeMinW)
}
$specBanner = if ($NoSpec -or $SpecType -eq "none") {
    "off"
} elseif ($SpecType -eq "routecrack2") {
    "RouteCrack 2.0 (832B L1 Micro-Arbiter)"
} elseif ($SpecType -eq "ngram-simple") {
    "ngram-simple (n=4, m=7)"
} else {
    $SpecType
}
Write-Host ("  Spec      : {0}" -f $specBanner)
if ($cfgPath -and -not $isQwen38 -and -not $isNemotron) { Write-Host ("  Live cfg  : {0}" -f $cfgPath) }
Write-Host ("  Offload   : {0}" -f $(if ($NGpuLayers -gt 0) { "-ngl $NGpuLayers, --n-cpu-moe $NCpuMoe (FRACTAL_64/VQ3_64 pinned to CPU)" } else { "CPU only" }))
if ($NGpuLayers -gt 0 -and -not $isQwen38) {
    Write-Host "  [!] Measured on this machine: hybrid is SLOWER for this model (2.7-3.4 tok/s vs 7.4-8.1" -ForegroundColor Yellow
    Write-Host "      CPU-only). FRACTAL_64/VQ3_64 tensors run on CPU, so every layer round-trips PCIe." -ForegroundColor Yellow
}
Write-Host ""
Write-Host ("  -> Web chat UI : http://127.0.0.1:{0}" -f $Port) -ForegroundColor Yellow
Write-Host ("  -> OpenAI API  : http://127.0.0.1:{0}/v1  (model: {1})" -f $Port, $modelAlias) -ForegroundColor Yellow
Write-Host "  Press CTRL+C in this window to stop the server." -ForegroundColor DarkGray
Write-Host ""

# Browser stays closed unless -OpenBrowser is passed
if ($OpenBrowser -and -not $NoBrowser) {
    $null = Start-Job -ScriptBlock {
        param($url)
        for ($i = 0; $i -lt 900; $i++) {
            Start-Sleep -Seconds 2
            try {
                $r = Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
                if ($r.StatusCode -eq 200) { Start-Process $url; break }
            } catch {}
        }
    } -ArgumentList "http://127.0.0.1:$Port"
}

# Native stderr would be turned into a terminating error by ErrorActionPreference=Stop
# under Windows PowerShell 5.1, so launch through Start-Process and inherit the console.
function Quote-Arg([string]$a) {
    if ($a -match '[\s"]') { '"' + ($a -replace '"', '\"') + '"' } else { $a }
}
$argLine = ($srvArgs | ForEach-Object { Quote-Arg $_ }) -join ' '
$proc = Start-Process -FilePath $exe -ArgumentList $argLine -NoNewWindow -Wait -PassThru
exit $proc.ExitCode


