# Builds the desktop preset tree for llama-modchip v3.1.0:
#   Desktop\Modchip Presets\<ctx>\fast|exact.lnk
# Re-run any time: powershell -ExecutionPolicy Bypass -File windows\make_shortcuts.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PresetDir = Join-Path $ScriptDir "presets"
$Desktop   = [Environment]::GetFolderPath("Desktop")
$Root      = Join-Path $Desktop "Modchip Presets"
$ServerExe = Join-Path $ScriptDir "bin-modchip\llama-server.exe"

New-Item -ItemType Directory -Path $PresetDir -Force | Out-Null
New-Item -ItemType Directory -Path $Root -Force | Out-Null

$contexts = @(32768, 65536, 131072, 262144, 524288, 1048576)

function Write-PresetBat([string]$file, [string]$ctx, [string]$extra, [string]$title) {
@"
@echo off
chcp 65001 >nul
title $title
cd /d "%~dp0.."

taskkill /IM llama-server.exe /F >nul 2>&1
ping -n 3 127.0.0.1 >nul

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\modchip_dsv4.ps1" -Ctx $ctx $extra
if errorlevel 1 (
    echo.
    echo [!] Server exited with code %errorlevel% - see the log above.
)
echo.
echo Window closes in 5 seconds ...
ping -n 6 127.0.0.1 >nul
"@ | Set-Content -LiteralPath (Join-Path $PresetDir $file) -Encoding ASCII
}

# stop helper
@"
@echo off
chcp 65001 >nul
title Stop llama-modchip server
taskkill /IM llama-server.exe /F >nul 2>&1
if errorlevel 1 (echo Server was not running.) else (echo Server stopped.)
ping -n 3 127.0.0.1 >nul
"@ | Set-Content -LiteralPath (Join-Path $PresetDir "STOP_server.bat") -Encoding ASCII

$wsh = New-Object -ComObject WScript.Shell

function New-Link([string]$lnk, [string]$target, [string]$desc) {
    $sc = $wsh.CreateShortcut($lnk)
    $sc.TargetPath = $target
    $sc.WorkingDirectory = $PresetDir
    if (Test-Path $ServerExe) { $sc.IconLocation = "$ServerExe,0" }
    $sc.Description = $desc
    $sc.Save()
}

$made = @()
foreach ($ctx in $contexts) {
    $label = if ($ctx -ge 1048576) { "1M" } else { "{0}K" -f ($ctx / 1024) }
    $sub = Join-Path $Root $label
    New-Item -ItemType Directory -Path $sub -Force | Out-Null

    # fast preset: pruned MoE Top-4 (for DeepSeek-V4) + RouteCrack 2.0
    Write-PresetBat -file ("ctx{0}_fast.bat" -f $ctx) -ctx "$ctx" -extra "-Fast" -title "llama-modchip $label fast"
    New-Link -lnk (Join-Path $sub "Fast (Top-4 + RouteCrack 2.0).lnk") -target (Join-Path $PresetDir ("ctx{0}_fast.bat" -f $ctx)) -desc "ctx $ctx, MoE Top-K 4 + RouteCrack 2.0"

    # exact preset: unpruned model experts + RouteCrack 2.0
    Write-PresetBat -file ("ctx{0}_exact.bat" -f $ctx) -ctx "$ctx" -extra "" -title "llama-modchip $label exact"
    New-Link -lnk (Join-Path $sub "Exact (Unpruned + RouteCrack 2.0).lnk") -target (Join-Path $PresetDir ("ctx{0}_exact.bat" -f $ctx)) -desc "ctx $ctx, unpruned MoE + RouteCrack 2.0"

    $made += (Join-Path $sub "Fast (Top-4 + RouteCrack 2.0).lnk")
    $made += (Join-Path $sub "Exact (Unpruned + RouteCrack 2.0).lnk")
}

New-Link -lnk (Join-Path $Root "STOP llama-modchip server.lnk") -target (Join-Path $PresetDir "STOP_server.bat") -desc "Stops llama-server.exe"
$made += (Join-Path $Root "STOP llama-modchip server.lnk")

$readme = @"
llama-modchip v3.1.0 -- Windows Desktop Presets (Qwen3.8 / DeepSeek-V4 MaxQ-v4 / Nemotron)

Folders correspond to context window length:
  32K / 64K / 128K / 256K / 512K / 1M

Each folder provides two launchers using RouteCrack 2.0 (--spec-type routecrack2):
  Fast  -- Applies Top-4 L2-quadrature expert pruning on DeepSeek-V4 for higher throughput.
  Exact -- Keeps all native routed experts unpruned for maximum output fidelity.

Usage: double-click any shortcut to launch the server at http://127.0.0.1:8080 (OpenAI /v1 API).
Only one preset runs at a time; launching a new shortcut stops any running llama-server.exe first.
Use "STOP llama-modchip server" to shut down the active engine cleanly.
"@
Set-Content -LiteralPath (Join-Path $Root "README.txt") -Value $readme -Encoding UTF8

Write-Host "Preset tree created:" -ForegroundColor Green
Get-ChildItem $Root -Recurse | Where-Object { $_.Extension -in '.lnk', '.txt' } | ForEach-Object { Write-Host ("  " + $_.FullName.Replace($Root, "")) }
