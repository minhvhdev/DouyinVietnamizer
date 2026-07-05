# Prepare or refresh the Windows portable runtime (GGUF TTS + tools + manifest).
# Default target: vendor/portable-runtime (used by tauri:dev and copied into dist-portable).
param(
    [string]$RuntimeRoot = "",
    [switch]$SkipModelDownload,
    [switch]$SkipVenvSync
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot\.."
if (-not $RuntimeRoot) {
    $RuntimeRoot = Join-Path $repoRoot "vendor\portable-runtime"
} else {
    $RuntimeRoot = Resolve-Path $RuntimeRoot
}

$runtimeBackend = Join-Path $RuntimeRoot "backend"
$runtimeVenv = Join-Path $RuntimeRoot ".venv"
$runtimePython = Join-Path $runtimeVenv "Scripts\python.exe"
$runtimeModels = Join-Path $RuntimeRoot "models"
$runtimeTools = Join-Path $RuntimeRoot "tools"
$voxcpmModels = Join-Path $runtimeModels "voxcpm2"
$voxcpmTools = Join-Path $runtimeTools "voxcpm2"
$voxcpmSource = Join-Path $repoRoot "vendor\voxcpm2"
$srcBackend = Join-Path $repoRoot "backend"

function Require-Path([string]$Path, [string]$Message) {
    if (-not (Test-Path $Path)) {
        throw $Message
    }
}

Write-Host "Preparing portable runtime at $RuntimeRoot"

Require-Path $runtimePython "Missing $runtimePython. Bootstrap vendor/portable-runtime once (embedded Python + uv venv)."
Require-Path (Join-Path $runtimeTools "ffmpeg\ffmpeg.exe") "Missing FFmpeg under $runtimeTools\ffmpeg."

New-Item -ItemType Directory -Force -Path $runtimeBackend, $voxcpmModels, $voxcpmTools | Out-Null

Write-Host "Syncing backend sources..."
robocopy (Join-Path $srcBackend "dv_backend") (Join-Path $runtimeBackend "dv_backend") /MIR /NJH /NJS /NDL /NFL /XD __pycache__ .venv-voxcpm | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy dv_backend failed with exit code $LASTEXITCODE" }
robocopy (Join-Path $srcBackend "scripts") (Join-Path $runtimeBackend "scripts") /MIR /NJH /NJS /NDL /NFL | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy scripts failed with exit code $LASTEXITCODE" }
Copy-Item -Force (Join-Path $srcBackend "pyproject.toml") (Join-Path $runtimeBackend "pyproject.toml")
Copy-Item -Force (Join-Path $srcBackend "uv.lock") (Join-Path $runtimeBackend "uv.lock")

$legacyVoxcpmVenv = Join-Path $runtimeBackend "dv_backend\.venv-voxcpm"
if (Test-Path $legacyVoxcpmVenv) {
    Remove-Item -Recurse -Force $legacyVoxcpmVenv
    Write-Host "Removed legacy PyTorch voxcpm venv: $legacyVoxcpmVenv"
}

if (-not $SkipVenvSync) {
    Write-Host "Syncing portable Python venv..."
    $env:UV_PROJECT_ENVIRONMENT = $runtimeVenv
    Push-Location $runtimeBackend
    uv sync --frozen --python $runtimePython
    if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }
    Pop-Location
}

Require-Path (Join-Path $voxcpmSource "voxcpm2-cli.exe") @"
voxcpm2-cli.exe not found under $voxcpmSource.
Build llama.cpp-omni (voxcpm2-cli + llama-tts-server, GGML_CUDA=ON) and copy Release binaries into vendor/voxcpm2/.
"@

Write-Host "Bundling voxcpm2-cli into portable tools..."
robocopy $voxcpmSource $voxcpmTools /MIR /NJH /NJS /NDL /NFL | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy voxcpm2 tools failed with exit code $LASTEXITCODE" }

$devModels = Join-Path $srcBackend "models\voxcpm2"
if (Test-Path $devModels) {
    Write-Host "Copying existing GGUF weights from $devModels (if any)..."
    robocopy $devModels $voxcpmModels *.gguf /NJH /NJS /NDL /NFL | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy GGUF models failed with exit code $LASTEXITCODE" }
}

$manifestPath = Join-Path $RuntimeRoot "manifest.json"
$manifest = @{
    schema_version = 1
    tools          = @(
        @{
            id               = "ffmpeg"
            display_name     = "FFmpeg"
            executable       = "ffmpeg/ffmpeg.exe"
            dev_command      = "ffmpeg"
            version_args     = @("-version")
            version_contains = "ffmpeg"
            required         = $true
            capability       = "media"
        },
        @{
            id               = "voxcpm2-cli"
            display_name     = "VoxCPM2 CLI"
            executable       = "voxcpm2/voxcpm2-cli.exe"
            dev_command      = "voxcpm2-cli"
            version_args     = @("--help")
            version_contains = ""
            success_exit_codes = @(0, 1)
            required         = $true
            capability       = "tts"
        },
        @{
            id               = "yt_dlp"
            display_name     = "yt-dlp"
            executable       = "yt-dlp/yt-dlp.exe"
            dev_command      = "yt-dlp"
            version_args     = @("--version")
            version_contains = ""
            required         = $true
            capability       = "download"
        }
    )
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding UTF8
Write-Host "Wrote $manifestPath"

$env:DV_VENDOR_DIR = $runtimeTools
$env:DV_MODELS_DIR = $runtimeModels
$env:DV_VOXCPM_CLI = Join-Path $voxcpmTools "voxcpm2-cli.exe"
$env:PYTHONPATH = $srcBackend

$setupArgs = @(
    (Join-Path $srcBackend "scripts\setup_voxcpm.py"),
    "--models-dir", $voxcpmModels
)
if ($SkipModelDownload) {
    $setupArgs += "--skip-download"
}

Write-Host "Verifying VoxCPM2 GGUF runtime..."
if (-not $SkipModelDownload) {
    Write-Host "Downloading GGUF weights (skip with -SkipModelDownload if already present)..."
}
& $runtimePython @setupArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Portable runtime ready at $RuntimeRoot"
