$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $repoRoot

$vendorRuntime = Join-Path $repoRoot "vendor\portable-runtime"
$dst = Join-Path $repoRoot "dist-portable\DouyinVietnamizer-0.1.0-portable"
$exe = Join-Path $dst "douyin-vietnamizer.exe"
$runtimeRoot = Join-Path $dst "portable-runtime"
$runtimeBackend = Join-Path $runtimeRoot "backend"
$runtimePython = Join-Path $runtimeRoot ".venv\Scripts\python.exe"
$runtimeVenv = Join-Path $runtimeRoot ".venv"

Write-Host "Refreshing vendor portable runtime (GGUF TTS bundle)..."
& "$PSScriptRoot\build-portable-runtime-windows.ps1" -RuntimeRoot $vendorRuntime
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Running pnpm tauri build --no-bundle..."
pnpm tauri build --no-bundle
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

New-Item -ItemType Directory -Force -Path $dst | Out-Null
Write-Host "Mirroring portable runtime into $runtimeRoot ..."
robocopy $vendorRuntime $runtimeRoot /MIR /NJH /NJS /NDL /NFL /XD __pycache__ | Out-Null
if ($LASTEXITCODE -ge 8) {
    Write-Error "Failed to mirror portable runtime (robocopy exit $LASTEXITCODE)."
    exit 1
}

if (-not (Test-Path $runtimePython)) {
    Write-Error "Missing $runtimePython. Ensure portable runtime venv is present."
    exit 1
}

# Sync Rust/frontend binary.
Copy-Item -Force (Join-Path $repoRoot "src-tauri\target\release\douyin-vietnamizer.exe") $exe
Write-Host "Updated $exe"

# Sync Python source so backend code edits land in the portable runtime.
# `dv_backend/` and `scripts/` are the only dirs the app actually loads at runtime;
# the embedded `.venv` and models are untouched.
$srcBackend = Join-Path $repoRoot "backend"
robocopy (Join-Path $srcBackend "dv_backend") (Join-Path $runtimeBackend "dv_backend") /MIR /NJH /NJS /NDL /NFL /XD __pycache__ .venv-voxcpm
robocopy (Join-Path $srcBackend "scripts") (Join-Path $runtimeBackend "scripts") /MIR /NJH /NJS /NDL /NFL
Copy-Item -Force (Join-Path $srcBackend "pyproject.toml") (Join-Path $runtimeBackend "pyproject.toml")
Copy-Item -Force (Join-Path $srcBackend "uv.lock") (Join-Path $runtimeBackend "uv.lock")
Write-Host "Synced backend Python to $runtimeBackend"

$staleBackendVenv = Join-Path $runtimeBackend ".venv"
if (Test-Path $staleBackendVenv) {
    Remove-Item -Recurse -Force $staleBackendVenv
    Write-Host "Removed stale backend/.venv created by an earlier sync attempt."
}

Write-Host "Syncing portable backend venv..."
$env:UV_PROJECT_ENVIRONMENT = $runtimeVenv
Push-Location $runtimeBackend
uv sync --frozen --python $runtimePython
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }
Pop-Location
Write-Host "Portable venv synced."
