$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..")

$dst = "dist-portable\DouyinVietnamizer-0.1.0-portable"
$exe = "$dst\douyin-vietnamizer.exe"
$runtimeBackend = "$dst\portable-runtime\backend"

Write-Host "Running pnpm tauri:build..."
pnpm tauri:build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not (Test-Path "$dst\portable-runtime")) {
    Write-Error "Missing $dst\portable-runtime. Copy vendor\portable-runtime there once."
    exit 1
}

# Sync Rust/frontend binary.
Copy-Item -Force "src-tauri\target\release\douyin-vietnamizer.exe" $exe
Write-Host "Updated $exe"

# Sync Python source so backend code edits land in the portable runtime.
# `dv_backend/` and `scripts/` are the only dirs the app actually loads at runtime;
# the embedded `.venv` and models are untouched.
$srcBackend = "backend"
robocopy $srcBackend\dv_backend $runtimeBackend\dv_backend /MIR /NJH /NJS /NDL /NFL /XD __pycache__ .venv-voxcpm
robocopy $srcBackend\scripts $runtimeBackend\scripts /MIR /NJH /NJS /NDL /NFL
Copy-Item -Force $srcBackend\pyproject.toml $runtimeBackend\pyproject.toml
Write-Host "Synced backend Python to $runtimeBackend"
