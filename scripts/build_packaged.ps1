# Packaging and Release build script for Douyin Vietnamizer Portable Edition
$ErrorActionPreference = "Stop"

$root = Resolve-Path "$PSScriptRoot\.."
$distDir = "$root\dist_portable"

Write-Host "=== Staging directory: $distDir ===" -ForegroundColor Cyan
if (Test-Path $distDir) {
    Remove-Item -Recurse -Force $distDir
}
New-Item -ItemType Directory -Path $distDir | Out-Null

# 1. Build React Frontend
Write-Host "=== Building React Frontend ===" -ForegroundColor Cyan
Push-Location "$root"
npm run build --workspace desktop
Pop-Location

# 2. Package Python Backend with PyInstaller
Write-Host "=== Packaging Python Backend ===" -ForegroundColor Cyan
Push-Location "$root\backend"
python -m pip install pyinstaller
Write-Host "Running PyInstaller..."
python -m PyInstaller --clean --onefile --name dv_backend --noconsole dv_backend/main.py
Pop-Location

# 3. Assemble Portable Folder Structure
Write-Host "=== Assembling Portable Files ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Path "$distDir\backend" | Out-Null
Copy-Item "$root\backend\dist\dv_backend.exe" "$distDir\backend\dv_backend.exe"

# Copy vendor manifest and directories
if (Test-Path "$root\vendor") {
    Write-Host "Copying vendor binaries..."
    Copy-Item -Recurse "$root\vendor" "$distDir\vendor"
} else {
    Write-Host "Warning: No vendor binaries staged in workspace. Creating empty vendor/ dir." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "$distDir\vendor" | Out-Null
    Copy-Item "$root\vendor\manifest.json" "$distDir\vendor\manifest.json"
}

# 4. Instructions for final Electron packaging
Write-Host "=== Packaging Complete ===" -ForegroundColor Green
Write-Host "Staged portable build files are ready in: $distDir" -ForegroundColor Green
Write-Host "To bundle this staged folder into a single setup installer, run:" -ForegroundColor Cyan
Write-Host "  npx electron-builder --dir" -ForegroundColor Yellow
