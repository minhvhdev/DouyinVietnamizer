$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..")
pnpm test
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
pnpm run build
