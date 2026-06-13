$ErrorActionPreference = "Stop"
Push-Location "$PSScriptRoot\..\backend"
python -m pytest -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Pop-Location

Push-Location "$PSScriptRoot\.."
npm test --workspace desktop
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
npm run build --workspace desktop
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Pop-Location
