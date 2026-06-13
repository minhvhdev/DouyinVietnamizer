$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot\.."

Push-Location "$root\backend"
python -m pip install -e ".[dev]"
$env:DV_ALLOW_PATH_TOOLS = "1"
Start-Process python -ArgumentList "-m", "dv_backend.main" -WorkingDirectory "$root\backend" -WindowStyle Hidden
Pop-Location

Push-Location $root
npm install
npm run dev --workspace desktop
Pop-Location

