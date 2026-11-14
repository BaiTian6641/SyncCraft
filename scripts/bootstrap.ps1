$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $PSScriptRoot
Set-Location $RepoDir
py -3 -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ is required'"
if ($LASTEXITCODE -ne 0) { throw "Python version check failed" }
py -3 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed" }
& .venv\Scripts\python.exe -m pip install -r requirements-core.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
& .venv\Scripts\python.exe -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw "Editable installation failed" }
& .venv\Scripts\python.exe scripts/check.py
if ($LASTEXITCODE -ne 0) { throw "CPU checks failed" }
Write-Output "Ready. Run: .venv\Scripts\python.exe scripts/demo.py"
