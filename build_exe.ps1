$ErrorActionPreference = "Stop"

py -m pip install -e ".[documents,desktop]"
py -m PyInstaller --noconfirm --clean LocalRAG.spec

Write-Host ""
Write-Host "Built dist\LocalRAG.exe"
