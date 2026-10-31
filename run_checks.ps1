# ==============================================================================
# Stetho Automated Verification Runner (Sprint 0 + Sprint 1)
# ==============================================================================

$ErrorActionPreference = "Stop"
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "STETHO AUTOMATED SUITE VERIFICATION" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Backend Pytest Suite
Write-Host "`n[1/5] Running Backend Unit & Integration Tests..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m pytest -v tests
if ($LASTEXITCODE -ne 0) {
    Write-Error "Backend test suite failed."
}
Write-Host "[OK] Backend tests passed." -ForegroundColor Green

# 2. Frontend TypeScript Check
Write-Host "`n[2/5] Running Frontend TypeScript Typechecks..." -ForegroundColor Yellow
Push-Location apps\web
try {
    & npm run typecheck
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Frontend TypeScript checks failed."
    }
    Write-Host "[OK] Frontend TypeScript checks passed." -ForegroundColor Green

    # 3. Frontend Lint
    Write-Host "`n[3/5] Running Frontend Linting..." -ForegroundColor Yellow
    & npm run lint
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Frontend lint checks failed."
    }
    Write-Host "[OK] Frontend linting passed." -ForegroundColor Green

    # 4. Frontend Build
    Write-Host "`n[4/5] Running Frontend Production Build..." -ForegroundColor Yellow
    & npm run build
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Frontend production build failed."
    }
    Write-Host "[OK] Frontend production build passed." -ForegroundColor Green
} finally {
    Pop-Location
}

# 5. Secret Leak & Git Check
Write-Host "`n[5/5] Checking Git Status and Exclusions..." -ForegroundColor Yellow
$env_ignored = & git check-ignore .env
if (-not $env_ignored) {
    Write-Error ".env file is not ignored by git!"
}
$wav_ignored = & git check-ignore data/mimii/raw/fan/id_00/normal/00000000.wav
if (-not $wav_ignored) {
    Write-Error "Raw WAV files are not ignored by git!"
}
Write-Host "[OK] Git ignore and dataset safety verified." -ForegroundColor Green

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "ALL VERIFICATION CHECKS (SPRINT 0 + SPRINT 1) PASSED!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
