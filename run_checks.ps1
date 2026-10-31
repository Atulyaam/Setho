# ==============================================================================
# Stetho Automated Verification Runner (Sprint 0)
# ==============================================================================

$ErrorActionPreference = "Stop"
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "STETHO AUTOMATED SUITE VERIFICATION" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Backend Pytest Suite
Write-Host "`n[1/5] Running Backend Unit & Integration Tests..." -ForegroundColor Yellow
& .\.venv\Scripts\pytest -v tests
if ($LASTEXITCODE -ne 0) {
    Write-Error "Backend test suite failed."
}
Write-Host "✓ Backend tests passed." -ForegroundColor Green

# 2. Frontend TypeScript Check
Write-Host "`n[2/5] Running Frontend TypeScript Typechecks..." -ForegroundColor Yellow
Push-Location apps\web
try {
    & npm run typecheck
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Frontend TypeScript checks failed."
    }
    Write-Host "✓ Frontend TypeScript checks passed." -ForegroundColor Green

    # 3. Frontend Lint
    Write-Host "`n[3/5] Running Frontend Linting..." -ForegroundColor Yellow
    & npm run lint
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Frontend lint checks failed."
    }
    Write-Host "✓ Frontend linting passed." -ForegroundColor Green

    # 4. Frontend Build
    Write-Host "`n[4/5] Running Frontend Production Build..." -ForegroundColor Yellow
    & npm run build
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Frontend production build failed."
    }
    Write-Host "✓ Frontend production build passed." -ForegroundColor Green
} finally {
    Pop-Location
}

# 5. Secret Leak & Git Check
Write-Host "`n[5/5] Checking Git Status and Secret Exclusions..." -ForegroundColor Yellow
$ignored = & git check-ignore .env
if ($ignored -notmatch "\.env") {
    Write-Error ".env file is not ignored by git!"
}
Write-Host "✓ Git ignore and secret protection verified." -ForegroundColor Green

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
