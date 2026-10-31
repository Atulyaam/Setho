"""Automated test runner and environment verification for Stetho (Sprint 0)."""

import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def run_command(cmd: list[str], cwd: Path | None = None, desc: str = "") -> None:
    print(f"\n[CHECK] {desc}...")
    print(f"Running: {' '.join(cmd)} in {cwd or ROOT_DIR}")
    result = subprocess.run(cmd, cwd=cwd or ROOT_DIR)
    if result.returncode != 0:
        print(f"\n[FAILED] {desc} returned non-zero exit code: {result.returncode}")
        sys.exit(result.returncode)
    print(f"[PASSED] {desc}")


def main():
    print("=" * 60)
    print("STETHO AUTOMATED TEST & VERIFICATION SUITE")
    print("=" * 60)

    # Detect Python binary
    venv_python = (
        ROOT_DIR / ".venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else ROOT_DIR / ".venv" / "bin" / "python"
    )
    python_bin = str(venv_python) if venv_python.exists() else sys.executable

    # 1. Backend tests
    run_command(
        [python_bin, "-m", "pytest", "-v", "tests"],
        cwd=ROOT_DIR,
        desc="Backend Pytest Unit & Integration Tests",
    )

    # 2. Frontend typecheck
    web_dir = ROOT_DIR / "apps" / "web"
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"

    run_command(
        [npm_cmd, "run", "typecheck"],
        cwd=web_dir,
        desc="Frontend TypeScript Verification",
    )

    # 3. Frontend lint
    run_command(
        [npm_cmd, "run", "lint"],
        cwd=web_dir,
        desc="Frontend ESLint Verification",
    )

    # 4. Frontend production build
    run_command(
        [npm_cmd, "run", "build"],
        cwd=web_dir,
        desc="Frontend Production Build",
    )

    # 5. Git ignore and secret check
    git_check = subprocess.run(
        ["git", "check-ignore", ".env", "data/mimii/raw/fan/id_00/normal/00000000.wav"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    )
    if ".env" not in git_check.stdout:
        print("[FAILED] .env is not ignored by git!")
        sys.exit(1)
    if "00000000.wav" not in git_check.stdout:
        print("[FAILED] Raw WAV audio files are not ignored by git!")
        sys.exit(1)
    print("[PASSED] Secret Leak & Dataset .gitignore Verification")

    print("\n" + "=" * 60)
    print("ALL STETHO AUTOMATED CHECKS (SPRINT 0 + SPRINT 1) PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

