"""Playwright E2E Browser Validation for Sprint 7.8 / E9 Field Data Workflow.

Executes all 18 manual verification scenarios against the live FastAPI backend
and Next.js frontend, asserting state transitions, UI element responses,
and persisting full-resolution screenshots into docs/evidence/browser_e9/.
"""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import numpy as np
from playwright.sync_api import sync_playwright
import soundfile as sf

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
EVIDENCE_DIR = Path("docs/evidence/browser_e9").resolve()
TEMP_AUDIO_DIR = Path("data/storage/test_audio").resolve()


def generate_wav(filepath: Path, duration_sec: float = 3.0, sample_rate: int = 16000, frequency: float = 440.0, amplitude: float = 0.25):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    if amplitude > 0:
        t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
        sig = amplitude * np.sin(2 * np.pi * frequency * t)
        sig += np.random.uniform(-0.002, 0.002, size=num_samples).astype(np.float32)
    else:
        sig = np.zeros(num_samples, dtype=np.float32)
    sf.write(str(filepath), sig, sample_rate, format="WAV", subtype="PCM_16")


def wait_for_port(url: str, timeout_sec: int = 30) -> bool:
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HealthCheck"})
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status in (200, 404, 307):
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def run_e2e_browser_audit():
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Prepare test audio files
    valid_wav = TEMP_AUDIO_DIR / "field_fan_bearing_valid.wav"
    generate_wav(valid_wav, duration_sec=3.0, sample_rate=16000, frequency=380.0, amplitude=0.25)

    subnyquist_wav = TEMP_AUDIO_DIR / "subnyquist_8khz.wav"
    generate_wav(subnyquist_wav, duration_sec=2.0, sample_rate=8000, frequency=300.0, amplitude=0.25)

    silent_wav = TEMP_AUDIO_DIR / "silent_audio.wav"
    generate_wav(silent_wav, duration_sec=2.0, sample_rate=16000, amplitude=0.0)

    print("Generated test audio files.")

    # 2. Launch FastAPI backend
    print("Starting FastAPI backend on port 8000...")
    backend_log = open("data/backend_test.log", "w", encoding="utf-8")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "stetho_api.main:create_app", "--factory", "--host", "127.0.0.1", "--port", "8000"],
        stdout=backend_log,
        stderr=subprocess.STDOUT,
        cwd=os.getcwd(),
        text=True,
    )

    # 3. Launch Next.js frontend
    print("Starting Next.js frontend on port 3000...")
    front_env = os.environ.copy()
    front_env["NEXT_PUBLIC_API_URL"] = "http://127.0.0.1:8000"
    frontend_log = open("data/frontend_test.log", "w", encoding="utf-8")
    frontend_proc = subprocess.Popen(
        ["cmd.exe", "/c", "npm", "run", "start", "--prefix", "apps/web"],
        stdout=frontend_log,
        stderr=subprocess.STDOUT,
        cwd=os.getcwd(),
        text=True,
        env=front_env,
    )

    try:
        if not wait_for_port("http://127.0.0.1:8000/docs", 30):
            raise RuntimeError("FastAPI backend failed to start on port 8000.")
        print("FastAPI backend is ready.")

        if not wait_for_port("http://127.0.0.1:3000/login", 45):
            raise RuntimeError("Next.js frontend failed to start on port 3000.")
        print("Next.js frontend is ready.")

        # 4. Start Playwright
        print("Launching Headless Chrome...")
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=CHROME_PATH,
                headless=True,
                args=["--no-sandbox", "--disable-gpu", "--window-size=1400,950"],
            )
            context = browser.new_context(viewport={"width": 1400, "height": 950})
            page = context.new_page()
            page.on("console", lambda msg: print(f"[Browser Console] {msg.type}: {msg.text}"))
            page.on("pageerror", lambda err: print(f"[Browser PageError] {err}"))
            page.on("response", lambda res: print(f"[HTTP {res.status}] {res.url}") if "/api/" in res.url else None)

            # Step 0: Login
            print("Authenticating as admin...")
            page.goto("http://127.0.0.1:3000/login")
            page.wait_for_selector("input[type='text']")
            page.fill("input[type='text']", "admin")
            page.fill("input[type='password']", "StethoAdmin2026!")
            page.click("button[type='submit']")
            page.wait_for_url("**/dashboard", timeout=15000)
            print("Authenticated successfully.")

            # Scenario 1: Open Field Data dashboard
            print("\nScenario 1: Open Field Data dashboard")
            page.click("a[href='/field-data']")
            page.wait_for_selector("h1:has-text('Field Data Collection & Pilot Readiness')", timeout=15000)
            page.screenshot(path=str(EVIDENCE_DIR / "01_dashboard_open.png"))
            print("Captured: 01_dashboard_open.png")

            # Scenario 2: Upload valid WAV file
            print("\nScenario 2: Upload valid WAV file")
            page.click("button:has-text('Import Field Audio')")
            page.wait_for_selector("h3:has-text('Import Field Acoustic Audio')")
            file_input = page.locator("input[type='file']")
            file_input.set_input_files(str(valid_wav))
            page.fill("input[placeholder='e.g. fan_unit_04a']", "fan_pilot_field_01")
            page.click("button:has-text('Verify & Ingest')")
            page.wait_for_selector("div:has-text('successfully imported and verified')", timeout=15000)
            page.screenshot(path=str(EVIDENCE_DIR / "02_upload_valid_wav.png"))
            print("Captured: 02_upload_valid_wav.png")

            # Scenario 3: Confirm recording appears with correct metadata
            print("\nScenario 3: Confirm recording appears with correct metadata")
            page.wait_for_selector("td:has-text('fan_pilot_field_01')")
            rec_row = page.locator("tr:has-text('fan_pilot_field_01')")
            assert rec_row.locator("text=16000 Hz").count() > 0
            assert rec_row.locator("text=fan").count() > 0
            page.screenshot(path=str(EVIDENCE_DIR / "03_recording_metadata_confirmed.png"))
            print("Captured: 03_recording_metadata_confirmed.png")

            # Scenario 4: Upload same file again and confirm duplicate rejection message
            print("\nScenario 4: Duplicate upload rejection")
            page.click("button:has-text('Import Field Audio')")
            page.wait_for_selector("h3:has-text('Import Field Acoustic Audio')")
            page.locator("input[type='file']").set_input_files(str(valid_wav))
            page.fill("input[placeholder='e.g. fan_unit_04a']", "fan_pilot_duplicate")
            page.click("button:has-text('Verify & Ingest')")
            page.wait_for_selector("div:has-text('already exists'), div:has-text('DUPLICATE_RECORDING'), div:has-text('rejected')", timeout=15000)
            page.screenshot(path=str(EVIDENCE_DIR / "04_duplicate_rejection.png"))
            print("Captured: 04_duplicate_rejection.png")
            page.click("button:has-text('Cancel')")
            if page.locator("button:has-text('Dismiss')").count() > 0:
                page.click("button:has-text('Dismiss')")

            # Scenario 5: Upload unsupported sample-rate file (8 kHz)
            print("\nScenario 5: Unsupported sample-rate file rejection")
            page.click("button:has-text('Import Field Audio')")
            page.wait_for_selector("h3:has-text('Import Field Acoustic Audio')")
            page.locator("input[type='file']").set_input_files(str(subnyquist_wav))
            page.fill("input[placeholder='e.g. fan_unit_04a']", "fan_subnyquist_01")
            page.click("button:has-text('Verify & Ingest')")
            page.wait_for_selector("div:has-text('sample rate'), div:has-text('UNSUPPORTED_SAMPLE_RATE'), div:has-text('rejected')", timeout=15000)
            page.screenshot(path=str(EVIDENCE_DIR / "05_sample_rate_rejection.png"))
            print("Captured: 05_sample_rate_rejection.png")
            page.click("button:has-text('Cancel')")
            if page.locator("button:has-text('Dismiss')").count() > 0:
                page.click("button:has-text('Dismiss')")

            # Scenario 6: Upload clipped or silent recording
            print("\nScenario 6: Silent recording rejection")
            page.click("button:has-text('Import Field Audio')")
            page.wait_for_selector("h3:has-text('Import Field Acoustic Audio')")
            page.locator("input[type='file']").set_input_files(str(silent_wav))
            page.fill("input[placeholder='e.g. fan_unit_04a']", "fan_silent_01")
            page.click("button:has-text('Verify & Ingest')")
            page.wait_for_selector("div:has-text('silent'), div:has-text('QUALITY_CHECK_FAILED'), div:has-text('rejected')", timeout=15000)
            page.screenshot(path=str(EVIDENCE_DIR / "06_silence_quality_rejection.png"))
            print("Captured: 06_silence_quality_rejection.png")
            page.click("button:has-text('Cancel')")
            if page.locator("button:has-text('Dismiss')").count() > 0:
                page.click("button:has-text('Dismiss')")

            # Scenario 7: Inspect recording details in table
            print("\nScenario 7: Inspect recording details in table")
            page.wait_for_selector("tr:has-text('fan_pilot_field_01')")
            page.screenshot(path=str(EVIDENCE_DIR / "07_table_details_inspected.png"))
            print("Captured: 07_table_details_inspected.png")

            # Scenario 8: Annotate recording as normal
            print("\nScenario 8: Annotate recording as normal")
            page.locator("tr:has-text('fan_pilot_field_01') button:has-text('Annotate')").first.click()
            page.wait_for_selector("h3:has-text('Ground-Truth Annotation')")
            annot_modal = page.locator("div.fixed:has-text('Ground-Truth Annotation')")
            annot_modal.locator("select").nth(0).select_option("normal")
            annot_modal.locator("select").nth(1).select_option("VISUAL_INSPECTION")
            annot_modal.locator("input[placeholder*='bearing']").fill("Technician confirmed normal acoustic and vibration baseline")
            annot_modal.locator("button:has-text('Commit Revision')").click()
            page.wait_for_selector("div:has-text('ground-truth updated to NORMAL')", timeout=15000)
            page.screenshot(path=str(EVIDENCE_DIR / "08_annotate_normal.png"))
            print("Captured: 08_annotate_normal.png")
            if page.locator("button:has-text('Dismiss')").count() > 0:
                page.click("button:has-text('Dismiss')")

            # Scenario 9: Add revised annotation with evidence
            print("\nScenario 9: Add revised annotation with evidence")
            page.locator("tr:has-text('fan_pilot_field_01') button:has-text('Annotate')").first.click()
            page.wait_for_selector("h3:has-text('Ground-Truth Annotation')")
            annot_modal = page.locator("div.fixed:has-text('Ground-Truth Annotation')")
            annot_modal.locator("select").nth(0).select_option("uncertain")
            annot_modal.locator("select").nth(1).select_option("OPERATIONAL_LOG")
            annot_modal.locator("input[placeholder*='bearing']").fill("Minor acoustic hiss reported during peak load handover")
            annot_modal.locator("button:has-text('Commit Revision')").click()
            page.wait_for_selector("div:has-text('ground-truth updated to UNCERTAIN')", timeout=15000)
            page.screenshot(path=str(EVIDENCE_DIR / "09_revised_annotation_added.png"))
            print("Captured: 09_revised_annotation_added.png")
            if page.locator("button:has-text('Dismiss')").count() > 0:
                page.click("button:has-text('Dismiss')")

            # Scenario 10: Confirm annotation history displays revisions 1 and 2
            print("\nScenario 10: Confirm annotation history displays revisions 1 and 2")
            page.locator("tr:has-text('fan_pilot_field_01') button:has-text('Annotate')").first.click()
            page.wait_for_selector("h3:has-text('Ground-Truth Annotation')")
            annot_modal = page.locator("div.fixed:has-text('Ground-Truth Annotation')")
            annot_modal.locator("text=Rev #1:").wait_for(timeout=10000)
            annot_modal.locator("text=Rev #2:").wait_for(timeout=10000)
            page.screenshot(path=str(EVIDENCE_DIR / "10_annotation_history_revisions_1_and_2.png"))
            annot_modal.locator("button:has-text('Cancel')").click()
            print("Captured: 10_annotation_history_revisions_1_and_2.png")

            # Scenario 11: Assign valid dataset role ('development')
            print("\nScenario 11: Assign valid dataset role ('development')")
            page.locator("tr:has-text('fan_pilot_field_01') button:has-text('Role')").first.click()
            page.wait_for_selector("h3:has-text('Update Dataset Separation Role')")
            role_modal = page.locator("div.fixed:has-text('Update Dataset Separation Role')")
            role_modal.locator("select").select_option("development")
            role_modal.locator("button:has-text('Update Role')").click()
            page.wait_for_selector("div:has-text('updated to development')", timeout=15000)
            page.screenshot(path=str(EVIDENCE_DIR / "11_role_assigned_development.png"))
            print("Captured: 11_role_assigned_development.png")
            if page.locator("button:has-text('Dismiss')").count() > 0:
                page.click("button:has-text('Dismiss')")

            # Scenario 12: Attempt to assign abnormal recording to calibration_enrollment
            print("\nScenario 12: Attempt to assign abnormal recording to calibration_enrollment")
            # First set label to abnormal
            page.locator("tr:has-text('fan_pilot_field_01') button:has-text('Annotate')").first.click()
            page.wait_for_selector("h3:has-text('Ground-Truth Annotation')")
            annot_modal = page.locator("div.fixed:has-text('Ground-Truth Annotation')")
            annot_modal.locator("select").nth(0).select_option("abnormal")
            annot_modal.locator("select").nth(1).select_option("VIBRATION_ORDER_ANALYSIS")
            annot_modal.locator("input[placeholder*='bearing']").fill("Verified bearing cage fracture flaw")
            annot_modal.locator("button:has-text('Commit Revision')").click()
            page.wait_for_selector("div:has-text('ground-truth updated to ABNORMAL')", timeout=15000)
            if page.locator("button:has-text('Dismiss')").count() > 0:
                page.click("button:has-text('Dismiss')")

            # Now try to set role to calibration_enrollment
            page.locator("tr:has-text('fan_pilot_field_01') button:has-text('Role')").first.click()
            page.wait_for_selector("h3:has-text('Update Dataset Separation Role')")
            role_modal = page.locator("div.fixed:has-text('Update Dataset Separation Role')")
            role_modal.locator("select").select_option("calibration_enrollment")
            role_modal.locator("button:has-text('Update Role')").click()
            page.wait_for_selector("div:has-text('Cannot assign role'), div:has-text('calibration_enrollment'), div:has-text('DATASET_ROLE_CONFLICT')", timeout=15000)
            page.screenshot(path=str(EVIDENCE_DIR / "12_calibration_enrollment_rejected.png"))
            role_modal.locator("button:has-text('Cancel')").click()
            print("Captured: 12_calibration_enrollment_rejected.png")
            if page.locator("button:has-text('Dismiss')").count() > 0:
                page.click("button:has-text('Dismiss')")

            # Scenario 13: Create dataset manifest
            print("\nScenario 13: Create dataset manifest")
            # First set to verified normal and sealed_evaluation
            page.locator("tr:has-text('fan_pilot_field_01') button:has-text('Annotate')").first.click()
            page.wait_for_selector("h3:has-text('Ground-Truth Annotation')")
            annot_modal = page.locator("div.fixed:has-text('Ground-Truth Annotation')")
            annot_modal.locator("select").nth(0).select_option("normal")
            annot_modal.locator("select").nth(1).select_option("VISUAL_INSPECTION")
            annot_modal.locator("input[placeholder*='bearing']").fill("Verified healthy baseline for pilot evaluation split")
            annot_modal.locator("button:has-text('Commit Revision')").click()
            page.wait_for_selector("div:has-text('ground-truth updated to NORMAL')", timeout=15000)
            if page.locator("button:has-text('Dismiss')").count() > 0:
                page.click("button:has-text('Dismiss')")

            page.locator("tr:has-text('fan_pilot_field_01') button:has-text('Role')").first.click()
            page.wait_for_selector("h3:has-text('Update Dataset Separation Role')")
            role_modal = page.locator("div.fixed:has-text('Update Dataset Separation Role')")
            role_modal.locator("select").select_option("sealed_evaluation")
            role_modal.locator("button:has-text('Update Role')").click()
            page.wait_for_selector("div:has-text('updated to sealed_evaluation')", timeout=15000)
            if page.locator("button:has-text('Dismiss')").count() > 0:
                page.click("button:has-text('Dismiss')")

            # Create manifest
            page.click("button:has-text('Seal Manifest')")
            page.wait_for_selector("h3:has-text('Create & Seal Manifest')")
            manifest_modal = page.locator("div.fixed:has-text('Create & Seal Manifest')")
            manifest_modal.locator("input[placeholder*='Pilot Stage 1']").fill("Sprint 7.8 Pilot Fan Benchmark")
            manifest_modal.locator("button[type='submit']:has-text('Generate Manifest')").click()
            page.wait_for_selector("div:has-text('created and sealed')", timeout=15000)
            page.screenshot(path=str(EVIDENCE_DIR / "13_manifest_created_and_sealed.png"))
            print("Captured: 13_manifest_created_and_sealed.png")
            if page.locator("button:has-text('Dismiss')").count() > 0:
                page.click("button:has-text('Dismiss')")

            # Scenario 14: Confirm sealed manifest in UI
            print("\nScenario 14: Confirm sealed manifest in UI")
            page.click("button:has-text('Sealed Manifests')")
            page.wait_for_selector("td:has-text('Sprint 7.8 Pilot Fan Benchmark')")
            assert page.locator("span:has-text('Sealed')").count() > 0
            page.screenshot(path=str(EVIDENCE_DIR / "14_sealed_manifest_confirmed.png"))
            print("Captured: 14_sealed_manifest_confirmed.png")

            # Scenario 15: Attempt to modify a sealed evaluation recording -> verify locked state
            print("\nScenario 15: Attempt to modify sealed evaluation recording")
            page.click("button:has-text('Field Audio Recordings')")
            page.wait_for_selector("tr:has-text('fan_pilot_field_01')")
            # Try role change
            page.locator("tr:has-text('fan_pilot_field_01') button:has-text('Role')").first.click()
            page.wait_for_selector("h3:has-text('Update Dataset Separation Role')")
            role_modal = page.locator("div.fixed:has-text('Update Dataset Separation Role')")
            role_modal.locator("select").select_option("training")
            role_modal.locator("button:has-text('Update Role')").click()
            page.wait_for_selector("div:has-text('sealed evaluation manifest'), div:has-text('cannot be reassigned'), div:has-text('MANIFEST_ALREADY_SEALED')", timeout=15000)
            page.screenshot(path=str(EVIDENCE_DIR / "15_sealed_recording_locked.png"))
            role_modal.locator("button:has-text('Cancel')").click()
            print("Captured: 15_sealed_recording_locked.png")
            if page.locator("button:has-text('Dismiss')").count() > 0:
                page.click("button:has-text('Dismiss')")

            # Scenario 16: Refresh browser and verify persistence
            print("\nScenario 16: Refresh browser and verify persistence")
            page.reload()
            page.wait_for_selector("h1:has-text('Field Data Collection & Pilot Readiness')")
            page.wait_for_selector("td:has-text('fan_pilot_field_01')")
            page.screenshot(path=str(EVIDENCE_DIR / "16_browser_refresh_persistence.png"))
            print("Captured: 16_browser_refresh_persistence.png")

            # Scenario 17: Restart backend and verify persistence
            print("\nScenario 17: Restart backend and verify persistence")
            backend_proc.terminate()
            backend_proc.wait(timeout=5)
            print("FastAPI backend stopped. Restarting...")
            backend_log2 = open("data/backend_restart.log", "w", encoding="utf-8")
            backend_proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "stetho_api.main:create_app", "--factory", "--host", "127.0.0.1", "--port", "8000"],
                stdout=backend_log2,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                text=True,
            )
            if not wait_for_port("http://127.0.0.1:8000/docs", 30):
                raise RuntimeError("FastAPI restart failed.")
            print("FastAPI restarted.")

            page.reload()
            page.wait_for_selector("h1:has-text('Field Data Collection & Pilot Readiness')")
            page.wait_for_selector("td:has-text('fan_pilot_field_01')")
            page.screenshot(path=str(EVIDENCE_DIR / "17_backend_restart_persistence.png"))
            print("Captured: 17_backend_restart_persistence.png")

            # Scenario 18: Check that dashboard reports INSUFFICIENT INDEPENDENT EVIDENCE warning banner
            print("\nScenario 18: Check warning banner")
            banner = page.locator("h3:has-text('INSUFFICIENT INDEPENDENT EVIDENCE: Controlled Field Pilot in Progress')")
            assert banner.count() > 0
            page.screenshot(path=str(EVIDENCE_DIR / "18_warning_banner_confirmed.png"))
            print("Captured: 18_warning_banner_confirmed.png")

            browser.close()
            print("\nAll 18 browser scenarios executed and verified successfully!")

    finally:
        print("Cleaning up background processes...")
        try:
            backend_proc.terminate()
            backend_proc.kill()
        except Exception:
            pass
        try:
            # On Windows kill node tree
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(frontend_proc.pid)], capture_output=True)
        except Exception:
            pass


if __name__ == "__main__":
    run_e2e_browser_audit()
