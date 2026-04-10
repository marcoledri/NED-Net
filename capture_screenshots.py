"""Capture screenshots for the NED-Net user manual.

Requires: pip install playwright && python -m playwright install chromium
Run the server first: python -m eeg_seizure_analyzer.dash_app.main
"""
import time
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8050"
OUT = "eeg_seizure_analyzer/dash_app/assets/screenshots"
EDF = "/Users/marcoledri/Dropbox/Work/eeg-seizure-shared/Recordings/Test.edf"


def wait(page, ms=2000):
    page.wait_for_timeout(ms)


def click_tab(page, tab_id):
    page.click(f"#tab-{tab_id}", timeout=5000)
    wait(page, 2000)


def shot(page, name, full_page=False):
    path = f"{OUT}/{name}"
    page.screenshot(path=path, full_page=full_page)
    print(f"  ✓ {name}")


def scroll_to(page, y):
    page.evaluate(f"document.querySelector('#tab-content')?.scrollTo(0, {y})")
    wait(page, 500)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        # ── Load file ────────────────────────────────────
        print("Loading EDF file...")
        page.goto(BASE)
        wait(page, 3000)

        page.click("#landing-load-file-btn", timeout=5000)
        wait(page, 1000)

        page.fill("#upload-path-input", EDF)
        page.click("#upload-path-btn", timeout=5000)
        wait(page, 4000)

        page.click("#upload-load-btn", timeout=10000)
        wait(page, 8000)

        # ── 1. interface_overview.png ────────────────────
        print("Taking screenshots...")
        scroll_to(page, 0)
        shot(page, "interface_overview.png")

        # ── 2. load_single.png ───────────────────────────
        shot(page, "load_single.png")

        # ── 3. animal_ids.png ────────────────────────────
        scroll_to(page, 250)
        shot(page, "animal_ids.png")

        # ── 4. theme_toggle.png ──────────────────────────
        scroll_to(page, 0)
        shot(page, "theme_toggle.png")

        # ── 5. viewer.png ────────────────────────────────
        click_tab(page, "viewer")
        wait(page, 3000)
        shot(page, "viewer.png")

        # ── 6. Detection → Seizure ──────────────────────
        click_tab(page, "detection")
        wait(page, 2000)

        # Find and click Seizure subtab
        try:
            page.click("text=Seizure", timeout=3000)
            wait(page, 2000)
        except Exception:
            pass  # May already be on seizure subtab

        scroll_to(page, 0)
        shot(page, "method_selector.png")

        # ── 7. Change method via RadioItems ──────────────
        # The method selector is dbc.RadioItems with id="sz-method-selector"
        try:
            # Spectral Band
            page.click("#sz-method-selector label:has-text('Spectral Band')", timeout=3000)
            wait(page, 1500)
            scroll_to(page, 0)
            shot(page, "spectral_band_params.png")

            # Autocorrelation
            page.click("#sz-method-selector label:has-text('Autocorrelation')", timeout=3000)
            wait(page, 1500)
            scroll_to(page, 0)
            shot(page, "autocorrelation_params.png")

            # Ensemble
            page.click("#sz-method-selector label:has-text('Ensemble')", timeout=3000)
            wait(page, 1500)
            scroll_to(page, 0)
            shot(page, "ensemble_params.png")

            # Back to Spike-Train
            page.click("#sz-method-selector label:has-text('Spike-Train')", timeout=3000)
            wait(page, 1000)
        except Exception as e:
            print(f"  ⚠ Method switch failed: {e}")

        # ── 8. Scroll to results/filters ─────────────────
        scroll_to(page, 500)
        shot(page, "detection_results.png")

        scroll_to(page, 400)
        shot(page, "filter_controls.png")

        # ── 9. Click a result row for inspector ──────────
        try:
            row = page.locator(".ag-row").first
            if row.count() > 0:
                row.click(timeout=3000)
                wait(page, 2000)
                scroll_to(page, 900)
                shot(page, "event_inspector.png")

                # Export controls (at the bottom)
                scroll_to(page, 9999)
                wait(page, 500)
                shot(page, "export_csv.png")
        except Exception as e:
            print(f"  ⚠ Inspector/export: {e}")

        # ── 10. Training tab ─────────────────────────────
        click_tab(page, "training_grp")
        wait(page, 3000)
        shot(page, "training_review.png")

        # ── 11. Detection → Interictal Spikes ────────────
        click_tab(page, "detection")
        wait(page, 2000)
        # Click the "Interictal Spikes" subtab
        try:
            page.click("text=Interictal Spikes", timeout=5000)
            wait(page, 3000)
            shot(page, "spike_annotation.png")
        except Exception as e:
            print(f"  ⚠ Spike tab: {e}")

        # ── 12. Dataset / Model tab ──────────────────────
        click_tab(page, "ml_grp")
        wait(page, 3000)
        shot(page, "ml_datasets.png")
        shot(page, "ml_training_progress.png")

        # ── 13. Analysis tab ─────────────────────────────
        click_tab(page, "analysis")
        wait(page, 2000)
        shot(page, "analysis_tab.png")

        # ── 14. Results tab ──────────────────────────────
        click_tab(page, "results")
        wait(page, 2000)
        shot(page, "results_summary.png")

        # ── 15. Batch progress (Detection tab) ───────────
        click_tab(page, "detection")
        wait(page, 1500)
        try:
            page.click("text=Seizure", timeout=3000)
            wait(page, 1500)
        except Exception:
            pass
        scroll_to(page, 350)
        shot(page, "batch_progress.png")

        browser.close()
        print(f"\nDone! {len(__import__('os').listdir(OUT))} screenshots saved to {OUT}/")


if __name__ == "__main__":
    main()
