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


def shot(page, name, full_page=False, clip=None):
    path = f"{OUT}/{name}"
    kwargs = {"path": path, "full_page": full_page}
    if clip:
        kwargs["clip"] = clip
    page.screenshot(**kwargs)
    print(f"  ✓ {name}")


def scroll_into_view(page, selector):
    """Scroll an element into view using JS."""
    page.evaluate(f"document.querySelector('{selector}')?.scrollIntoView({{behavior: 'instant', block: 'center'}})")
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
        # Full-page shot showing sidebar + tab bar + main content
        print("Taking screenshots...")
        shot(page, "interface_overview.png", full_page=True)

        # ── 2. load_single.png ───────────────────────────
        # Crop to just the main content area (no sidebar)
        shot(page, "load_single.png")

        # ── 3. animal_ids.png ────────────────────────────
        # Scroll to show the channel table with Animal ID column
        scroll_into_view(page, "#upload-channel-ids-grid")
        wait(page, 500)
        shot(page, "animal_ids.png")

        # ── 4. theme_toggle.png ──────────────────────────
        # Click the dark mode toggle, take screenshot, then switch back
        page.click("#theme-toggle", timeout=3000)
        wait(page, 1000)
        # Scroll back to top so the toggle is visible in sidebar
        page.evaluate("window.scrollTo(0, 0)")
        wait(page, 500)
        shot(page, "theme_toggle.png")
        # Switch back to light mode
        page.click("#theme-toggle", timeout=3000)
        wait(page, 1000)

        # ── 5. viewer.png ────────────────────────────────
        click_tab(page, "viewer")
        wait(page, 3000)
        shot(page, "viewer.png")

        # ── 6. Detection → Seizure ──────────────────────
        click_tab(page, "detection")
        wait(page, 2000)

        # Ensure we're on the Seizure subtab
        try:
            page.click("text=Seizure", timeout=3000)
            wait(page, 2000)
        except Exception:
            pass  # May already be on seizure subtab

        shot(page, "method_selector.png")

        # ── 7. Change method via RadioItems ──────────────
        try:
            # Spectral Band
            page.click("#sz-method-selector label:has-text('Spectral Band')", timeout=3000)
            wait(page, 1500)
            shot(page, "spectral_band_params.png")

            # Autocorrelation
            page.click("#sz-method-selector label:has-text('Autocorrelation')", timeout=3000)
            wait(page, 1500)
            shot(page, "autocorrelation_params.png")

            # Ensemble
            page.click("#sz-method-selector label:has-text('Ensemble')", timeout=3000)
            wait(page, 1500)
            shot(page, "ensemble_params.png")

            # Back to Spike-Train
            page.click("#sz-method-selector label:has-text('Spike-Train')", timeout=3000)
            wait(page, 1000)
        except Exception as e:
            print(f"  ⚠ Method switch failed: {e}")

        # ── 8. Run detection to get results ──────────────
        try:
            scroll_into_view(page, "#sz-detect-btn")
            page.click("#sz-detect-btn", timeout=5000)
            wait(page, 10000)  # Wait for detection to complete
        except Exception as e:
            print(f"  ⚠ Detection run: {e}")

        # ── 9. Detection results ─────────────────────────
        try:
            scroll_into_view(page, "#sz-results-grid")
            wait(page, 1000)
            shot(page, "detection_results.png")
        except Exception:
            shot(page, "detection_results.png")

        # ── 10. Filter controls ──────────────────────────
        try:
            scroll_into_view(page, "#sz-filter-enabled")
            wait(page, 500)
            shot(page, "filter_controls.png")
        except Exception:
            scroll_into_view(page, "#sz-results-grid")
            shot(page, "filter_controls.png")

        # ── 11. Click a result row → event inspector ─────
        try:
            # ag-grid needs a click on a cell to trigger row selection
            # First scroll grid into view
            scroll_into_view(page, "#sz-results-grid")
            wait(page, 500)
            # Click the first data cell in the grid
            cell = page.locator("#sz-results-grid .ag-cell").first
            cell.click(timeout=5000)
            wait(page, 4000)
            # Scroll to show the inspector
            scroll_into_view(page, "#sz-event-inspector")
            wait(page, 1500)
            shot(page, "event_inspector.png")
        except Exception as e:
            print(f"  ⚠ Inspector click: {e}")
            shot(page, "event_inspector.png")

        # ── 12. Export controls ──────────────────────────
        try:
            scroll_into_view(page, "#sz-export-section")
            wait(page, 500)
            shot(page, "export_csv.png")
        except Exception as e:
            print(f"  ⚠ Export section: {e}")
            # Try scrolling to very bottom
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            wait(page, 500)
            shot(page, "export_csv.png")

        # ── 13. Batch progress ───────────────────────────
        # This shows the detect-all / batch controls area
        try:
            scroll_into_view(page, "#sz-detect-btn")
            wait(page, 500)
            shot(page, "batch_progress.png")
        except Exception:
            shot(page, "batch_progress.png")

        # ── 14. Training tab ─────────────────────────────
        click_tab(page, "training_grp")
        wait(page, 3000)
        shot(page, "training_review.png")

        # ── 15. Detection → Interictal Spikes ────────────
        click_tab(page, "detection")
        wait(page, 2000)
        try:
            page.click("text=Interictal Spikes", timeout=5000)
            wait(page, 3000)
            shot(page, "spike_annotation.png")
        except Exception as e:
            print(f"  ⚠ Spike tab: {e}")

        # ── 16. Dataset / Model tab ──────────────────────
        click_tab(page, "ml_grp")
        wait(page, 3000)
        shot(page, "ml_datasets.png")

        # For ml_training_progress, we can't show actual training
        # Take the same view — manual caption explains it
        shot(page, "ml_training_progress.png")

        # ── 17. Analysis tab ─────────────────────────────
        click_tab(page, "analysis")
        wait(page, 2000)
        shot(page, "analysis_tab.png")

        # ── 18. Results tab ──────────────────────────────
        click_tab(page, "results")
        wait(page, 2000)
        shot(page, "results_summary.png")

        browser.close()
        print(f"\nDone! {len(__import__('os').listdir(OUT))} screenshots saved to {OUT}/")


if __name__ == "__main__":
    main()
