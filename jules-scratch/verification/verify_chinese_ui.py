from playwright.sync_api import sync_playwright
import time

def run(playwright):
    app = playwright.tk.launch_persistent_context("")
    page = app.main_window()
    page.locator('ttk::combobox').click()
    page.get_by_role('option', name='中文').click()
    time.sleep(2)
    page.screenshot(path="jules-scratch/verification/verification.png")
    app.close()

with sync_playwright() as playwright:
    run(playwright)
