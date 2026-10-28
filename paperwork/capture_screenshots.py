from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    b=p.chromium.launch(channel="chrome",headless=True)
    pg=b.new_page(viewport={"width":1440,"height":900},device_scale_factor=1.5)
    pg.goto("http://localhost:8766/",wait_until="networkidle",timeout=60000)
    time.sleep(3)
    pg.screenshot(path="ui_dashboard.png")
    # click a tehsil
    try:
        pg.locator("svg#tehsil-map .is-pilot").nth(0).click(); time.sleep(2)
    except Exception as e: print("click",e)
    pg.screenshot(path="ui_dashboard_detail.png")
    for v in ["sandbox","reports","sources","howitworks"]:
        pg.click(f'button.navitem[data-view="{v}"]'); time.sleep(3)
        pg.screenshot(path=f"ui_{v}.png")
    b.close()
