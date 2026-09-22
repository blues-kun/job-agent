"""使用虚构简历操作本机工作台并截图；企业名遮罩，不修改页面数据。"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8094")
    parser.add_argument("--output", type=Path, default=Path("docs/screenshots"))
    args = parser.parse_args()
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("仅允许访问本机工作台；不自动向远程发送简历")
    args.output.mkdir(parents=True, exist_ok=True)
    captures, errors = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1440, "height": 1080},
                                      device_scale_factor=1, locale="zh-CN", timezone_id="Asia/Shanghai")
        page = context.new_page()
        page.set_default_timeout(90000)
        page.on("pageerror", lambda error: errors.append(str(error)))

        def capture(name, caption, *, mask=None, element=None):
            page.evaluate("document.fonts.ready")
            path = args.output / name
            (element or page).screenshot(path=str(path), animations="disabled", mask=mask or [], mask_color="#dce5df")
            captures.append({"file": name, "caption": caption,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "viewport": page.viewport_size, "company_masked": bool(mask),
                             "scope": "岗位详情弹窗" if element else "完整视口"})

        page.goto(args.url, wait_until="networkidle")
        page.locator("#metrics .metric").first.wait_for()
        capture("01-demand.png", "真实历史岗位快照的需求概览；不代表当前招聘状态")

        page.locator('[data-view="match"]').click()
        page.locator("#sample").select_option("python-junior")
        page.locator("#recommend").click()
        page.locator("#confirm-profile").wait_for()
        if page.locator("#resume-form .field").evaluate_all("fields => fields.some(field => [...field.querySelectorAll('input,select')].some(input => input.getBoundingClientRect().right > field.getBoundingClientRect().right + 1))"):
            raise RuntimeError("简历表单控件超出网格列")
        page.evaluate("window.scrollTo(0, 0)")
        capture("02-profile.png", "完全虚构的两年 Python 简历，推荐前逐字段确认来源")
        if page.locator("#ack-conflicts").count():
            page.locator("#ack-conflicts").check()
        page.locator("#confirm-profile").click()
        page.locator(".job-card").first.wait_for()
        page.evaluate("window.scrollTo(0, 0)")
        capture("03-matching.png", "虚构简历在真实岗位库上的推荐；企业名称已遮罩",
                mask=[page.locator(".job-company")])

        page.locator("[data-detail]").first.click()
        page.locator("#detail-title").wait_for()
        capture("04-evidence.png", "岗位与简历双侧原文证据；企业名称已遮罩",
                mask=[page.locator(".detail-heading .muted")], element=page.locator("#detail-dialog"))
        page.locator("#close-dialog").click()

        page.locator('[data-view="lab"]').click()
        page.locator("#run-comparison").click()
        page.locator("#comparison-results tbody").wait_for()
        page.evaluate("window.scrollTo(0, 0)")
        capture("05-comparison.png", "相同虚构画像运行三组算法；覆盖率不等于相关性金标")

        page.locator('[data-view="research"]').click()
        page.locator("#research-metrics .metric").first.wait_for()
        page.locator("#research-runs .loading").wait_for(state="detached")
        page.evaluate("window.scrollTo(0, 0)")
        capture("06-research.png", "实际研究与标注进度；模型标注不冒充人工金标，未领取原文任务")

        page.locator('[data-view="demand"]').click()
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("window.scrollTo(0, 0)")
        capture("07-mobile.png", "390 像素宽度的移动端岗位需求概览")
        mobile_overflow = page.evaluate("document.documentElement.scrollWidth > innerWidth")
        health = context.request.get(args.url.rstrip("/") + "/api/v2/health").json()
        context.request.delete(args.url.rstrip("/") + "/api/v2/feedback")
        browser.close()

    manifest = {"captured_at": datetime.now(timezone.utc).isoformat(),
                "method": "Playwright 真实页面交互与浏览器截图，未注入虚构指标或替换接口结果",
                "sample": "python-junior（完全虚构）", "health": health,
                "page_errors": errors, "mobile_overflow": mobile_overflow, "screenshots": captures}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"截图数": len(captures), "页面异常数": len(errors), "移动端横向溢出": mobile_overflow}, ensure_ascii=False))
    if errors or mobile_overflow:
        raise RuntimeError("浏览器验收失败，请检查清单中的异常")


if __name__ == "__main__":
    main()
