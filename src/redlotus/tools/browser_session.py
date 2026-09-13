from __future__ import annotations

import asyncio
from functools import wraps

from redlotus.config.app_config import get_env
from redlotus.infra.path_sandbox import resolve_readable_path


def page_action(operation):
    """Serialize page actions and report browser failures to the Agent."""

    @wraps(operation)
    async def run(self, *args, **kwargs):
        async with self._lock:
            try:
                await self._start()
                return await operation(self, *args, **kwargs)
            except (ImportError, RuntimeError) as exc:
                return f"Error: Browser unavailable: {exc}"
            except self._browser_error as exc:
                return f"Error: {operation.__name__}: {exc}"

    return run


class PlaywrightBrowserSession:
    """A lazy browser owned and closed by the Agent's event loop."""

    def __init__(self, workspace):
        self.workspace = workspace
        self._lock = asyncio.Lock()
        self._playwright = self._browser = self._page = None
        self._browser_error = ()

    async def _start(self):
        if self._page is not None:
            return
        from playwright.async_api import Error, async_playwright

        self._browser_error = Error
        self._playwright = await async_playwright().start()
        try:
            headless = (get_env("BROWSER_HEADLESS", warn=False) or "").lower() not in (
                "0",
                "false",
                "no",
            )
            self._browser = await self._playwright.chromium.launch(headless=headless)
            self._page = await self._browser.new_page(
                viewport={"width": 1280, "height": 720}, locale="zh-CN"
            )
            self._page.set_default_timeout(30_000)
        except BaseException:
            await self._close()
            raise

    async def _close(self):
        try:
            if self._browser is not None:
                await self._browser.close()
        finally:
            if self._playwright is not None:
                await self._playwright.stop()
            self._playwright = self._browser = self._page = None

    async def close(self):
        async with self._lock:
            await self._close()

    @page_action
    async def browser_navigate(
        self, url: str, wait_until: str = "domcontentloaded"
    ) -> str:
        """Open a URL in Chromium; wait_until accepts domcontentloaded, load or networkidle.

        Requires playwright and Chromium. Set BROWSER_HEADLESS=0 to show a window.
        """
        await self._page.goto(url, wait_until=wait_until, timeout=60_000)
        return f"OK\nURL: {self._page.url}\nTitle: {await self._page.title()}"

    @page_action
    async def browser_get_content(self) -> str:
        """Read all visible page text, including dynamically rendered content."""
        text = await self._page.locator("body").inner_text()
        return f"URL: {self._page.url}\n{text}"

    @page_action
    async def browser_screenshot(self, name: str, full_page: bool = False) -> str:
        """Save a screenshot to a project path; full_page includes the scrollable page."""
        path = resolve_readable_path(name, work_base=self.workspace.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        await self._page.screenshot(path=str(path), full_page=full_page)
        return f"Screenshot saved: {path}"

    @page_action
    async def browser_click(self, selector: str) -> str:
        """Click an element using a Playwright CSS or text selector."""
        await self._page.click(selector)
        return f"Clicked: {selector}"

    @page_action
    async def browser_fill(self, selector: str, text: str) -> str:
        """Replace an input element's text using a Playwright selector."""
        await self._page.fill(selector, text)
        return f"Filled: {selector}"

    @page_action
    async def browser_press_key(self, key: str) -> str:
        """Press a Playwright keyboard key, such as Enter, Tab or ArrowDown."""
        await self._page.keyboard.press(key)
        return f"Pressed: {key}"

    @page_action
    async def browser_wait_for_selector(
        self, selector: str, timeout_ms: int = 30_000
    ) -> str:
        """Wait until an element appears in the page."""
        await self._page.wait_for_selector(selector, timeout=timeout_ms)
        return f"Visible: {selector}"

    @page_action
    async def browser_evaluate(self, javascript_expression: str) -> str:
        """Evaluate JavaScript in the current page and return its result."""
        return repr(await self._page.evaluate(javascript_expression))

    async def browser_close(self) -> str:
        """Release this browser; the next browser action starts a new session."""
        await self.close()
        return "Browser closed"

    @property
    def tools(self):
        return [
            self.browser_navigate,
            self.browser_get_content,
            self.browser_screenshot,
            self.browser_click,
            self.browser_fill,
            self.browser_press_key,
            self.browser_wait_for_selector,
            self.browser_evaluate,
            self.browser_close,
        ]
