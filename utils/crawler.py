from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}


class CrawlerError(Exception):
    pass


@dataclass
class ExtractionConfig:
    max_comments: int = 0
    max_pages: int = 50
    timeout_seconds: int = 20


class BaseCrawler:
    def __init__(self, config: ExtractionConfig) -> None:
        self.config = config

    def extract(self, url: str) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def _deduplicate(items: Iterable[str], max_comments: int) -> List[str]:
        seen = set()
        results = []
        for item in items:
            normalized = " ".join(item.split())
            if len(normalized) < 2:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            results.append(normalized)
            if max_comments > 0 and len(results) >= max_comments:
                break
        return results

    def _limit_reached(self, results: List[str]) -> bool:
        return self.config.max_comments > 0 and len(results) >= self.config.max_comments


class YouTubeCrawler(BaseCrawler):
    def extract(self, url: str) -> List[str]:
        try:
            from youtube_comment_downloader import SORT_BY_POPULAR, YoutubeCommentDownloader
        except Exception as exc:
            raise CrawlerError(
                "YouTube-specific crawling requires youtube-comment-downloader."
            ) from exc

        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
        extracted = []
        for item in comments:
            text = item.get("text", "").strip()
            if text:
                extracted.append(text)
            if self._limit_reached(extracted):
                break
        return self._deduplicate(extracted, self.config.max_comments)


class StaticPageCrawler(BaseCrawler):
    COMMENT_SELECTORS = (
        "[class*='comment']",
        "[id*='comment']",
        "[data-testid*='comment']",
        "[itemprop='reviewBody']",
        "[class*='review']",
        "[class*='content']",
    )

    def extract(self, url: str) -> List[str]:
        try:
            response = requests.get(
                url,
                headers=DEFAULT_HEADERS,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise CrawlerError(f"Failed to fetch URL: {exc}") from exc

        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        results.extend(self._extract_from_selectors(soup))
        results.extend(self._extract_from_json_ld(soup))
        return self._deduplicate(results, self.config.max_comments)

    def _extract_from_selectors(self, soup: BeautifulSoup) -> List[str]:
        comments: List[str] = []
        for selector in self.COMMENT_SELECTORS:
            for element in soup.select(selector):
                text = element.get_text(" ", strip=True)
                if 5 <= len(text) <= 500:
                    comments.append(text)
        return comments


class ShopeeCrawler(BaseCrawler):
    def extract(self, url: str) -> List[str]:
        try:
            static_comments = StaticPageCrawler(self.config).extract(url)
            if static_comments:
                return static_comments
        except CrawlerError:
            pass
        return SeleniumCrawler(self.config).extract(url)

    def _extract_from_json_ld(self, soup: BeautifulSoup) -> List[str]:
        comments: List[str] = []
        for script in soup.select("script[type='application/ld+json']"):
            raw = script.string or script.get_text(strip=True)
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            comments.extend(self._walk_json(payload))
        return comments

    def _walk_json(self, payload) -> List[str]:
        comments: List[str] = []
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key.lower() in {"reviewbody", "description", "comment", "text"}:
                    if isinstance(value, str) and 5 <= len(value) <= 500:
                        comments.append(value)
                else:
                    comments.extend(self._walk_json(value))
        elif isinstance(payload, list):
            for item in payload:
                comments.extend(self._walk_json(item))
        return comments


class SeleniumCrawler(BaseCrawler):
    PAGINATION_CSS_SELECTORS = (
        "button[aria-label*='next' i]",
        "button[aria-label*='sau' i]",
        "a[aria-label*='next' i]",
        "a[aria-label*='sau' i]",
        "button.next",
        "a.next",
        "li.next button",
        "li.next a",
        "[class*='pagination'] button",
        "[class*='pagination'] a",
        "[class*='pager'] button",
        "[class*='pager'] a",
    )

    def extract(self, url: str) -> List[str]:
        try:
            from selenium import webdriver
            from selenium.common.exceptions import WebDriverException
            from selenium.webdriver import ActionChains
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.support.ui import WebDriverWait
        except Exception as exc:
            raise CrawlerError("Selenium is not installed or unavailable.") from exc

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1600,1400")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = None
        try:
            driver = webdriver.Chrome(options=options)
            driver.get(url)
            wait = WebDriverWait(driver, self.config.timeout_seconds)
            wait.until(lambda browser: browser.execute_script("return document.readyState") == "complete")
            time.sleep(2)

            static_crawler = StaticPageCrawler(self.config)
            action_chains = ActionChains(driver)
            collected: List[str] = []
            page_number = 1

            while page_number <= self.config.max_pages:
                self._scroll_review_area(driver)
                soup = BeautifulSoup(driver.page_source, "html.parser")
                page_comments = static_crawler._extract_from_selectors(soup)
                if not page_comments:
                    page_comments = static_crawler._extract_from_json_ld(soup)

                before_count = len(collected)
                merged = collected + page_comments
                collected = self._deduplicate(merged, self.config.max_comments)

                if self._limit_reached(collected):
                    break

                moved = self._go_to_next_review_page(
                    driver=driver,
                    wait=wait,
                    action_chains=action_chains,
                    current_page=page_number,
                    previous_count=before_count,
                )
                if not moved:
                    break
                page_number += 1

            return collected
        except WebDriverException as exc:
            raise CrawlerError(
                "Dynamic page crawling failed. Ensure Chrome and ChromeDriver are available."
            ) from exc
        finally:
            if driver is not None:
                driver.quit()

    def _scroll_review_area(self, driver) -> None:
        last_height = 0
        for _ in range(6):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.2)
            height = driver.execute_script("return document.body.scrollHeight")
            if height == last_height:
                break
            last_height = height

        review_keywords = [
            "đánh giá",
            "nhận xét",
            "review",
            "comments",
            "ratings",
            "customer reviews",
        ]
        xpath = " | ".join(
            [
                f"//*[contains(translate(normalize-space(.), "
                f"'ABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-ỹ', "
                f"'abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúăđĩũơưạ-ỹ'), '{keyword}')]"
                for keyword in review_keywords
            ]
        )

        try:
            elements = driver.find_elements("xpath", xpath)
        except Exception:
            elements = []

        if elements:
            driver.execute_script(
                "arguments[0].scrollIntoView({behavior: 'instant', block: 'center'});",
                elements[0],
            )
            time.sleep(1)

    def _go_to_next_review_page(
        self,
        driver,
        wait,
        action_chains,
        current_page: int,
        previous_count: int,
    ) -> bool:
        next_targets = self._find_next_pagination_targets(driver, current_page)
        for target in next_targets:
            try:
                if not target.is_displayed() or not target.is_enabled():
                    continue
                driver.execute_script(
                    "arguments[0].scrollIntoView({behavior: 'instant', block: 'center'});",
                    target,
                )
                time.sleep(0.6)
                try:
                    target.click()
                except Exception:
                    action_chains.move_to_element(target).click(target).perform()

                time.sleep(1.5)
                wait.until(lambda browser: browser.execute_script("return document.readyState") == "complete")
                time.sleep(1.2)
                self._scroll_review_area(driver)
                if self._page_changed(driver, current_page, previous_count):
                    return True
            except Exception:
                continue
        return False

    def _find_next_pagination_targets(self, driver, current_page: int):
        targets = []

        for css_selector in self.PAGINATION_CSS_SELECTORS:
            try:
                targets.extend(driver.find_elements("css selector", css_selector))
            except Exception:
                continue

        xpath_candidates = [
            f"//a[normalize-space()='{current_page + 1}']",
            f"//button[normalize-space()='{current_page + 1}']",
            "//a[@rel='next']",
            "//button[contains(@aria-label, 'Next') or contains(@aria-label, 'next')]",
            "//a[contains(@aria-label, 'Next') or contains(@aria-label, 'next')]",
            "//button[contains(normalize-space(.), '>')]",
            "//a[contains(normalize-space(.), '>')]",
            "//button[contains(normalize-space(.), '›')]",
            "//a[contains(normalize-space(.), '›')]",
            "//button[contains(normalize-space(.), '→')]",
            "//a[contains(normalize-space(.), '→')]",
            "//button[contains(normalize-space(.), 'Sau')]",
            "//a[contains(normalize-space(.), 'Sau')]",
        ]
        for xpath in xpath_candidates:
            try:
                targets.extend(driver.find_elements(By.XPATH, xpath))
            except Exception:
                continue

        deduped = []
        seen = set()
        for element in targets:
            try:
                key = (
                    element.tag_name,
                    element.text.strip(),
                    element.get_attribute("aria-label") or "",
                    element.location.get("x"),
                    element.location.get("y"),
                )
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            deduped.append(element)
        return deduped

    def _page_changed(self, driver, previous_page: int, previous_count: int) -> bool:
        soup = BeautifulSoup(driver.page_source, "html.parser")
        comments = StaticPageCrawler(self.config)._extract_from_selectors(soup)
        if len(comments) > previous_count:
            return True

        page_markers = [
            f">{previous_page + 1}<",
            f'current">{previous_page + 1}<',
            f'active">{previous_page + 1}<',
            f'aria-current="page">{previous_page + 1}<',
        ]
        html = driver.page_source
        return any(marker in html for marker in page_markers)


class CommentCrawler:
    def __init__(
        self,
        max_comments: int = 0,
        max_pages: int = 50,
        timeout_seconds: int = 20,
    ) -> None:
        self.config = ExtractionConfig(
            max_comments=max_comments,
            max_pages=max_pages,
            timeout_seconds=timeout_seconds,
        )
        self.static_crawler = StaticPageCrawler(self.config)
        self.dynamic_crawler = SeleniumCrawler(self.config)
        self.domain_crawlers = {
            "youtube.com": YouTubeCrawler(self.config),
            "www.youtube.com": YouTubeCrawler(self.config),
            "youtu.be": YouTubeCrawler(self.config),
            "m.youtube.com": YouTubeCrawler(self.config),
            "shopee.vn": ShopeeCrawler(self.config),
            "www.shopee.vn": ShopeeCrawler(self.config),
            "shopee.com": ShopeeCrawler(self.config),
            "www.shopee.com": ShopeeCrawler(self.config),
            "mall.shopee.vn": ShopeeCrawler(self.config),
        }

    def extract_comments(self, url: str) -> Sequence[str]:
        domain = urlparse(url).netloc.lower()
        if domain in self.domain_crawlers:
            comments = self.domain_crawlers[domain].extract(url)
            if comments:
                return comments

        comments = self.static_crawler.extract(url)
        if comments:
            return comments

        return self.dynamic_crawler.extract(url)
