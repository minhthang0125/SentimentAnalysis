from dataclasses import dataclass
from typing import Iterable, List


class CrawlerError(Exception):
    pass


@dataclass
class ExtractionConfig:
    max_comments: int = 0


class YouTubeCrawler:
    def __init__(self, max_comments: int = 0) -> None:
        self.config = ExtractionConfig(max_comments=max_comments)

    def extract_comments(self, url: str) -> List[str]:
        try:
            from youtube_comment_downloader import SORT_BY_POPULAR, YoutubeCommentDownloader
        except Exception as exc:
            raise CrawlerError(
                "Khong the tai bo thu thap comment YouTube. Hay kiem tra youtube-comment-downloader."
            ) from exc

        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
        extracted: List[str] = []

        for item in comments:
            text = item.get("text", "").strip()
            if text:
                extracted.append(text)
            if self.config.max_comments > 0 and len(extracted) >= self.config.max_comments:
                break

        return self._deduplicate(extracted)

    def _deduplicate(self, items: Iterable[str]) -> List[str]:
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

        return results
