import base64
import csv
import io
import logging
import sqlite3
from collections import Counter
from functools import lru_cache
from typing import Dict, List
from urllib.parse import urlparse

import matplotlib
from flask import (
    Blueprint,
    Response,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from models.sentiment_model import SentimentAnalyzer
from utils.crawler import CommentCrawler, CrawlerError
from utils.text_cleaner import (
    clean_comment,
    detect_supported_platform,
    is_valid_url,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


main_bp = Blueprint("main", __name__)
LOGGER = logging.getLogger(__name__)


DISPLAY_SENTIMENT_LABEL = {
    "Positive": "Positive",
    "Neutral": "Neural",
    "Negative": "Negative",
}

PLATFORM_SOURCE_TYPES = {
    "youtube": "youtube",
}


def get_db_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(current_app.config["DATABASE_PATH"])
    connection.row_factory = sqlite3.Row
    return connection


@lru_cache(maxsize=4)
def _get_cached_analyzer(model_name: str, model_candidates: tuple[str, ...]) -> SentimentAnalyzer:
    return SentimentAnalyzer(
        model_name=model_name,
        model_candidates=list(model_candidates),
    )


def build_analyzer() -> SentimentAnalyzer:
    return _get_cached_analyzer(
        model_name=current_app.config["SENTIMENT_MODEL_NAME"],
        model_candidates=tuple(current_app.config["SENTIMENT_MODEL_CANDIDATES"]),
    )


def save_analysis(
    source_url: str,
    source_type: str,
    model_backend: str,
    analyzed_comments: List[Dict[str, object]],
) -> int:
    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO analysis_runs (source_url, source_type, model_backend)
            VALUES (?, ?, ?)
            """,
            (source_url, source_type, model_backend),
        )
        analysis_run_id = cursor.lastrowid
        connection.executemany(
            """
            INSERT INTO comments (analysis_run_id, comment_text, sentiment_label, confidence_score)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    analysis_run_id,
                    str(item["comment"]),
                    str(item["sentiment"]),
                    float(item.get("score", 0.0)),
                )
                for item in analyzed_comments
            ],
        )
        connection.commit()
        return int(analysis_run_id)


def fetch_analysis(analysis_run_id: int) -> Dict[str, object] | None:
    with get_db_connection() as connection:
        run = connection.execute(
            "SELECT * FROM analysis_runs WHERE id = ?",
            (analysis_run_id,),
        ).fetchone()
        if run is None:
            return None

        comments = connection.execute(
            """
            SELECT comment_text, sentiment_label, confidence_score
            FROM comments
            WHERE analysis_run_id = ?
            ORDER BY id ASC
            """,
            (analysis_run_id,),
        ).fetchall()

    items = [
        {
            "comment": row["comment_text"],
            "sentiment": row["sentiment_label"],
            "sentiment_display": DISPLAY_SENTIMENT_LABEL.get(
                row["sentiment_label"],
                row["sentiment_label"],
            ),
            "score": round(float(row["confidence_score"] or 0.0), 4),
            "confidence_percent": round(float(row["confidence_score"] or 0.0) * 100, 1),
        }
        for row in comments
    ]
    return {
        "id": run["id"],
        "source_url": run["source_url"],
        "source_type": run["source_type"],
        "model_backend": run["model_backend"],
        "comments": items,
    }


def create_chart(sentiment_counts: Counter) -> str:
    labels = ["Positive", "Neural", "Negative"]
    sizes = [
        sentiment_counts.get("Positive", 0),
        sentiment_counts.get("Neutral", 0),
        sentiment_counts.get("Negative", 0),
    ]
    colors = ["#159a6f", "#d99a33", "#d65151"]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    fig.patch.set_facecolor("#fffdf8")
    if sum(sizes) == 0:
        ax.text(0.5, 0.5, "Khong co du lieu", ha="center", va="center", fontsize=16)
        ax.axis("off")
    else:
        ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
            textprops={"fontsize": 10},
        )
        ax.axis("equal")

    output = io.BytesIO()
    fig.tight_layout()
    fig.savefig(output, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(output.getvalue()).decode("utf-8")


def summarize_comments(comments: List[Dict[str, object]]) -> Dict[str, object]:
    sentiment_counts = Counter(str(item["sentiment"]) for item in comments)
    total_comments = len(comments)
    dominant_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else "Neutral"
    average_confidence = (
        round(sum(float(item["score"]) for item in comments) / total_comments * 100, 1)
        if total_comments
        else 0.0
    )
    return {
        "total_comments": total_comments,
        "positive_count": sentiment_counts.get("Positive", 0),
        "neutral_count": sentiment_counts.get("Neutral", 0),
        "negative_count": sentiment_counts.get("Negative", 0),
        "dominant_sentiment": dominant_sentiment,
        "dominant_sentiment_display": DISPLAY_SENTIMENT_LABEL.get(
            dominant_sentiment,
            dominant_sentiment,
        ),
        "average_confidence": average_confidence,
        "chart_data": create_chart(sentiment_counts),
    }


def _analyze_and_store_comments(source_url: str, source_type: str, comments: List[str]) -> int:
    cleaned_comments = []
    for comment in comments:
        cleaned = clean_comment(comment)
        if cleaned:
            cleaned_comments.append(cleaned)

    if not cleaned_comments:
        raise ValueError("Khong tim thay binh luan hop le de phan tich.")

    analyzer = build_analyzer()
    analyzed_comments = analyzer.analyze_comments(cleaned_comments)
    analysis_run_id = save_analysis(
        source_url=source_url,
        source_type=source_type,
        model_backend=analyzer.backend_name,
        analyzed_comments=analyzed_comments,
    )
    session["last_analysis_run_id"] = analysis_run_id
    return analysis_run_id


def _extract_comments_by_platform(source_url: str, platform_name: str) -> List[str]:
    crawler = CommentCrawler(
        max_comments=current_app.config["MAX_COMMENTS"],
        max_pages=current_app.config["MAX_REVIEW_PAGES"],
    )

    if platform_name == "youtube":
        return list(crawler.extract_comments(source_url))

    raise CrawlerError("Nen tang chua duoc ho tro.")


@main_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@main_bp.route("/analyze", methods=["POST"])
def analyze():
    source_url = request.form.get("url", "").strip()

    if not source_url:
        flash("Vui long nhap URL YouTube", "error")
        return redirect(url_for("main.index"))

    if not is_valid_url(source_url):
        flash("URL khong hop le", "error")
        return redirect(url_for("main.index"))

    platform_name = detect_supported_platform(source_url)
    if platform_name != "youtube":
        flash("URL khong hop le", "error")
        return redirect(url_for("main.index"))

    try:
        raw_comments = _extract_comments_by_platform(source_url, platform_name)
        analysis_run_id = _analyze_and_store_comments(
            source_url=source_url,
            source_type=PLATFORM_SOURCE_TYPES.get(platform_name, "url"),
            comments=raw_comments,
        )
    except CrawlerError as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.index"))
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.index"))
    except Exception as exc:
        LOGGER.exception("Unexpected error while processing URL: %s", source_url)
        flash(f"Da xay ra loi khi xu ly URL: {exc}", "error")
        return redirect(url_for("main.index"))

    return redirect(url_for("main.result", analysis_run_id=analysis_run_id))


@main_bp.route("/analyze-text", methods=["POST"])
def analyze_text():
    manual_comment = request.form.get("manual_comment", "").strip()
    if not manual_comment:
        flash("Vui long nhap it nhat mot binh luan de AI phan tich.", "error")
        return redirect(url_for("main.index"))

    comments = [line.strip() for line in manual_comment.splitlines() if line.strip()]
    try:
        analysis_run_id = _analyze_and_store_comments(
            source_url="Nhap thu cong",
            source_type="manual",
            comments=comments,
        )
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.index"))

    return redirect(url_for("main.result", analysis_run_id=analysis_run_id))


@main_bp.route("/result", methods=["GET"])
def result():
    analysis_run_id = request.args.get("analysis_run_id", type=int)
    if analysis_run_id is None:
        analysis_run_id = session.get("last_analysis_run_id")

    if analysis_run_id is None:
        flash("Chua co ket qua phan tich. Vui long chay phan tich moi.", "error")
        return redirect(url_for("main.index"))

    analysis = fetch_analysis(analysis_run_id)
    if analysis is None:
        flash("Khong tim thay ket qua phan tich.", "error")
        return redirect(url_for("main.index"))

    summary = summarize_comments(analysis["comments"])
    domain = (
        "Nhap thu cong"
        if analysis["source_type"] == "manual"
        else urlparse(str(analysis["source_url"])).netloc
    )
    return render_template(
        "result.html",
        analysis=analysis,
        summary=summary,
        domain=domain,
    )


@main_bp.route("/export/<int:analysis_run_id>", methods=["GET"])
def export_csv(analysis_run_id: int):
    analysis = fetch_analysis(analysis_run_id)
    if analysis is None:
        flash("Khong tim thay ket qua phan tich.", "error")
        return redirect(url_for("main.index"))

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Nguon", "Binh luan", "Cam xuc", "Do tin cay"])
    for item in analysis["comments"]:
        writer.writerow(
            [
                analysis["source_url"],
                item["comment"],
                item["sentiment_display"],
                f'{item["confidence_percent"]}%',
            ]
        )

    csv_data = buffer.getvalue().encode("utf-8-sig")
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={
            "Content-Disposition": (
                f"attachment; filename=ket_qua_phan_tich_cam_xuc_{analysis_run_id}.csv"
            )
        },
    )
