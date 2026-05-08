import os
import sqlite3
from pathlib import Path

from flask import Flask


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "SentimentAnalysis.db"


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing_columns = {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing_columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_url TEXT NOT NULL,
                source_type TEXT DEFAULT 'url',
                model_backend TEXT DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_run_id INTEGER NOT NULL,
                comment_text TEXT NOT NULL,
                sentiment_label TEXT NOT NULL,
                confidence_score REAL DEFAULT 0.0,
                FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(id)
            )
            """
        )
        _ensure_column(connection, "analysis_runs", "source_type", "TEXT DEFAULT 'url'")
        _ensure_column(connection, "analysis_runs", "model_backend", "TEXT DEFAULT 'unknown'")
        _ensure_column(connection, "comments", "confidence_score", "REAL DEFAULT 0.0")
        connection.commit()


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "development-secret-key")
    app.config["DATABASE_PATH"] = str(DATABASE_PATH)
    app.config["MAX_COMMENTS"] = int(os.getenv("MAX_COMMENTS", "0"))
    app.config["MAX_REVIEW_PAGES"] = int(os.getenv("MAX_REVIEW_PAGES", "50"))
    app.config["SENTIMENT_MODEL_NAME"] = os.getenv(
        "SENTIMENT_MODEL_NAME",
        "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    )
    app.config["SENTIMENT_MODEL_CANDIDATES"] = [
        item.strip()
        for item in os.getenv(
            "SENTIMENT_MODEL_CANDIDATES",
            app.config["SENTIMENT_MODEL_NAME"],
        ).split(",")
        if item.strip()
    ]

    init_database()

    from app.routes import main_bp

    app.register_blueprint(main_bp)
    return app
