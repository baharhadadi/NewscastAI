"""
services/api/settings.py
-------------------------
Central configuration for the API service tier.

All values are read from environment variables.  Set them in your .env file
or docker-compose environment block; the defaults below are safe for local dev.
"""

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # --- Database ---
    database_url: Optional[str] = None

    # --- Cache / Queue ---
    redis_url: Optional[str] = None

    # --- Internal service URLs ---
    # Base URL of the internal MCP service.
    mcp_url: str = "http://mcp:7000"

    # --- Object storage (MinIO) ---
    minio_endpoint: str = "minio:9000"
    minio_bucket: str = "newscast-audio"

    # --- Localisation ---
    # IANA timezone string used for scheduling and timestamp display.
    tz: str = "America/Toronto"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
