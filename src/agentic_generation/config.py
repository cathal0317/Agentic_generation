"""One set of database settings shared by the application and migrations."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    postgres_host: str = Field(default="127.0.0.1", min_length=1)
    postgres_port: int = Field(default=5433, ge=1, le=65535)
    postgres_db: str = Field(default="agentic_generation", min_length=1)
    postgres_user: str = Field(default="agentic", min_length=1)
    postgres_password: SecretStr = Field(min_length=1)

    @property
    def database_url(self) -> URL:
        # URL.create preserves special characters without manual URL escaping.
        return URL.create(
            "postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )
