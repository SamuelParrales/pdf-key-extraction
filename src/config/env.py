from pydantic_settings import BaseSettings, SettingsConfigDict


class St(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env",extra="ignore")
    data_path: str

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"
    



settings = St()

__all__ = ['settings']