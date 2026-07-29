from pydantic_settings import BaseSettings, SettingsConfigDict


class St(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env",extra="ignore")
    data_path: str
    model_path: str
    field_key_prefix: str
    field_value_prefix: str
    header_prefix: str
    item_prefix: str
    row_tolerance: int
    edge_tolerance: int
    column_tolerance: int
    page_max_distance: int
    ambiguity_k: float
    hf_token: str | None = None
    hf_model_repo_id: str | None = None
    app_env: str

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"
    



settings = St()

__all__ = ['settings']
