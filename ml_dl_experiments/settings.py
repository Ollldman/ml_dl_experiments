from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from dotenv import load_dotenv

# Загрузка файла конфигураций окружения
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE_PATH)

# Создание валидационного объекта pydantic
class Settings(BaseSettings):
    model_config=SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8")
    SOURCE_PATH: str
    KAGGLE_USERNAME: str
    KAGGLE_KEY: str

settings: Settings = Settings() # type:ignore