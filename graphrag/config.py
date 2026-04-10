from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # Neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:9b"
    ollama_embedding_model: str = "nomic-embed-text"

    # Processing
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 5

    # Configuración de Pydantic V2: ignora variables en el .env que no estén en esta clase
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

class GeminiSettings(BaseSettings):
    # Neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Gemini
    # Pydantic buscará automáticamente 'GEMINI_API_KEY' (ignorando mayúsculas/minúsculas)
    gemini_api_key: str 
    
    # Si en tu .env la variable se llama GOOGLE_MODEL pero en código quieres usar gemini_model, 
    # usamos un alias de validación en lugar de os.getenv()
    gemini_model: str = Field(validation_alias="GEMINI_MODEL")

    # Processing
    chunk_size: int = 800
    chunk_overlap: int = 80
    top_k_results: int = 5

    # La clave del éxito: extra="ignore" evita el error con las variables de Ollama
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    return Settings()

@lru_cache()
def get_gemini_settings() -> GeminiSettings:
    return GeminiSettings()