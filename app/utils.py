import os
import json
import logging
from google.generativeai import GenerativeModel, configure as configure_genai
from llama_cpp import Llama
from app.config import SYSTEM_PROMPT_FILE, TRIVIAL_QUESTIONS_FILE, VECTOR_STORE_DIR, MODEL_PATH

logger = logging.getLogger(__name__)

USE_GEMINI = os.getenv("USE_GEMINI", "False").lower() == "true"
API_KEY = os.getenv("API_KEY")
MODEL_VARIANT = os.getenv("MODEL_VARIANT", "gemini-1.5-flash")

def get_system_prompt() -> str:
    if SYSTEM_PROMPT_FILE.exists():
        return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    return "Eres un asistente experto en administración de escuelas públicas en Chile."

def cargar_preguntas_triviales() -> list[str]:
    if TRIVIAL_QUESTIONS_FILE.exists():
        try:
            return json.loads(TRIVIAL_QUESTIONS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.error(f"Error leyendo trivial_questions.json: {e}")
            return []
    return []

def es_pregunta_trivial(pregunta: str) -> bool:
    trivias = cargar_preguntas_triviales()
    pregunta_lower = pregunta.lower()
    return any(t in pregunta_lower for t in trivias)

def cargar_modelo_local():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
    return Llama(
        model_path=str(MODEL_PATH),
        n_ctx=4096,
        n_threads=22,
        chat_format="chatml"
    )

def cargar_modelo_gemini():
    if not API_KEY:
        raise ValueError("API_KEY no definida en variables de entorno")
    configure_genai(api_key=API_KEY)
    return GenerativeModel(model_name=MODEL_VARIANT)

def crear_respuesta(llm, prompt: str, system_msg: str = "", max_tokens=512):
    if USE_GEMINI:
        try:
            response = llm.generate_content(
                contents=[{
                    "role": "user",
                    "parts": [f"{system_msg}\n\n{prompt}"]
                }],
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.2
                }
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"⚠️ Error al usar Gemini: {e}")
    else:
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
        )
        return output["choices"][0]["message"]["content"]
