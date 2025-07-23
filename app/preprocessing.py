import json
import os
import re

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "../prompts")
TRIVIAL_PATH = os.path.join(PROMPTS_DIR, "trivial_questions.json")
PROMPT_DINAMICA_PATH = os.path.join(PROMPTS_DIR, "prompt_dinamica.json")

_trivial_data = None
_prompt_data = None

def load_trivial_data():
    global _trivial_data
    if _trivial_data is None:
        with open(TRIVIAL_PATH, "r", encoding="utf-8") as f:
            _trivial_data = json.load(f)
    return _trivial_data

def load_prompt_data():
    global _prompt_data
    if _prompt_data is None:
        with open(PROMPT_DINAMICA_PATH, "r", encoding="utf-8") as f:
            _prompt_data = json.load(f)
    return _prompt_data

def detect_intent(user_input: str) -> str:
    """Devuelve la intención según palabras clave. Se puede extender fácilmente."""

    intent_keywords = {
        "utp": ["utp", "planificación", "pme", "evaluación", "cobertura curricular"],
        "docencia": ["profesor", "clase", "currículo", "estrategia pedagógica"],
        "inspectoría": ["inspectoría", "asistencia", "sanciones", "convivencia"],
        "finanzas": ["faep", "rendición", "presupuesto", "finanzas", "gastos"],
        "direccion": ["director", "liderazgo", "jefatura", "supervisión"],
        "apoderados": ["apoderado", "matrícula", "reunión", "familia", "padres"],
        "pie": ["pie", "inclusión", "nee", "adecuación curricular", "integración"],
        "infraestructura": ["infraestructura", "mantención", "edificio", "seguridad estructural"],
        "asistencia": ["inasistencia", "revinculación", "registro de asistencia"],
        "convivencia": ["convivencia", "acoso", "reglamento", "ciudadanía"]
        }

    input_lower = user_input.lower()
    for intent, keywords in intent_keywords.items():
        if any(k in input_lower for k in keywords):
            return intent

    return "default"

def preprocess_user_input(user_input: str) -> dict:
    """
    Procesa el input y retorna dict con:
        - respuesta_previa: si se debe responder sin pasar al modelo
        - prompt_dinamico: system prompt a usar
        - intencion: detectada
    """
    data = load_trivial_data()
    prompts = load_prompt_data()
    input_clean = user_input.strip().lower()

    # 1. Input muy corto
    if len(input_clean) < 5:
        return {
            "respuesta_previa": data.get("mensaje_corto"),
            "prompt_dinamico": prompts.get("default"),
            "intencion": "default"
        }

    # 2. Saludos simples
    if any(saludo in input_clean for saludo in data.get("saludos", [])):
        return {
            "respuesta_previa": data.get("respuesta_generica"),
            "prompt_dinamico": prompts.get("default"),
            "intencion": "default"
        }

    # 3. Preguntas triviales o genéricas
    if any(p in input_clean for p in data.get("preguntas_generales", [])):
        return {
            "respuesta_previa": data.get("respuesta_generica"),
            "prompt_dinamico": prompts.get("default"),
            "intencion": "default"
        }

    # 4. Detectar intención y retornar prompt dinámico
    intencion = detect_intent(input_clean)
    prompt = prompts.get(intencion, prompts.get("default"))

    return {
        "respuesta_previa": None,
        "prompt_dinamico": prompt,
        "intencion": intencion
    }
