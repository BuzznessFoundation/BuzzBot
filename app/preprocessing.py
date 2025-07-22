import json
import os

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
TRIVIAL_PATH = os.path.join(PROMPTS_DIR, "trivial_questions.json")
PROMPT_BASE_PATH = os.path.join(PROMPTS_DIR, "prompt_base.txt")

# Variables globales para cargar solo una vez
_trivial_data = None
_prompt_base = None

def load_trivial_data():
    global _trivial_data
    if _trivial_data is None:
        with open(TRIVIAL_PATH, "r", encoding="utf-8") as f:
            _trivial_data = json.load(f)
    return _trivial_data

def load_prompt_base():
    global _prompt_base
    if _prompt_base is None:
        with open(PROMPT_BASE_PATH, "r", encoding="utf-8") as f:
            _prompt_base = f.read().strip()
    return _prompt_base

def preprocess_user_input(user_input: str) -> str | None:
    user_input_clean = user_input.strip().lower()
    data = load_trivial_data()
    prompt_base = load_prompt_base()

    # Revisar saludos
    for saludo in data.get("saludos", []):
        if saludo in user_input_clean:
            return data["respuestas"].get("saludos", prompt_base)

    # Inputs cortos
    if len(user_input_clean) < 5:
        return data.get("mensaje_corto", "Creo que necesito algo mas de contexto para resumir aquello...")

    # Preguntas triviales BuzzBot
    for pregunta in data.get("preguntas_buzzbot", []):
        if pregunta in user_input_clean:
            return data["respuestas"].get("preguntas_buzzbot", prompt_base)

    return None
