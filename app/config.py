import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_STORE_DIR = BASE_DIR / "vector_store"
PROMPTS_DIR = BASE_DIR / "prompts"
CONFIG_DIR = BASE_DIR / "config"

SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system_prompt.txt"
TRIVIAL_QUESTIONS_FILE = CONFIG_DIR / "trivial_questions.json"
MODEL_PATH = VECTOR_STORE_DIR / "models" / "phi-3-mini-128k-instruct.Q4_K_M.gguf"
