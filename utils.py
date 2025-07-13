from llama_cpp import Llama
import os

def cargar_modelo():
    modelo_path = os.path.join("models", "tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf")

    if not os.path.exists(modelo_path):
        raise FileNotFoundError(f"No se encontró el modelo en {modelo_path}")

    llm = Llama(
        model_path=modelo_path,
        n_ctx=2048,
        n_threads=1,        # 🔽 uso mínimo de CPU
        chat_format="chatml"
    )
    return llm
