import os
from dotenv import load_dotenv

# Carga las variables de entorno
load_dotenv()

USE_GEMINI = os.getenv("USE_GEMINI", "False") == "True"

if USE_GEMINI:
    import google.generativeai as genai
else:
    from llama_cpp import Llama

# Inicializa el modelo (local o Gemini)
def cargar_modelo():
    if USE_GEMINI:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY no está definido")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
        return model

    else:
        modelo_path = os.path.join("models", "phi-3-mini-128k-instruct.Q4_K_M.gguf")
        if not os.path.exists(modelo_path):
            raise FileNotFoundError(f"No se encontró el modelo en {modelo_path}")

        llm = Llama(
            model_path=modelo_path,
            n_ctx=4096,
            n_threads=22,
            chat_format="chatml"
        )
        return llm

# Crear respuesta según modelo
def crear_respuesta(llm, prompt):
    if USE_GEMINI:
        response = llm.generate_content(prompt)
        return response.text.strip()
    else:
        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256
        )
        return output["choices"][0]["message"]["content"].strip()
