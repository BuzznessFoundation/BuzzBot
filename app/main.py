from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from sentence_transformers import SentenceTransformer

from app.utils import (
    cargar_modelo_local,
    cargar_modelo_gemini,
    crear_respuesta,
    get_system_prompt,
    es_pregunta_trivial,
    USE_GEMINI,
)
from app.rag import cargar_vector_store, construir_contexto_por_tokens

app = FastAPI(title="BuzzBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    llm = cargar_modelo_gemini() if USE_GEMINI else cargar_modelo_local()
    N_CTX = 8192 if USE_GEMINI else 4096
    logger.info(f"Modelo cargado correctamente. USE_GEMINI={USE_GEMINI}")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}")
    llm = None
    N_CTX = 0

try:
    index, chunks, metadata = cargar_vector_store()
    logger.info("Vector store cargado correctamente")
except Exception as e:
    logger.error(f"Error al cargar vector store: {e}")
    index = None
    chunks = []
    metadata = []

embedder = SentenceTransformer("all-mpnet-base-v2")

def contar_tokens(texto: str) -> int:
    if USE_GEMINI:
        return len(texto) // 4
    try:
        return len(llm.tokenize(texto.encode("utf-8")))
    except Exception as e:
        logger.error(f"Error tokenizando texto: {e}")
        return 0

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        pregunta = data.get("prompt", "").strip()
        if not pregunta:
            raise HTTPException(status_code=400, detail="Prompt vacío")

        if llm is None or index is None:
            return {"respuesta": f"(modo eco) {pregunta}"}

        if es_pregunta_trivial(pregunta):
            return {"respuesta": "Hola! ¿En qué puedo ayudarte con la administración escolar?"}

        embedding = embedder.encode([pregunta], normalize_embeddings=True)
        _, I = index.search(embedding, 5)
        indices = I[0]

        max_context_tokens = N_CTX - 512
        contexto = construir_contexto_por_tokens(
            chunks, metadata, indices, contar_tokens, max_context_tokens
        )

        system_prompt = get_system_prompt()
        prompt = (
            f"Contextos disponibles:\n{contexto}\n\n"
            f"Pregunta:\n{pregunta}\n\n"
            f"Respuesta clara y precisa:"
        )

        respuesta = crear_respuesta(llm, prompt=prompt, system_msg=system_prompt)
        return {"respuesta": respuesta}

    except Exception as e:
        logger.error(f"Error en /chat: {e}")
        return JSONResponse(status_code=500, content={"error": "Error interno del servidor"})

@app.get("/")
async def root():
    return {"status": "online", "message": "BuzzBot API lista para responder"}
