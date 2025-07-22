from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import traceback

from sentence_transformers import SentenceTransformer

from app.utils import (
    cargar_modelo_local,
    cargar_modelo_gemini,
    crear_respuesta,
    get_system_prompt,
    es_pregunta_trivial,
    USE_GEMINI,
)
from app.rag import (
    reordenar_chunks,
    cargar_vector_store,
    construir_contexto_por_tokens  # si lo usas
)

# Inicializar FastAPI
app = FastAPI(title="BuzzBot API")

# CORS (ajustable según el frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar modelo (local o Gemini)
try:
    llm = cargar_modelo_gemini() if USE_GEMINI else cargar_modelo_local()
    N_CTX = 8192 if USE_GEMINI else 4096
    logger.info(f"✅ Modelo cargado correctamente. USE_GEMINI={USE_GEMINI}")
except Exception as e:
    logger.error(f"❌ Error al cargar el modelo: {traceback.format_exc()}")
    llm = None
    N_CTX = 0

# Cargar vector store
try:
    index, chunks, metadata = cargar_vector_store()
    logger.info(f"✅ Vector store cargado correctamente con {len(chunks)} chunks")
except Exception as e:
    logger.error(f"❌ Error al cargar vector store: {traceback.format_exc()}")
    index = None
    chunks = []
    metadata = []

# Cargar embedder
try:
    embedder = SentenceTransformer("all-mpnet-base-v2")
    logger.info("✅ Embedder cargado correctamente")
except Exception as e:
    logger.error(f"❌ Error al cargar embedder: {traceback.format_exc()}")
    embedder = None

# Función de conteo de tokens
def contar_tokens(texto: str) -> int:
    if USE_GEMINI:
        return len(texto) // 4
    try:
        return len(llm.tokenize(texto.encode("utf-8")))
    except Exception as e:
        logger.error(f"⚠️ Error tokenizando texto: {traceback.format_exc()}")
        return 0

# Ruta principal del chat
@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        pregunta = data.get("prompt", "").strip()
        if not pregunta:
            raise HTTPException(status_code=400, detail="Prompt vacío")

        logger.info(f"📥 Pregunta recibida: {pregunta}")

        # Modo eco si faltan componentes
        if llm is None:
            return {"respuesta": f"(modo eco: modelo no cargado) {pregunta}"}
        if index is None:
            return {"respuesta": f"(modo eco: vector store no cargado) {pregunta}"}
        if embedder is None:
            return {"respuesta": f"(modo eco: embedder no cargado) {pregunta}"}

        # Manejo de preguntas triviales
        if es_pregunta_trivial(pregunta):
            logger.info("🤖 Pregunta trivial detectada")
            return {"respuesta": "Hola, soy BuzzBot, tu asistente en temas educativos. ¿Podrías especificar mejor qué necesitas?"}

        logger.info("🔎 Generando embedding...")
        embedding = embedder.encode([pregunta], normalize_embeddings=True)

        logger.info("📚 Buscando en FAISS index...")
        _, I = index.search(embedding, 15)
        indices = I[0]

        logger.info("🧠 Reordenando fragmentos relevantes...")
        fragmentos = reordenar_chunks(chunks, metadata, indices, top_k=10)

        logger.info("🧮 Construyendo contexto limitado por tokens...")
        max_context_tokens = N_CTX - 512
        contexto = ""
        total_tokens = 0
        for frag in fragmentos:
            tokens = contar_tokens(frag)
            if total_tokens + tokens > max_context_tokens:
                break
            contexto += frag + "\n\n"
            total_tokens += tokens

        logger.info(f"📏 Total tokens de contexto: {total_tokens}")

        system_prompt = get_system_prompt()
        prompt = (
            f"Contextos disponibles:\n{contexto}\n\n"
            f"Pregunta:\n{pregunta}\n\n"
            f"Respuesta clara y precisa:"
        )

        logger.info("🧠 Llamando al modelo para generar respuesta...")
        respuesta = crear_respuesta(llm, prompt=prompt, system_msg=system_prompt)

        logger.info("✅ Respuesta generada correctamente")
        return {"respuesta": respuesta}

    except Exception as e:
        logger.error(f"❌ Error en /chat:\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": "Error interno del servidor"})

# Ruta simple de verificación
@app.get("/")
async def root():
    return {
        "status": "online",
        "modelo": "gemini" if USE_GEMINI else "local",
        "llm": llm is not None,
        "vector_store": index is not None,
        "chunks": len(chunks),
    }
