from fastapi import FastAPI, Request, HTTPException, Depends
from sentence_transformers import SentenceTransformer
from app.preprocessing import preprocess_user_input
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import traceback

from app.auth import get_api_key
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
    construir_contexto_por_tokens
)

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
    N_CTX = 16384 if USE_GEMINI else 4096
    logger.info(f"✅ Modelo cargado correctamente. USE_GEMINI={USE_GEMINI}")
except Exception as e:
    logger.error(f"❌ Error al cargar el modelo: {traceback.format_exc()}")
    llm = None
    N_CTX = 0

try:
    index, chunks, metadata = cargar_vector_store()
    logger.info(f"✅ Vector store cargado correctamente con {len(chunks)} chunks")
except Exception as e:
    logger.error(f"❌ Error al cargar vector store: {traceback.format_exc()}")
    index = None
    chunks = []
    metadata = []

try:
    embedder = SentenceTransformer("all-mpnet-base-v2")
    logger.info("✅ Embedder cargado correctamente")
except Exception as e:
    logger.error(f"❌ Error al cargar embedder: {traceback.format_exc()}")
    embedder = None

def contar_tokens(texto: str) -> int:
    if USE_GEMINI:
        return len(texto) // 4
    try:
        return len(llm.tokenize(texto.encode("utf-8")))
    except Exception as e:
        logger.error(f"⚠️ Error tokenizando texto: {traceback.format_exc()}")
        return 0

@app.post("/chat")
async def chat(request: Request, api_key: str = Depends(get_api_key)):

    try:
        data = await request.json()
        pregunta = data.get("prompt", "").strip()
        if not pregunta:
            raise HTTPException(status_code=400, detail="Prompt vacío")

        logger.info(f"📥 Pregunta recibida: {pregunta}")

        if llm is None:
            return {"respuesta": f"(modo eco: modelo no cargado) {pregunta}"}
        if index is None:
            return {"respuesta": f"(modo eco: vector store no cargado) {pregunta}"}
        if embedder is None:
            return {"respuesta": f"(modo eco: embedder no cargado) {pregunta}"}

        logger.info("🧪 Ejecutando preprocesamiento...")
        preproc_result = preprocess_user_input(pregunta)

        if preproc_result["respuesta_previa"]:
            logger.info("🤖 Intercepción por preprocesamiento (respuesta previa)")
            return {"respuesta": preproc_result["respuesta_previa"]}

        system_prompt = preproc_result["prompt_dinamico"]
        logger.info(f"📌 Prompt dinámico aplicado según intención '{preproc_result['intencion']}'")

        logger.info("🔎 Generando embedding...")
        embedding = embedder.encode([pregunta], normalize_embeddings=True)

        logger.info("📚 Buscando en FAISS index...")
        _, I = index.search(embedding, 15)
        indices = I[0]

        logger.info("🧠 Reordenando fragmentos relevantes...")
        fragmentos = reordenar_chunks(
            chunks, metadata, indices, pregunta=pregunta,
            modo="local",
            llm_model=llm if USE_GEMINI else None,
            top_k=10
        )

        logger.info("🧮 Construyendo contexto limitado por tokens...")
        max_context_tokens = N_CTX - 1024
        contexto = ""
        total_tokens = 0
        for frag in fragmentos:
            tokens = contar_tokens(frag)
            if total_tokens + tokens > max_context_tokens:
                break
            contexto += frag + "\n\n"
            total_tokens += tokens

        logger.info(f"📏 Total tokens de contexto: {total_tokens}")

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

@app.get("/")
async def root():
    return {
        "status": "online",
        "modelo": "gemini" if USE_GEMINI else "local",
        "llm": llm is not None,
        "vector_store": index is not None,
        "chunks": len(chunks),
    }

@app.get("/health")
async def health_check(api_key: str = Depends(get_api_key)):
    return {
        "status": "authenticated",
        "modelo": "gemini" if USE_GEMINI else "local", 
        "components": {
            "llm": llm is not None,
            "vector_store": index is not None,
            "embedder": embedder is not None,
            "chunks_count": len(chunks)
        }
    }