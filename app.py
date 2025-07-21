from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import faiss
import pickle
from sentence_transformers import SentenceTransformer

from utils import (
    cargar_modelo_local,
    cargar_modelo_gemini,
    crear_respuesta,
    get_system_prompt,
    es_pregunta_trivial,
    USE_GEMINI,
    VECTOR_STORE_DIR,
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

# Carga modelo (local o Gemini)
try:
    if USE_GEMINI:
        llm = cargar_modelo_gemini()
        N_CTX = 8192  # Ajustar según specs reales Gemini Flash
    else:
        llm = cargar_modelo_local()
        N_CTX = 4096
    logger.info(f"Modelo cargado correctamente. USE_GEMINI={USE_GEMINI}")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}")
    llm = None
    N_CTX = 0

# Carga vector_store
try:
    index = faiss.read_index(str(VECTOR_STORE_DIR / "index.faiss"))
    with open(VECTOR_STORE_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open(VECTOR_STORE_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    logger.info("Vector store cargado correctamente")
except Exception as e:
    logger.error(f"Error al cargar vector store: {e}")
    index = None
    chunks = []
    metadata = []

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def contar_tokens(texto: str) -> int:
    if USE_GEMINI:
        # Aproximación tokens Gemini: 1 token ≈ 4 caracteres
        return len(texto) // 4
    else:
        try:
            return len(llm.tokenize(texto.encode("utf-8")))
        except Exception as e:
            logger.error(f"Error tokenizando texto: {e}")
            return 0

def construir_contexto(fragmentos: list[str], max_tokens: int) -> str:
    contexto = ""
    total_tokens = 0
    for frag in fragmentos:
        tks = contar_tokens(frag)
        if total_tokens + tks > max_tokens:
            break
        contexto += frag + "\n\n"
        total_tokens += tks
    return contexto

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        pregunta = data.get("prompt", "").strip()
        if not pregunta:
            raise HTTPException(status_code=400, detail="Prompt vacío")

        if llm is None or index is None:
            return {"respuesta": f"(modo eco) {pregunta}"}

        # Preguntas triviales: responder rápido sin buscar en vector_store
        if es_pregunta_trivial(pregunta):
            return {"respuesta": "Hola! ¿En qué puedo ayudarte con la administración escolar?"}

        embedding = embedder.encode([pregunta], normalize_embeddings=True)
        k = 5
        _, I = index.search(embedding, k)
        indices = I[0]

        fragmentos = []
        for i in indices:
            tipo = "Documento"
            fuente = metadata[i]
            texto = chunks[i]
            fragmentos.append(f"[{tipo}] ({fuente}): {texto}")

        reserved_tokens = 512
        max_context_tokens = N_CTX - reserved_tokens
        contexto = construir_contexto(fragmentos, max_context_tokens)

        system_prompt = get_system_prompt()

        # Crear prompt separado para Gemini y local
        prompt = (
            f"Contextos disponibles:\n{contexto}\n\n"
            f"Pregunta:\n{pregunta}\n\n"
            f"Respuesta clara y precisa:"
        )

        respuesta = crear_respuesta(
            llm,
            prompt=prompt,
            system_msg=system_prompt,
            max_tokens=512
        )

        return {"respuesta": respuesta}

    except Exception as e:
        logger.error(f"Error en /chat: {e}")
        return JSONResponse(status_code=500, content={"error": "Error interno del servidor"})

@app.get("/")
async def root():
    return {"status": "online", "message": "BuzzBot API lista para responder"}
