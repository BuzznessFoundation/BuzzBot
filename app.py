from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from utils import cargar_modelo, crear_respuesta

app = FastAPI(title="BuzzBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Carga modelo LLM ---
try:
    llm = cargar_modelo()
    # Obtén la ventana de contexto de tu modelo
    N_CTX = llm.n_ctx()  # por ejemplo 4096
    logger.info(f"Modelo cargado. n_ctx={N_CTX}")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}")
    llm = None
    N_CTX = 0

# --- Carga vector_store ---
VECTOR_DIR = "vector_store"
try:
    index = faiss.read_index(f"{VECTOR_DIR}/index.faiss")
    with open(f"{VECTOR_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open(f"{VECTOR_DIR}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    logger.info("Vector store cargado correctamente")
except Exception as e:
    logger.error(f"Error al cargar vector store: {e}")
    index = None
    chunks = []
    metadata = []

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def contar_tokens(texto) -> int:
    if not isinstance(texto, str):
        texto = str(texto)
    try:
        return len(llm.tokenize(texto.encode("utf-8")))  # ⚠️ .encode() a bytes
    except Exception as e:
        logger.error(f"Error al tokenizar: {e} | tipo: {type(texto)} | contenido: {texto[:100]}")
        return 0

def construir_contexto(fragmentos: list[str], max_ctx_tokens: int) -> str:
    contexto = ""
    total = 0
    for frag in fragmentos:
        tokens = contar_tokens(frag)
        if total + tokens > max_ctx_tokens:
            break
        contexto += frag + "\n\n"
        total += tokens
    return contexto


@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        pregunta = data.get("prompt", "").strip()
        if not pregunta:
            raise HTTPException(status_code=400, detail="Prompt vacío")

        if llm is None or index is None:
            # fallback modo eco
            return {"respuesta": f"(modo eco) {pregunta}"}

        # 1. Embed pregunta y recuperar top k documentos
        embedding = embedder.encode([pregunta], normalize_embeddings=True)
        k = 5  # puedes ajustar
        _, I = index.search(embedding, k)
        indices = I[0]

        # 2. Construir lista de fragmentos con metadatos
        fragmentos = []
        for i in indices:
            tipo = "Documento"
            fuente = metadata[i]
            texto = chunks[i]
            fragmentos.append(f"[{tipo}] ({fuente}): {texto}")

        # 3. Calcular presupuesto de tokens para contexto
        reserved = 512  # tokens que dejamos para la respuesta
        max_ctx = N_CTX - reserved
        contexto = construir_contexto(fragmentos, max_ctx)

        # 4. Prompt base más desarrollado
        system_msg = (
            "Eres un asistente especializado en la administración de escuelas públicas en Chile. "
            "Tu tarea es responder con precisión y citar las fuentes (nombres de archivos) cuando corresponda. "
            "Las preguntas pueden versar sobre leyes, reglamentos, oficios, protocolos y buenas prácticas.\n"
            "Si no sabes la respuesta, admite desconocimiento."
        )

        user_msg = (
            f"CONTEXTOS DISPONIBLES:\n{contexto}\n\n"
            f"PREGUNTA:\n{pregunta}\n\n"
            "RESPUESTA (clara y precisa):"
        )

        # 5. Llamada al modelo con formato chatml
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        logger.info(f"Prompt final tokens: {contar_tokens(user_msg) + contar_tokens(system_msg)}")
        logger.info(f"System msg: {system_msg[:200]}")
        logger.info(f"User msg: {user_msg[:200]}")

        respuesta = crear_respuesta(llm, prompt)
        return {"respuesta": respuesta}


        resp = output["choices"][0]["message"]["content"].strip()
        return {"respuesta": resp}

    except Exception as e:
        logger.error(f"Error en /chat: {e}")
        return JSONResponse(status_code=500, content={"error": "Error interno del servidor"})

@app.get("/")
async def root():
    return {"status": "online", "message": "BuzzBot API lista para responder"}
