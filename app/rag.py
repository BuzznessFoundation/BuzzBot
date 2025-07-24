import faiss
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.config import VECTOR_STORE_DIR
from app.utils import crear_respuesta
from typing import List, Literal

def cargar_vector_store():
    try:
        index = faiss.read_index(str(VECTOR_STORE_DIR / "index.faiss"))
        with open(VECTOR_STORE_DIR / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        with open(VECTOR_STORE_DIR / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return index, chunks, metadata
    except Exception as e:
        raise RuntimeError(f"Error cargando vector store: {e}")

def construir_contexto_por_tokens(chunks, metadata, indices, contar_tokens, max_tokens):
    contexto = ""
    total = 0
    for i in indices:
        fragmento = f"[Documento] ({metadata[i]}): {chunks[i]}"
        tokens = contar_tokens(fragmento)
        if total + tokens > max_tokens:
            break
        contexto += fragmento + "\n\n"
        total += tokens
    return contexto

def prioridad_tipo(tipo: str) -> int:
    orden = {
        "ley": 1,
        "reglamento": 2,
        "oficio": 3,
        "protocolo": 4,
        "guia": 5,
        "otro": 6
    }
    return orden.get(tipo.lower(), 99)

def reordenar_chunks(
    chunks: List[str],
    metadata: List[dict],
    indices: List[int],
    pregunta: str,
    modo: Literal["local", "llm"] = "local",
    top_k: int = 10,
    llm_model=None  # solo si modo='llm'
) -> List[str]:
    seleccionados = [chunks[i] for i in indices]
    seleccionados_meta = [metadata[i] for i in indices]

    if modo == "local":
        return _rerank_local(pregunta, seleccionados, top_k)
    elif modo == "llm":
        return _rerank_llm(pregunta, seleccionados, llm_model, top_k)
    else:
        raise ValueError(f"Modo de reordenamiento no soportado: {modo}")

def _rerank_local(pregunta: str, fragmentos: List[str], top_k: int) -> List[str]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")  # usa el mismo embedder del sistema
    pregunta_emb = model.encode([pregunta], normalize_embeddings=True)
    fragmentos_emb = model.encode(fragmentos, normalize_embeddings=True)

    sims = cosine_similarity(pregunta_emb, fragmentos_emb)[0]
    orden = np.argsort(sims)[::-1][:top_k]

    return [fragmentos[i] for i in orden]

def _rerank_llm(pregunta: str, fragmentos: List[str], llm_model, top_k: int) -> List[str]:
    prompt = (
        "Dada la siguiente pregunta:\n"
        f"{pregunta}\n\n"
        "Y los siguientes fragmentos:\n"
    )
    for i, frag in enumerate(fragmentos):
        prompt += f"[{i}] {frag[:500]}...\n"

    prompt += (
        "\nIndica los índices (por ejemplo: 1,3,5) de los fragmentos más relevantes "
        f"para responder esta pregunta. Máximo {top_k}."
    )

    respuesta = crear_respuesta(llm_model, prompt=prompt, system_msg="Eres un modelo experto en selección de contexto.")

    try:
        indices = [int(i.strip()) for i in respuesta.strip().split(",") if i.strip().isdigit()]
        indices = indices[:top_k]
    except:
        indices = list(range(top_k))  # fallback

    return [fragmentos[i] for i in indices if i < len(fragmentos)]
