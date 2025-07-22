import faiss
import pickle
from app.config import VECTOR_STORE_DIR

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
