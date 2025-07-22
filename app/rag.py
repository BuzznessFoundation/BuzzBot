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

def reordenar_chunks(chunks: list[str], metadatos: list[dict], indices: list[int], top_k: int = 10) -> list[str]:
    seleccionados = []

    for i in indices:
        meta = metadatos[i]
        chunk = chunks[i]
        tipo = meta.get("tipo", "otro")
        prioridad = prioridad_tipo(tipo)

        seleccionados.append({
            "chunk": chunk,
            "tipo": tipo,
            "fuente": meta.get("fuente", ""),
            "prioridad": prioridad
        })

    # Ordenar por prioridad
    seleccionados.sort(key=lambda x: x["prioridad"])

    # Retornar los top_k mejores
    return [f"[{x['tipo'].upper()}] ({x['fuente']}): {x['chunk']}" for x in seleccionados[:top_k]]

