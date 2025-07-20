# 🐝 BuzzBot – Asistente IA para Escuelas Públicas en Chile

BuzzBot es un asistente inteligente diseñado para apoyar la gestión educativa en establecimientos públicos chilenos. Este sistema responde consultas sobre leyes, normativas, protocolos, oficios y buenas prácticas mediante inteligencia artificial contextual, utilizando una arquitectura RAG (Retrieval-Augmented Generation).
Desarrollado como un MVP robusto y funcional, BuzzBot busca entregar **información precisa, trazable y jerarquizada** para equipos directivos, UTP, docentes, inspectores y asistentes de la educación, democratizando el acceso al conocimiento normativo y administrativo.

---

## 🧠 Propósito

En muchos establecimientos públicos, acceder rápidamente a normativas o protocolos actualizados implica revisar decenas de PDFs o buscar entre documentos mal organizados. BuzzBot nace como una solución concreta a ese dolor: un asistente que *entiende la estructura legal y administrativa de la educación chilena*, y entrega respuestas claras, justificadas y citadas.

> “La información ya está ahí, solo que no es accesible para quienes más la necesitan.”

---

## ⚙️ ¿Cómo funciona?

BuzzBot combina:

- 🔍 **Recuperación semántica**: identifica los documentos más relevantes usando embeddings (`sentence-transformers`) y FAISS.
- 🧠 **Modelo LLM**: genera respuestas naturales y fundadas, usando:
  - `phi-3-mini` localmente (offline)
  - `Gemini 1.5 Pro` mediante API (plan de pago)
- 📄 **Jerarquización documental**: prioriza leyes estructurales sobre protocolos o buenas prácticas al construir el contexto.

Todo esto encapsulado en una API REST minimalista y lista para desplegarse vía Docker.

---

## 🛠️ Tecnologías utilizadas

- `FastAPI` para la API REST
- `SentenceTransformers` para embeddings semánticos
- `FAISS` para recuperación vectorial
- `Phi-3 Mini` en modo offline (`llama-cpp`)
- `Google Gemini` para integración en nube (opcional)
- `Docker` para empaquetado y despliegue

---

## 🧰 Estructura del proyecto

BuzzBot/
├── app.py # API principal con lógica RAG
├── utils.py # Abstracción de modelo (local o Gemini)
├── models/ # Contiene phi-3-mini (.gguf) si se usa offline
├── vector_store/ # Índice FAISS, chunks de texto, metadata
├── Dockerfile # Para contenedores reproducibles
├── requirements.txt
└── README.md
