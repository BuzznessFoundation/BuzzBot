from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from utils import cargar_modelo

app = FastAPI(title="BuzzBot API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    llm = cargar_modelo()
    logger.info("Modelo cargado correctamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}")
    llm = None  # evita crashear por OOM

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "").strip()

        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt vacío")

        if llm is None:
            return {"respuesta": f"(modo eco) {prompt}"}

        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128  # Limita longitud para bajo RAM
        )

        return {"respuesta": output["choices"][0]["message"]["content"]}

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Error interno del servidor"})

@app.get("/")
async def root():
    return {"status": "online", "message": "BuzzBot API lista para Render"}
