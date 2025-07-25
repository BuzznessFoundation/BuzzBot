import os
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

FRONTEND_API_KEY = os.getenv("FRONTEND_API_KEY")

if not FRONTEND_API_KEY:
    raise ValueError("FRONTEND_API_KEY no está definida en las variables de entorno")

security = HTTPBearer()

async def verify_api_key(request: Request) -> bool:
    authorization = request.headers.get("Authorization")
    
    if not authorization:
        raise HTTPException(
            status_code=401, 
            detail="Token de autorización requerido"
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, 
            detail="Formato de token inválido. Use: Bearer <token>"
        )
    
    token = authorization.replace("Bearer ", "")
    
    if token != FRONTEND_API_KEY:
        raise HTTPException(
            status_code=401, 
            detail="API key inválida"
        )
    return True

async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if credentials.credentials != FRONTEND_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="API key inválida"
        )
    return credentials.credentials