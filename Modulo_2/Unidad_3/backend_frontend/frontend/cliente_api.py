from __future__ import annotations
import httpx  #Cliente HTTP 

URL_BASE = "http://127.0.0.1:8000"

TIMEOUT = 5.0

def pedir_salud() -> dict:
    
    try: 
        r = httpx.get(f"{URL_BASE}/salud", timeout=TIMEOUT)
        r.raise_for_status()
        return {"estado": "exito", "datos": r.json()}
    
    except httpx.ConnectError:
        return {
            "estado": "error de conexión",
            "mensaje": "El backend no está.",
        }
    
    except httpx.HTTPError as error:
        return {"estado": "otro_error", "mensaje": str(error)}


def pedir_prediccion(cuerpo: dict, motor: str = "sklearn") -> dict:
    
    ruta = "/predecir/sklearn" if motor == "sklearn" else "/predecir/red"
    
    try:
        r = httpx.post(f"{URL_BASE}{ruta}", json=cuerpo, timeout=TIMEOUT)
    
    except httpx.ConnectError:
        return {
            "estado": "error de conexión",
            "mensaje": "El backend no está."
        }
        
    except httpx.TimeoutException:
        
        return {"estado": "otro_error", "mensaje": "La API tardó más de 5 seg"}
    
    if r.status_code == 200:
        #EXITO: el backend ya corrió predict y nos mandó el numero
        return {"estado": "exito", "datos": r.json()}
    
    if r.status_code == 422:
        return {"estado": "error de validación", "mensaje": r.json()}
    
    if r.status_code == 404:
        return {"estado": "error_no_encontrado", "mensaje": r.json().get("detail", r.text)}
    return {"estado": "error_otro", "mensaje": f"HTTP {r.status_code}: {r.text}"}


def pedir_por_cliente(cliente_id: int, motor: str = "sklearn") -> dict:
    
    try:
        r = httpx.get(
            f"{URL_BASE}/predecir/cliente/{cliente_id}",
            params={"motor": "sklearn" if motor == "sklearn" else "red"},
            timeout=TIMEOUT,
        )
        
    except httpx.ConnectError:
        return {
            "estado": "erro de conexión",
            "mensaje": "No hay API en el puerto 8000",
        }
        
    if r.status_code == 200:
    #EXITO: el backend ya corrió predict y nos mandó el numero
        return {"estado": "exito", "datos": r.json()}
    
    if r.status_code == 404:
        return {"estado": "error_no_encontrado", "mensaje": r.json().get("detail", r.text)}
    return {"estado": "error_otro", "mensaje": f"HTTP {r.status_code}: {r.text}"}