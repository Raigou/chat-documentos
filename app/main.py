import os
from fastapi import FastAPI
from pydantic import BaseModel
from model.rag import cargar_documentos, crear_vectorstore, crear_cadena_rag, buscar_chunks

APP_ENV = os.getenv("APP_ENV", "development")

app = FastAPI(title="Chat con Documentos API")

documentos = cargar_documentos()
vectorstore_global = crear_vectorstore(documentos)
cadena_rag = crear_cadena_rag(vectorstore_global)

class Pregunta(BaseModel):
    pregunta: str

@app.get("/")
def root():
    return {
        "status": "ok",
        "descripcion": "Chat con documentos usando RAG",
        "environment": APP_ENV
    }

@app.post("/chat")
def chat(body: Pregunta):
    buscar_chunks(body.pregunta, vectorstore_global)
    respuesta = cadena_rag.invoke(body.pregunta)
    return {
        "pregunta": body.pregunta,
        "respuesta": respuesta
    }