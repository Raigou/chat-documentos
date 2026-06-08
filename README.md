# Chat con Documentos — RAG API

Sistema de preguntas y respuestas sobre documentos PDF usando RAG (Retrieval-Augmented Generation), construido con FastAPI, LangChain, ChromaDB y Ollama.

## Tecnologías
- Python 3.11
- FastAPI
- LangChain
- ChromaDB
- Ollama (Llama3 + nomic-embed-text)
- Docker

## Arquitectura

PDFs → chunks → embeddings (nomic-embed-text) → ChromaDB
                                                      ↓
usuario pregunta → busca chunks relevantes → Llama3 responde

## Requisitos previos
- Ollama instalado y corriendo
- Modelos descargados:

ollama pull llama3
ollama pull nomic-embed-text

## Cómo correr el proyecto

### 1. Clonar el repositorio

git clone https://github.com/Raigou/chat-documentos.git
cd chat-documentos

### 2. Agregar PDFs
Coloca tus archivos PDF en la carpeta data/

### 3. Instalar dependencias

pip install -r requirements.txt

### 4. Correr la API

uvicorn app.main:app --host 127.0.0.1 --port 8000

### 5. Probar
Abre http://127.0.0.1:8000/docs

## Ejemplo

Request:
{
  "pregunta": "Como funciona el encoder y decoder en transformers?"
}

Response:
{
  "pregunta": "Como funciona el encoder y decoder en transformers?",
  "respuesta": "El encoder está compuesto por una pila de N=6 capas idénticas..."
}

## Características
- Responde en el mismo idioma de la pregunta (español e inglés)
- System prompt configurable para controlar el comportamiento del LLM
- No alucina, responde solo con información de los documentos
