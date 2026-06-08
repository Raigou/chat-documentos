import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DATA_PATH = "data/"
CHROMA_PATH = "chroma_db/"

def cargar_documentos():
    documentos = []
    for archivo in os.listdir(DATA_PATH):
        if archivo.endswith(".pdf"):
            loader = PyPDFLoader(DATA_PATH + archivo)
            documentos.extend(loader.load())
    print(f"{len(documentos)} páginas cargadas.")
    return documentos

def crear_vectorstore(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documentos)
    print(f"{len(chunks)} chunks creados.")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    print("Vectorstore creado.")
    return vectorstore

def crear_cadena_rag(vectorstore):
    llm = OllamaLLM(model="llama3")

    system_prompt = """Eres un asistente experto multilingüe.
Responde SIEMPRE en el mismo idioma en que está escrita la pregunta.
Responde SOLO con la información de los documentos provistos.
Si no encuentras la respuesta, di exactamente:
'No tengo esa información en los documentos.' (en el idioma de la pregunta)
No inventes información.

Contexto: {context}

Pregunta: {question}
Respuesta:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=system_prompt
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    cadena = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return cadena

def inicializar_rag():
    documentos = cargar_documentos()
    vectorstore = crear_vectorstore(documentos)
    cadena = crear_cadena_rag(vectorstore)
    return cadena

def buscar_chunks(query: str, vectorstore):
    docs = vectorstore.similarity_search(query, k=5)
    for i, doc in enumerate(docs):
        print(f"\n--- Chunk {i+1} ---")
        print(doc.page_content)