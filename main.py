import os
import shutil
from datetime import datetime, timedelta
import traceback
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from jose import jwt, JWTError
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load ENV
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRY_MINUTES = int(os.getenv("JWT_EXPIRY_MINUTES", 60))

if not JWT_SECRET:
    raise ValueError("JWT_SECRET missing!")

app = FastAPI(title="Supabase + Groq RAG API with JWT Auth")

# =====================================================================
# SIMPLE JWT AUTH (NO OAUTH2, NO CLIENT ID, NO CLIENT SECRET)
# =====================================================================

class LoginRequest(BaseModel):
    username: str
    password: str

jwt_bearer = HTTPBearer()  # Accepts Bearer <token>


def create_jwt_token(data: dict):
    expiry = datetime.utcnow() + timedelta(minutes=JWT_EXPIRY_MINUTES)
    data.update({"exp": expiry})
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(jwt_bearer)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# Dummy User
FAKE_USER = {
    "username": "admin",
    "password": "admin123"
}

# =====================================================================
# LOGIN ENDPOINT — SIMPLE USERNAME + PASSWORD = TOKEN
# =====================================================================

@app.post("/login")
async def login(request: LoginRequest):
    if request.username != FAKE_USER["username"] or request.password != FAKE_USER["password"]:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    token = create_jwt_token({"sub": request.username})
    return {"access_token": token, "token_type": "bearer"}


# =====================================================================
# SUPABASE + EMBEDDING SETUP
# =====================================================================

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents",
)

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)


# =====================================================================
# REQUEST MODELS
# =====================================================================

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


# =====================================================================
# UPLOAD ENDPOINT (PROTECTED WITH JWT)
# =====================================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    _jwt=Depends(verify_jwt_token)
):
    temp_file_path = f"temp_{file.filename}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile")
        chunks = splitter.split_documents(docs)

        SupabaseVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents"
        )

        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "chunks_created": len(chunks)
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# =====================================================================
# QUERY ENDPOINT (PROTECTED WITH JWT)
# =====================================================================

@app.post("/query", response_model=QueryResponse)
async def query_db(
    request: QueryRequest,
    _jwt=Depends(verify_jwt_token)
):
    try:
        docs = vector_store.similarity_search(request.query, k=request.k)

        if not docs:
            return {"answer": "No relevant info found.", "sources": []}

        context = "\n\n".join([d.page_content for d in docs])
        sources = [d.metadata.get("source", "Unknown") for d in docs]

        template = """
        Answer the question strictly using the context below:

        {context}

        Question: {question}

        Answer:
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": request.query})

        return {"answer": response, "sources": list(set(sources))}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))
