import os
import json
import uuid
import pypdf
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="PageIndex AI Simulator Backend")

# DeepSeek Client setup
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_key_here":
    print("WARNING: DEEPSEEK_API_KEY not found or invalid in .env")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# ChromaDB Setup
db_path = os.path.join(os.getcwd(), "chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)
# Using local embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Memory store for the Tree Structure (since it's hierarchical, JSON is better than pure Vector for this)
TREE_STORE_PATH = "document_tree.json"

def save_tree(tree):
    with open(TREE_STORE_PATH, "w") as f:
        json.dump(tree, f)

def load_tree():
    if os.path.exists(TREE_STORE_PATH):
        with open(TREE_STORE_PATH, "r") as f:
            return json.load(f)
    return None

class QueryRequest(BaseModel):
    query: str

# Helper to extract PDF text
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

@app.get("/")
async def read_index():
    return FileResponse("index.html")

@app.get("/tutorial")
async def read_tutorial():
    return FileResponse("tutorial/index.html")

@app.post("/api/index")
async def index_document(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        content = ""
        filename = "Manual Input"
        
        if file:
            filename = file.filename
            temp_path = f"temp_{uuid.uuid4()}_{file.filename}"
            with open(temp_path, "wb") as f:
                f.write(await file.read())
            
            if file.filename.endswith(".pdf"):
                content = extract_text_from_pdf(temp_path)
            else:
                with open(temp_path, "r", encoding="utf-8") as f:
                    content = f.read()
            
            os.remove(temp_path)
        elif text:
            content = text
        else:
            raise HTTPException(status_code=400, detail="No content provided")

        # 1. Generate Hierarchical Tree using DeepSeek (Agentic RAG)
        prompt_indexing = f"""Tugas Anda adalah membaca teks di bawah ini lalu merangkum dan memecahnya menjadi struktur pohon hierarki (Bab, Sub-bab).
        
        DATA: {content[:10000]} # Limit to first 10k chars for stability in indexing
        
        ATURAN JSON:
        1. Output murni JSON.
        2. Struktur: {{ "id": "root", "title": "...", "summary": "...", "content": "...", "children": [...] }}
        """

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert document indexer. Output ONLY valid JSON."},
                {"role": "user", "content": prompt_indexing}
            ],
            response_format={"type": "json_object"}
        )
        
        tree_data = json.loads(response.choices[0].message.content)
        tree_data["title"] = filename if "title" not in tree_data else tree_data["title"]
        save_tree(tree_data)

        # 2. Store in ChromaDB (Vector RAG)
        # We split the content into chunks for vector RAG comparison
        collection = chroma_client.get_or_create_collection(
            name="document_chunks",
            embedding_function=sentence_transformer_ef
        )
        # Clear previous docs
        try:
            chroma_client.delete_collection("document_chunks")
            collection = chroma_client.create_collection(
                name="document_chunks",
                embedding_function=sentence_transformer_ef
            )
        except:
            pass

        # Simple overlap chunking
        chunk_size = 1000
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size - 100)]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)

        return {"status": "success", "tree": tree_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_document(req: QueryRequest):
    try:
        tree = load_tree()
        if not tree:
            raise HTTPException(status_code=404, detail="No document indexed")

        # --- PHASE 1: Agentic RAG (Mapping via Tree) ---
        # Generate a small map for the LLM to choose from
        def get_map(node):
            return {
                "id": node["id"], "title": node["title"], "summary": node["summary"],
                "children": [get_map(c) for c in node.get("children", [])]
            }
        
        tree_map = get_map(tree)
        
        map_prompt = f"""PETA DOKUMEN: {json.dumps(tree_map)}
        PERTANYAAN: "{req.query}"
        Pilih 1 node ID yang paling relevan untuk menjawab. Output JSON: {{"selected_node_id": "..."}}"""

        map_res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a document navigator. Output JSON only."},
                {"role": "user", "content": map_prompt}
            ],
            response_format={"type": "json_object"}
        )
        selected_id = json.loads(map_res.choices[0].message.content).get("selected_node_id", "root")

        # Find content of selected node
        def find_node(node, target_id):
            if node["id"] == target_id: return node
            for c in node.get("children", []):
                found = find_node(c, target_id)
                if found: return found
            return None
        
        selected_node = find_node(tree, selected_id)
        context_agentic = selected_node["content"] if selected_node else tree["content"]

        # --- PHASE 2: Vector RAG (Semantic Search via ChromaDB) ---
        collection = chroma_client.get_collection(name="document_chunks", embedding_function=sentence_transformer_ef)
        vector_res = collection.query(query_texts=[req.query], n_results=3)
        context_vector = "\n---\n".join(vector_res["documents"][0])

        # --- PHASE 3: Synthesis (Final Answer) ---
        # We'll use the agentic context but mention vector results if different
        synthesis_prompt = f"""KONTEKS (Hierarchical/Agentic): {context_agentic}
        
        KONTEKS (Vector Recovery): {context_vector}
        
        PERTANYAAN: {req.query}
        
        Jawab pertanyaan dengan akurat berdasarkan konteks di atas. Gunakan Bahasa Indonesia."""

        final_res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Anda adalah asisten AI PageIndex."},
                {"role": "user", "content": synthesis_prompt}
            ]
        )

        return {
            "answer": final_res.choices[0].message.content,
            "selected_node_id": selected_id,
            "vector_context_used": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files (if any extra assets exist) - not strictly needed if we just serve index.html via root
# But good for local images if any.
# app.mount("/static", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
