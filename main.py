import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 目录准备
os.makedirs("uploads", exist_ok=True)
os.makedirs("data", exist_ok=True)

# 加载模型
print("加载模型中...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("模型加载完成")

# 持久化的 ChromaDB（重启不丢数据）
client_db = chromadb.PersistentClient(path="data")
collection = client_db.get_or_create_collection("knowledge")

# AI 客户端
ai_client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 切块函数
def split_text(text, chunk_size=300):
    """按段落切，再把过长的段落按长度切"""
    chunks = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            for i in range(0, len(para), chunk_size):
                chunks.append(para[i:i+chunk_size])
    return chunks

# ========== 接口 1：上传文件 ==========
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    filepath = f"uploads/{file.filename}"
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_text(text)
    embeddings = model.encode(chunks)

    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids
    )

    return {
        "filename": file.filename,
        "chunks": len(chunks),
        "total_in_db": collection.count()
    }

# ========== 接口 2：查看状态 ==========
@app.get("/status")
def status():
    return {"total_chunks": collection.count()}

# ========== 接口 3：问答（RAG 核心） ==========
class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: AskRequest):
    # 1. 检索最相关的 3 块
    q_emb = model.encode([req.question])
    results = collection.query(
        query_embeddings=q_emb.tolist(),
        n_results=3
    )
    context = "\n\n".join(results["documents"][0])

    # 2. 拼提示词
    prompt = f"""基于以下资料回答问题，不要编造资料以外的内容。

资料：
{context}

问题：{req.question}
"""

    # 3. 调用 AI
    response = ai_client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content

    return {"answer": answer, "sources": results["documents"][0]}