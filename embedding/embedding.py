import json
import numpy as np
import os
import faiss
import torch
import gc
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# File Paths
DATA_PATH = "./data"
QA_FILE = f"{DATA_PATH}/QA_set/easy_single.json"
ALL_CHUNKS_FILE = f"{DATA_PATH}/chunked_text_all_together_cleaned.json"

# Use CUDA
device = "cpu"

# Load models on GPU
bge_model = SentenceTransformer("BAAI/bge-large-en", device=device)
mpnet_model = SentenceTransformer("all-mpnet-base-v2", device=device)

# Load all passages from the unified file
def load_all_passages():
    if not os.path.exists(ALL_CHUNKS_FILE):
        print(f"File not found: {ALL_CHUNKS_FILE}")
        return []
    with open(ALL_CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    passages = [entry["passage"] for entry in data if "passage" in entry]
    print(f"Total passages loaded: {len(passages)}")
    return passages

# Load QA questions from the QA file
def load_qa_questions():
    if not os.path.exists(QA_FILE):
        print(f"QA file not found: {QA_FILE}")
        return []
    with open(QA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Assuming each QA entry has a "question" key
    questions = [item["question"] for item in data if "question" in item]
    print(f"Total QA questions loaded: {len(questions)}")
    return questions

# Generate embeddings
def embed_texts(texts, model, batch_size=8):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        batch_embeds = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(batch_embeds)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return np.vstack(embeddings)

# Store FAISS index
def store_faiss_index(embeddings, filename):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, filename)
    print(f"FAISS index stored: {filename}")

# Unified passage processing
def process_all_hp_passages(model, model_name):
    print(f"\nProcessing unified HP embeddings for model: {model_name}")
    passages = load_all_passages()
    if not passages:
        print("No passages to embed.")
        return
    embeddings = embed_texts(passages, model)
    os.makedirs(DATA_PATH, exist_ok=True)
    index_path = f"{DATA_PATH}/hp_all_{model_name}.index"
    emb_path = f"{DATA_PATH}/hp_all_{model_name}_embeddings.npy"
    store_faiss_index(embeddings, index_path)
    np.save(emb_path, embeddings)
    print(f"Embeddings and FAISS index saved for {model_name}")

# Unified QA processing
def process_qa_files(model, model_name):
    print(f"\nProcessing QA embeddings for model: {model_name}")
    questions = load_qa_questions()
    if not questions:
        print("No QA questions found.")
        return
    embeddings = embed_texts(questions, model)
    index_path = f"{DATA_PATH}/{model_name}_qa.index"
    emb_path = f"{DATA_PATH}/qa_embeddings_{model_name}.npy"
    store_faiss_index(embeddings, index_path)
    np.save(emb_path, embeddings)
    print(f"QA embeddings and FAISS index saved for {model_name}")

# Run for both models using the updated functions
process_all_hp_passages(bge_model, "bge")
process_qa_files(bge_model, "bge")

process_all_hp_passages(mpnet_model, "mpnet")
process_qa_files(mpnet_model, "mpnet")