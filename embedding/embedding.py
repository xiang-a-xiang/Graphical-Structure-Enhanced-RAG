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
MCQ_FILE = f"{DATA_PATH}/Harry_Potter_Data.json"
HP_FILES = [f"{DATA_PATH}/hp{i}_chunked.json" for i in range(1, 8)]

# Use cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BGE on GPU
bge_model = SentenceTransformer("BAAI/bge-large-en", device=device)

# Load HP Chunks from a given file
def load_chunks(filename):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return []
    
    try:
        with open(filename, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return [chunk["passage"] for chunk in chunks if "passage" in chunk]  # Extract passages
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filename}: {e}")
        return []

# Load MCQs
def load_mcq_questions():
    if not os.path.exists(MCQ_FILE):
        print(f"MCQ file not found: {MCQ_FILE}")
        return []
    
    try:
        with open(MCQ_FILE, "r", encoding="utf-8") as f:
            questions = json.load(f)
        return [q["question"] for q in questions if "question" in q]  # Extract only questions
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {MCQ_FILE}: {e}")
        return []

# Generate embeddings
def embed_texts(texts, model, batch_size=8):
    if not texts:
        return np.array([])  # Return empty array if no texts

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing"):
        batch = texts[i:i+batch_size]
        embeddings.append(model.encode(batch, convert_to_numpy=True, normalize_embeddings=True))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return np.vstack(embeddings)

# Store FAISS Index
def store_faiss_index(embeddings, filename):
    if embeddings is None or len(embeddings) == 0:
        print(f"Skipping FAISS index for {filename} due to empty embeddings.")
        return
    
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, filename)
        print(f"FAISS index stored: {filename}")
    except Exception as e:
        print(f"Error storing FAISS index for {filename}: {e}")

# Process all HP chunked files
for hp_file in HP_FILES:
    hp_chunks = load_chunks(hp_file)
    
    if not hp_chunks:
        continue
    
    embeddings = embed_texts(hp_chunks, bge_model)

    # Create FAISS index filename and save index & embeddings
    index_filename = hp_file.replace("_chunked.json", ".index")
    store_faiss_index(embeddings, index_filename)

    embeddings_filename = hp_file.replace("_chunked.json", "_embeddings.npy")
    np.save(embeddings_filename, embeddings)
    print(f"Embeddings saved to: {embeddings_filename}")

# Process MCQs
print("Generating embeddings for MCQs...")
mcq_questions = load_mcq_questions()
if mcq_questions:
    mcq_embeddings = embed_texts(mcq_questions, bge_model)
    store_faiss_index(mcq_embeddings, f"{DATA_PATH}/bge_mcq.index")
    np.save(f"{DATA_PATH}/mcq_embeddings.npy", mcq_embeddings)
else:
    print("No MCQ questions found to process.")

# Load MPNet on GPU
mpnet_model = SentenceTransformer("all-mpnet-base-v2", device=device)

# Load HP Chunks from a given file
def load_chunks(filename):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return []
    
    try:
        with open(filename, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return [chunk["passage"] for chunk in chunks if "passage" in chunk]  # Extract passages
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filename}: {e}")
        return []

# Load MCQs
def load_mcq_questions():
    if not os.path.exists(MCQ_FILE):
        print(f"MCQ file not found: {MCQ_FILE}")
        return []
    
    try:
        with open(MCQ_FILE, "r", encoding="utf-8") as f:
            questions = json.load(f)
        return [q["question"] for q in questions if "question" in q]  # Extract questions only
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {MCQ_FILE}: {e}")
        return []

# Generate Embeddings
def embed_texts(texts, model, batch_size=8):
    if not texts:
        return np.array([])

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing"):
        batch = texts[i:i+batch_size]
        embeddings.append(model.encode(batch, convert_to_numpy=True, normalize_embeddings=True))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return np.vstack(embeddings)

# Store FAISS Index
def store_faiss_index(embeddings, filename):
    if embeddings is None or len(embeddings) == 0:
        print(f"Skipping FAISS index for {filename} due to empty embeddings.")
        return
    
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, filename)
        print(f"FAISS index stored: {filename}")
    except Exception as e:
        print(f"Error storing FAISS index for {filename}: {e}")

# Process all HP chunked files
for hp_file in HP_FILES:
    print(f"Processing file: {hp_file}")
    hp_chunks = load_chunks(hp_file)
    
    if not hp_chunks:
        continue
    
    embeddings = embed_texts(hp_chunks, mpnet_model)

    # Create FAISS index filename and save index & embeddings
    index_filename = hp_file.replace("_chunked.json", "_mpnet.index")
    store_faiss_index(embeddings, index_filename)

    embeddings_filename = hp_file.replace("_chunked.json", "_mpnet_embeddings.npy")
    np.save(embeddings_filename, embeddings)
    print(f"MPNet embeddings saved to: {embeddings_filename}")

# Process MCQs
print("Generating MPNet embeddings for MCQs...")
mcq_questions = load_mcq_questions()
if mcq_questions:
    mcq_embeddings = embed_texts(mcq_questions, mpnet_model)
    store_faiss_index(mcq_embeddings, f"{DATA_PATH}/mpnet_mcq.index")
    np.save(f"{DATA_PATH}/mcq_embeddings_mpnet.npy", mcq_embeddings)
    print("MPNet MCQ Embeddings saved successfully.")
else:
    print("No MCQ questions found to process.")