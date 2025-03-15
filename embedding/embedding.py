import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load embedding models
bge_model = SentenceTransformer("BAAI/bge-large-en", device="cpu")  # Force CPU for debugging
mpnet_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

# Reduce sequence length to lower memory usage
bge_model.max_seq_length = 512  
mpnet_model.max_seq_length = 512

# Load only HP1
DATA_PATH = "./data"
hp1_file = f"{DATA_PATH}/hp1_chunked.json"

# Load chunks
def load_hp1_chunks():
    try:
        with open(hp1_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"✅ Successfully loaded {len(chunks)} chunks from HP1.")  # Debugging print
        return [chunk["passage"] for chunk in chunks]  
    except FileNotFoundError:
        print(f"❌ Error: File '{hp1_file}' not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format in '{hp1_file}': {e}")
        return []

hp1_chunks = load_hp1_chunks()
if len(hp1_chunks) == 0:
    print("❌ No text chunks loaded. Exiting.")
    exit()

# Generate embeddings with additional debugging
def embed_texts(texts, model, model_name):
    try:
        print(f"🔹 Generating embeddings for {len(texts)} texts using {model_name}...")
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        print(f"✅ Embeddings generated successfully! Shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f"❌ Error generating embeddings with {model_name}: {e}")
        return None

# Run embedding generation
bge_hp1_embeddings = embed_texts(hp1_chunks, bge_model, "BGE-Large-EN")
mpnet_hp1_embeddings = embed_texts(hp1_chunks, mpnet_model, "MPNet")

# Store FAISS index if embeddings exist
def store_faiss_index(embeddings, filename):
    if embeddings is None:
        print(f"❌ Skipping FAISS index creation for {filename}")
        return
    try:
        print(f"🔹 Storing FAISS index for {filename}...")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, filename)
        print(f"✅ FAISS index stored: {filename}")
    except Exception as e:
        print(f"❌ Error storing FAISS index for {filename}: {e}")

store_faiss_index(bge_hp1_embeddings, "bge_hp1.index")
store_faiss_index(mpnet_hp1_embeddings, "mpnet_hp1.index")