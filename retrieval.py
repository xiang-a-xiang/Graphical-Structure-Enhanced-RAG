from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

# Dense retrieval
def dense_retrieval(query, model, faiss_index, passages, top_k=5):
    # Encode query using the same model and normalization settings
    query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    # Search FAISS index; note that FAISS works with L2 distance, so using normalized embeddings makes this similar to cosine similarity
    D, I = faiss_index.search(np.expand_dims(query_embedding, axis=0), top_k)
    results = [passages[i] for i in I[0]]
    return results, D[0], I[0]
# Load the FAISS index
index_path = f"{DATA_PATH}/hp_all_bge.index"
faiss_index = faiss.read_index(index_path)

#Sparse retrieval: TFIDF & BM25
# Build the TF-IDF matrix on your passages
passages = load_all_passages()
vectorizer = TfidfVectorizer(stop_words='english')
doc_matrix = vectorizer.fit_transform(passages)

def tfidf_retrieval(query, vectorizer, doc_matrix, passages, top_k=5):
    query_vec = vectorizer.transform([query])
    # Compute cosine similarity between query and all documents
    cosine_similarities = (doc_matrix * query_vec.T).toarray().flatten()
    indices = np.argsort(cosine_similarities)[::-1][:top_k]
    results = [passages[i] for i in indices]
    return results, cosine_similarities[indices], indices

# BM25
tokenized_passages = [word_tokenize(doc.lower()) for doc in passages]
bm25 = BM25Okapi(tokenized_passages)
def bm25_retrieval(query, bm25, passages, top_k=5):
    # Tokenize the query in the same way as the passages
    tokenized_query = word_tokenize(query.lower())
    # Get BM25 scores for the query against all passages
    scores = bm25.get_scores(tokenized_query)
    # Get the indices of the top scoring passages
    top_indices = scores.argsort()[::-1][:top_k]
    results = [passages[i] for i in top_indices]
    return results, scores[top_indices], top_indices