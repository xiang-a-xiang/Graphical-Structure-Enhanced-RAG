from utils import TextChunker, make_chunked_json_file
from tqdm import tqdm
import multiprocessing as mp
from types import SimpleNamespace
from eval_utils import *

args = {
    'encoder_model_name' : 'sentence-transformers/all-mpnet-base-v1',
    'context_window' : 5, 
    'similarity_percentile_threshold' : 30, 
    'min_chunk_size' : 5, 
    'keep_natural_order' : True
}

args = SimpleNamespace(**args)

chunker = TextChunker(args.encoder_model_name)

# Create chunked json files for each Harry Potter book
def create_json_chunked_files():
    print("Creating chunked json files for each Harry Potter book...")
    for i in tqdm(range(1, 8)):
        make_chunked_json_file(f"data/hp{i}.txt", f"data/hp{i}_chunked.json", chunker, args.context_window, args.similarity_percentile_threshold, args.min_chunk_size, args.keep_natural_order)
    print("Done!")


eval_collection = MetricCollection({
    "recall@5": RecallAtK(k=5),
    "recall@10": RecallAtK(k=10),
    "precision@5": PrecisionAtK(k=5),
    "precision@10": PrecisionAtK(k=10),
    "mrr": MRR(),
    "ndcg": nDCG(),
    "bleu": BLEU(),
    "rouge": ROUGE(),
    "bertscore": BERTScore()
})

if __name__ == "__main__":
    create_json_chunked_files()