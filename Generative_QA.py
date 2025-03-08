from utils import TextChunker, make_chunked_json_file_wrapper
from tqdm import tqdm
import multiprocessing as mp
from types import SimpleNamespace

args = {
    'encoder_model_name' : 'sentence-transformers/all-mpnet-base-v1',
    'context_window' : 5, 
    'similarity_percentile_threshold' : 30, 
    'min_chunk_size' : 5, 
    'keep_natural_order' : True
}

args = SimpleNamespace(**args)

# Create chunked json files for each Harry Potter book
def create_json_chunked_files():
    print("Creating chunked json files for each Harry Potter book...")
    tasks = [
            ("data/hp1.txt", "data/hp1_chunked.json", args.encoder_model_name, args.context_window, args.similarity_percentile_threshold, args.min_chunk_size, args.keep_natural_order),
            ("data/hp2.txt", "data/hp2_chunked.json", args.encoder_model_name, args.context_window, args.similarity_percentile_threshold, args.min_chunk_size, args.keep_natural_order),
            ("data/hp3.txt", "data/hp3_chunked.json", args.encoder_model_name, args.context_window, args.similarity_percentile_threshold, args.min_chunk_size, args.keep_natural_order),
            ("data/hp4.txt", "data/hp4_chunked.json", args.encoder_model_name, args.context_window, args.similarity_percentile_threshold, args.min_chunk_size, args.keep_natural_order),
            ("data/hp5.txt", "data/hp5_chunked.json", args.encoder_model_name, args.context_window, args.similarity_percentile_threshold, args.min_chunk_size, args.keep_natural_order),
            ("data/hp6.txt", "data/hp6_chunked.json", args.encoder_model_name, args.context_window, args.similarity_percentile_threshold, args.min_chunk_size, args.keep_natural_order),
            ("data/hp7.txt", "data/hp7_chunked.json", args.encoder_model_name, args.context_window, args.similarity_percentile_threshold, args.min_chunk_size, args.keep_natural_order)
        ]
    ctx = mp.get_context('spawn')
    with ctx.Pool(7) as p:
        results = list(tqdm(p.imap_unordered(make_chunked_json_file_wrapper, tasks), total=7))
    print("Done!")

if __name__ == "__main__":
    create_json_chunked_files()