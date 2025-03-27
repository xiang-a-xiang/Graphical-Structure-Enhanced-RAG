from utils import *
from retrieval import *
from dotenv import load_dotenv
from generation.pythia_generation import generation_answer, load_model_and_tokenizer
from generation.cohere_generation import CohereGenerator
from embedding.embedding import bge_model, store_faiss_index
import json



def clean_subquestions(raw_list):
    """
    Cleans and extracts proper sub-questions from a messy list of strings.
    Only retains strings that appear to be actual questions.
    """
    cleaned = []
    for item in raw_list:
        item = item.strip()
        # Accept only strings that:
        # - start with a digit (e.g., "1. What...")
        # - or end with a question mark and are not intros
        if re.match(r"^\d+\.\s", item):
            cleaned.append(item)
    return cleaned


if __name__ == "__main__":
    hard_single_path = 'data/QA_set/hard_multi.json'

    with open(hard_single_path, 'r') as f:
        data = json.load(f)

    query1 = data[0]['question']
    print(f'Query: {query1}')

    ## QAc
    load_dotenv("key.env")
    api_key = os.environ.get('COHERE_API_KEY')
    llm = MyLLM(api_key=api_key)
    qo_a = QOAdvanced(embedding_model_name="sentence-transformers/all-mpnet-base-v1", llm=llm)
    sub_questions = clean_subquestions(qo_a.sub_questions(query1))

    print("Sub-Questions:")
    print(sub_questions)


    ## Retrieval
    # DATA_PATH = "./data"
    # QA_FILE = f"{DATA_PATH}/QA_set/easy_single.json"
    # ALL_CHUNKS_FILE = f"{DATA_PATH}/chunked_text_all_together_cleaned.json"


    # vectorizer = TfidfVectorizer(stop_words='english')
    # doc_matrix = vectorizer.fit_transform(passages)
    passages, chunk_ids = load_passages_and_chunk_ids()
    # faiss_index_stored = store_faiss_index(bge_model, filename='./faiss_index')
    # tokenized_passages = [word_tokenize(doc.lower()) for doc in passages]
    # bm25 = BM25Okapi(tokenized_passages)

    index_file = 'hp_all_bge.index'

    retrieve_results = dense_retrieval_subqueries(sub_questions, bge_model ,index_file, passages, chunk_ids, top_k=5)

    out_file = "evaluation/sub_queries_retrieval_example.json"

    # with open(out_file, "w") as out_f:
    #     json.dump(retrieve_results, out_f, indent=4)

    # for item in results:
    #     print(item, '\n\n')


    try:
        generator = CohereGenerator(api_key=api_key)
        generated_results,final_answer = generator.generation_answer(query1, sub_questions, retrieve_results)
    except Exception as e:
        print("Generation Error:", e)
        

    #Generate Answer
    # try:
    #     model, tokenizer2 = load_model_and_tokenizer()
    #     generation_answer(sub_questions, results, model, tokenizer2)
    # except Exception as e:
    #     print("Generation failed:", e)