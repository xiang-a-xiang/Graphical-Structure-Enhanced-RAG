import json
import os
import re
import faiss
import numpy as np
from dotenv import load_dotenv

from utils import MyLLM, QOAdvanced
from retrieval import load_passages_and_chunk_ids, dense_retrieval_subqueries
from dotenv import load_dotenv
from generation.pythia_generation import generation_answer, load_model_and_tokenizer
from generation.cohere_generation import CohereGenerator



def clean_subquestions(raw_list):
    """
    Cleans and extracts proper sub-questions from a messy list of strings.
    Only retains strings that appear to be actual questions.
    """
    cleaned = []
    for item in raw_list:
        item = item.strip()
        # Accept only strings that start with a digit followed by a period and a space.
        if re.match(r"^\d+\.\s", item):
            cleaned.append(item)
    return cleaned


def load_query(question_idx, qa_path):
    """Load a query from the QA JSON file."""
    with open(qa_path, 'r') as f:
        data = json.load(f)
    query = data[question_idx]['question']
    print(f"Query: {query}")
    return query, data


def get_sub_questions(query, qa_qo_path, question_idx, api_key):
    """
    Get sub-questions using QOAdvanced. If no sub-questions are generated,
    fall back to those stored in the QA QO file.
    """
    llm = MyLLM(api_key=api_key)
    qo_a = QOAdvanced(embedding_model_name="sentence-transformers/all-mpnet-base-v1", llm=llm)
    sub_questions = clean_subquestions(qo_a.sub_questions(query))
    
    # Fallback: if no sub-questions generated, load from qa_qo file.
    if not sub_questions:
        with open(qa_qo_path, 'r') as f:
            qa_qo_data = json.load(f)
        sub_questions = clean_subquestions(qa_qo_data[question_idx]['sub_questions'])
    
    print("Sub-Questions:")
    print(sub_questions)
    return sub_questions


def load_subquestion_list(qa_qo_path):
    """Flatten all sub-questions from the QA QO JSON file into a single list."""
    with open(qa_qo_path, 'r') as f:
        qa_qo_data = json.load(f)
    subquestion_list = []
    for item in qa_qo_data:
        for sub in item['sub_questions']:
            subquestion_list.append(sub)
    return subquestion_list


def perform_retrieval(sub_questions, subquestion_list, query_index_path, chunk_index_path, top_k=5):
    """Load FAISS indices, load passages, and run dense retrieval."""
    # Load passages and chunk IDs.
    passages, chunk_ids = load_passages_and_chunk_ids()
    
    # Load FAISS indices.
    query_index = faiss.read_index(query_index_path)
    chunk_index = faiss.read_index(chunk_index_path)
    
    # Perform retrieval.
    results = dense_retrieval_subqueries(
        sub_questions,
        subquestion_list,
        query_index,
        chunk_index,
        passages,
        chunk_ids,
        top_k=top_k
    )
    return results


def save_results(results, out_file):
    """Save retrieval results to a JSON file."""
    with open(out_file, "w") as out_f:
        json.dump(results, out_f, indent=4)
    print(f"Results saved to {out_file}")


def generate_final_answer(query, sub_questions, retrieve_results, api_key):
    """Generate final answer using CohereGenerator."""
    try:
        generator = CohereGenerator(api_key=api_key)
        generated_results, final_answer = generator.generation_answer(query, sub_questions, retrieve_results)
        return generated_results, final_answer
    except Exception as e:
        print("Generation Error:", e)
        return None, None


def main():
    # Load environment variables.
    load_dotenv("key.env")
    api_key = os.environ.get('COHERE_API_KEY')
    
    # File paths and parameters.
    question_idx = 3
    qa_path = 'data/QA_set/medium_single.json'
    qa_qo_path = 'data/QO_set/medium_single_QO.json'
    query_index_path = 'bge_qa_subqueries.index'
    chunk_index_path = 'hp_all_bge.index'
    out_file = "evaluation/sub_queries_retrieval_example2.json"
    
    # Load the main query.
    query, _ = load_query(question_idx, qa_path)
    
    # Obtain sub-questions.
    sub_questions = get_sub_questions(query, qa_qo_path, question_idx, api_key)
    
    # Load complete subquestion list.
    subquestion_list = load_subquestion_list(qa_qo_path)
    
    # Perform dense retrieval.
    retrieve_results = perform_retrieval(sub_questions, subquestion_list, query_index_path, chunk_index_path, top_k=5)
    
    # Save retrieval results.
    save_results(retrieve_results, out_file)
    
    # Generate final answer.
    generated_results, final_answer = generate_final_answer(query, sub_questions, retrieve_results, api_key)
    if final_answer:
        print("Final Answer:")
        print(final_answer)
    else:
        print("No final answer generated.")


if __name__ == "__main__":
    main()