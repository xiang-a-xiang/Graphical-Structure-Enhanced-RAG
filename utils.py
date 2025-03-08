# Description: This file contains utility functions that are used in the main script.

from transformers import AutoTokenizer, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import re
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

class NLPBase:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.trainer = None

    def preprocess_function(self, examples):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class NER(NLPBase):
    def __init__(self, model_name, label_list):
        super().__init__(model_name)
        self.label_list = label_list
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.id2label = {i: label for i, label in enumerate(label_list)}
        
        

    def preprocess_function(self, examples):
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    # print(label)
                    # print(label[word_idx])
                    label_ids.append(label[word_idx])
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def train(self, dataset, training_args, model):
        tokenized_dataset = dataset.map(self.preprocess_function, batched=True)
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        model = model.from_pretrained(
            self.model_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,

        )
        self.trainer.train()
    def eval(self):
        """
        Evaluate the model on the validation set.
        """
        if self.trainer is None:
            raise ValueError("Model has not been trained yet.")
        
        results = self.trainer.evaluate()
        print(f"Evaluation results: {results}")
        return results



class QA(NLPBase):
    def __init__(self, model_name):
        super().__init__(model_name)

    """
    Assume the dataset has the following structure:
    {
    "context": "Elizabeth Bennett met Mr. Darcy at a ball in the town of Meryton...",
    "question": "Where did Elizabeth Bennett meet Mr. Darcy?",
    "answers": [
        {
        "text": "at a ball in the town of Meryton",
        "answer_start": 31
        }
    ]
    }
    """
    def preprocess_function(self, examples):
        # Truncate or pad to max_length, handle offsets for extractive QA
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        # TODO: Handle answers according to real our data structure

        return inputs

    def train(self, dataset, training_args, model):
        processed_dataset = dataset.map(self.preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
        model = model.from_pretrained(self.model_name)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"]
        )
        trainer.train()


num2title = {
    1: "Harry Potter and the Philosopher's Stone",
    2: "Harry Potter and the Chamber of Secrets",
    3: "Harry Potter and the Prisoner of Azkaban",
    4: "Harry Potter and the Goblet of Fire",
    5: "Harry Potter and the Order of the Phoenix",
    6: "Harry Potter and the Half-Blood Prince",
    7: "Harry Potter and the Deathly Hallows"
}

@dataclass
class Chapter:
    """
    Class to represent a chapter.
    Chapter number is the number in the Harry Potter series.
    Start from 1.
    Chapter name is the name of the chapter.
    Chapter text is strings.
    """
    chapter_num: int
    chapter_name: str
    chapter_text: str

@dataclass
class Book:
    """
    Class to represent a book.
    Book number is the number in the Harry Potter series.
    Start from 1 to 7.
    Title is the title of the book.
    Chapters is a list of Chapter objects.
    Structure of chapter text is a list of strings.
    """
    book_num: int
    title: str
    chapters: List[Chapter]


def normalize_special_chars(text):
    # Define a mapping of Unicode special characters to straight (ASCII) ones.
    mapping = {
        '\u2018': "'",    # left single quotation mark
        '\u2019': "'",    # right single quotation mark
        '\u201c': '"',    # left double quotation mark
        '\u201d': '"',    # right double quotation mark
        '\u2013': '-',    # en dash
        '\u2014': '-',    # em dash
        '\u2026': '...',  # ellipsis
        # Add additional mappings here if needed.
    }
    
    # Create a regex pattern that matches any of the keys in the mapping.
    pattern = re.compile('|'.join(re.escape(key) for key in mapping.keys()))
    
    # Replace each occurrence of a special character with its mapped ASCII version.
    normalized_text = pattern.sub(lambda m: mapping[m.group(0)], text)
    return normalized_text

def process_book_text(book_text, book_num):
    """
    Process the text of a Harry Potter book.
    book_text: str, text of the book
    book_num: int, number of the book
    return: Book object
    """
    title = num2title[book_num]
    clean_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', book_text)
    normal_text = normalize_special_chars(clean_text)
    book = [line for line in normal_text.splitlines() if line]
    chapter_starts = [i for i, line in enumerate(book) if line.startswith("Chapter")]
    # chapter name is the next line after "Chapter"
    chapter_names = [book[i + 1] for i in chapter_starts]
    chapter_texts = []
    for i in range(len(chapter_starts) - 1):
        chapter_texts.append('\n'.join(book[(chapter_starts[i]+2):chapter_starts[i + 1]]) + '\n')
    chapter_texts.append('\n'.join(book[(chapter_starts[-1]+2):]) + '\n')
    # return Book class
    chapters = [Chapter(i + 1, chapter_names[i], chapter_texts[i]) for i in range(len(chapter_names))]
    return Book(book_num, title, chapters)


class TextChunker:

    def __init__(self, encoder_model_name='sentence-transformers/all-mpnet-base-v1'):
        self.model = SentenceTransformer(encoder_model_name)

    def process_text(self, text, context_window=5, similarity_percentile_threshold=30, min_chunk_size=5, keep_natural_order=True):
        """
        Process text.
        """
        sentences = self._tokenize(text)
        context_sentences = self._add_context(sentences, context_window)
        embeddings = self.model.encode(context_sentences)

        similarity_matrix = self._calculate_similarity(embeddings)
        breakpoints = self._identify_breakpoints(similarity_matrix, similarity_percentile_threshold)
        init_chunks = self._create_chunks(sentences, breakpoints)
        final_chunks = self._merge_small_chunks(init_chunks, min_chunk_size, keep_natural_order)

        return final_chunks

    def _tokenize(self, text):
        return sent_tokenize(text)
    
    def _add_context(self, sentences, window_size):
        """
        Add context to sentences.
        """
        n = len(sentences)
        context_sentences = []
        for i in range(n):
            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)
            context = sentences[start:i] + sentences[i:end]
            context_sentences.append(context)
        return context_sentences
    
    def _calculate_similarity(self, embeddings, bAllPairs=False):
        """
        Calculate similarity matrix of embeddings.

        embeddings: np.array, shape (n, d), n is the number of embeddings, d is the dimension of embeddings.
        bAllPairs: bool, whether to calculate all pairs of similarities. Default is False. If False, only calculate the adjacent ones.
        return: np.array, shape (n, n), similarity matrix, upper triangular.
        """
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            end_point = n if bAllPairs else min(i + 2, n)
            for j in range(i, end_point):
                similarity_matrix[i, j] = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        return similarity_matrix
    
    def _identify_breakpoints(self, similarity_matrix, threshold_percentile):
        """
        Identify breakpoints in a similarity matrix.

        similarity_matrix: np.array, shape (n, n), similarity matrix
        threshold_percentile: float, percentile threshold
        return: List[int], breakpoints
        """
        n = similarity_matrix.shape[0]
        adjacent_similarities = np.diag(similarity_matrix, k=1)
        threshold = np.percentile(adjacent_similarities, threshold_percentile)
        breakpoints = [0]
        for i in range(n - 1):
            if similarity_matrix[i, i + 1] < threshold:
                breakpoints.append(i)
        breakpoints.append(n)
        return breakpoints
    
    def _create_chunks(self, sentences, breakpoints):
        """
        Create chunks based on breakpoints.

        sentences: List[str], sentences
        breakpoints: List[int], breakpoints
        return: List[str], chunks
        """
        chunks = []
        for i in range(len(breakpoints) - 1):
            chunk = ' '.join(sentences[breakpoints[i]:breakpoints[i + 1]])
            chunks.append(chunk)
        return chunks
    
    def _merge_small_chunks(self, chunks, min_chunk_size, keep_natural_order):
        """
        Merge small chunks with their most similar neighbors.
        """
        n = len(chunks)
        if n == 1:
            return chunks
        if keep_natural_order:
            return self._merge_small_chunks_natural_order(chunks, min_chunk_size)
        else:
            return self._merge_small_chunks_similarity_order(chunks, min_chunk_size)
        
    def _merge_small_chunks_natural_order(self, chunks, min_chunk_size):
        """
        Merge small chunks with their most similar neighbors in natural order.
        """
        embeddings = self.model.encode(chunks)
        similarity_matrix = self._calculate_similarity(embeddings)
        n = len(chunks)
        merged_chunks = []
        i = 0
        last_chunk_size = 0
        while i < n:
            this_chunk_size = len(self._tokenize(chunks[i]))
            if i == 0:
                merged_chunks.append(chunks[i])
                last_chunk_size = this_chunk_size
            elif this_chunk_size < min_chunk_size:
                if i == n - 1:
                    merged_chunks[-1] += ' ' + chunks[i]
                    last_chunk_size += this_chunk_size
                elif similarity_matrix[i - 1, i] < similarity_matrix[i, i + 1]:
                    if last_chunk_size < min_chunk_size:
                        merged_chunks[-1] += ' ' + chunks[i] + ' ' + chunks[i + 1]
                        last_chunk_size += (this_chunk_size + len(self._tokenize(chunks[i + 1])))
                    else:
                        merged_chunks.append(chunks[i] + ' ' + chunks[i + 1])
                        last_chunk_size = this_chunk_size + len(self._tokenize(chunks[i + 1]))
                    i += 1
                else:
                    merged_chunks[-1] += ' ' + chunks[i]
                    last_chunk_size += this_chunk_size
            elif last_chunk_size < min_chunk_size:
                merged_chunks[-1] += ' ' + chunks[i]
                last_chunk_size += this_chunk_size
            else:
                merged_chunks.append(chunks[i])
                last_chunk_size = this_chunk_size
            i += 1
        return merged_chunks
    
    def _merge_small_chunks_similarity_order(self, chunks, min_chunk_size):
        """
        Iteratively merges the most similar pair of chunks (where at least one chunk is below
        the length threshold) until every chunk is at or above the threshold.
        
        Parameters:
        chunks (list of str): List of text chunks.
        min_chunk_size (int): Minimum length for a chunk (e.g., measured in characters).
        
        Returns:
        list of str: The merged list of chunks.
        """
        while True:
            # If all chunks are above threshold, we're done.
            if all(len(self._tokenize(chunk)) >= min_chunk_size for chunk in chunks):
                break
            
            # Compute embeddings for all chunks
            embeddings = self.model.encode(chunks)
            similarity_matrix = self._calculate_similarity(embeddings, True)
            
            best_sim = -1
            best_pair = None
            
            # Identify indices of chunks that are below threshold
            small_indices = {i for i, chunk in enumerate(chunks) if len(self._tokenize(chunk)) < min_chunk_size}
            
            # Iterate over all pairs; consider pairs where at least one is small.
            for i in range(len(chunks)):
                for j in range(i+1, len(chunks)):
                    if i in small_indices or j in small_indices:
                        if similarity_matrix[i][j] > best_sim:
                            best_sim = similarity_matrix[i][j]
                            best_pair = (i, j)
                            
            # If we found no pair to merge, exit.
            if best_pair is None:
                break
            
            i, j = best_pair
            # Merge the two chunks (order here is arbitrary since order is not preserved)
            merged_chunk = chunks[i] + " " + chunks[j]
            
            # Remove the merged chunks and add the new merged chunk
            new_chunks = [chunk for idx, chunk in enumerate(chunks) if idx not in best_pair]
            new_chunks.append(merged_chunk)
            chunks = new_chunks
            
        return chunks
    
def make_chunked_json_file(book_file, output_file, encoder_name, context_window=5, similarity_percentile_threshold=30, min_chunk_size=5, keep_natural_order=True):
    """
    Make a chunked json file from a Harry Potter book text file.
    book_file: str, path to the book text file
    output_file: str, path to the output json file
    encoder_name: TextChunker object model name
    context_window: int, context window size
    similarity_percentile_threshold: float, percentile threshold for similarity
    min_chunk_size: int, minimum chunk size
    keep_natural_order: bool, whether to keep natural order
    """
    chunker = TextChunker(encoder_name)
    with open(book_file, "r") as f:
        book_text = f.read()
    book = process_book_text(book_text, int(book_file.split('.')[0][-1]))
    with open(output_file, "w") as f:
        title = book.title
        title_num = book.book_num
        for chapter in book.chapters:
            this_chunk = chunker.process_text(chapter.chapter_text, context_window, similarity_percentile_threshold, min_chunk_size, keep_natural_order)
            for passage in this_chunk:
                json.dump({"title_num" : title_num, "title": title, "chapter_num" : chapter.chapter_num, "chapter_name": chapter.chapter_name, "passage": passage}, f)
                f.write("\n")

# Define a wrapper that unpacks the tuple and instantiates the chunker in the worker.
def make_chunked_json_file_wrapper(args_tuple):
    (input_file, output_file, encoder_model_name, context_window,
     similarity_threshold, min_chunk_size, keep_natural_order) = args_tuple
    return make_chunked_json_file(input_file, output_file, encoder_model_name, context_window, similarity_threshold, min_chunk_size, keep_natural_order)


if __name__ == "__main__":
    make_chunked_json_file("data/hp1.txt", "data/hp1_chunked.json", "sentence-transformers/all-mpnet-base-v1")
    print("Done!")
