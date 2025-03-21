import torch
import numpy as np
from transformers import GPTNeoXForCausalLM, AutoTokenizer


def embed_text(text):
        inputs = tokenizer(text, return_tensors="pt")  # PyTorch tensors
        with torch.no_grad():  # No gradient computation needed
            outputs = model(**inputs, output_hidden_states=True)
        
        # Get last hidden state
        last_hidden_state = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_dim)

        # Average pooling over the sequence length dimension (dim=1)
        sentence_embedding = last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_dim)
        
        return sentence_embedding  # Tensor of shape: (1, hidden_dim)

# Define a function to compute cosine similarity between two embeddings
def cosine_similarity(a, b):
    # Normalize vectors
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)

    # Compute cosine similarity (dot product since they're normalized)
    sim = torch.sum(a_norm * b_norm, dim=1)

    return sim  # Tensor with similarity score

if __name__ == "__main__":

    # Load model and tokenizer (PyTorch by default)
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-1b-deduped",
        revision="step143000",  # Use the final checkpoint
        cache_dir="./generation/pythia-1b-deduped/step143000",
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-1b-deduped",
        revision="step143000",
        cache_dir="generation/pythia-1b-deduped/step143000",
    )


    # Example dataset: a list of document chunks
    data = [
        {
            "title_num": 1,
            "title": "Harry Potter and the Philosopher's Stone",
            "chapter_num": 1,
            "chapter_name": "The Boy Who Lived",
            "passage": (
                "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say "
                "that they were perfectly normal, thank you very much. They were the last "
                "people you'd expect to be involved in anything strange or mysterious, because "
                "they just didn't hold with such nonsense. Mr. Dursley was the director of a firm "
                "called Grunnings, which made drills. He was a big, beefy man with hardly any neck, "
                "although he did have a very large mustache. Mrs. Dursley was thin and blonde and had "
                "nearly twice the usual amount of neck, which came in very useful as she spent so much "
                "of her time craning over garden fences, spying on the neighbors."
            )
        }
    ]

    # Precompute and store embeddings for all passages
    embeddings = []
    for chunk in data:
        passage = chunk["passage"]
        emb = embed_text(passage)
        embeddings.append(emb)

    

    # --- Retrieval and QA Pipeline ---

    # 1. Define your question
    question = "What is Mr. Dursley’s role at Grunnings?"

    # 2. Compute the embedding for the question
    question_emb = embed_text(question)

    # 3. Compare with stored passage embeddings to find the most relevant passage
    similarities = []
    for emb in embeddings:
        sim = cosine_similarity(question_emb, emb)
        similarities.append(sim.item())  # Convert tensor to scalar float

    best_idx = np.argmax(similarities)
    retrieved_passage = data[best_idx]["passage"]

    # 4. Construct a prompt that includes the retrieved passage
    prompt = (
        "You are an assistant with expert knowledge of the Harry Potter series. "
        "Based on the following passage, answer the question concisely in one sentence.\n\n"
        "Passage:\n" + retrieved_passage + "\n\n"
        "Question: " + question + "\n"
        "Answer:"
    )

    # 5. Tokenize and generate the answer
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=250,
            pad_token_id=tokenizer.eos_token_id,  # Should match your tokenizer's eos_token_id
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Generated Text:\n", generated_text)