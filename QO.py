# Import the QO class from utils.py
from utils import QO
import cohere
import os

# Assume we have an LLM instance that supports a .generate() method.
# This could be any language model interface you have implemented.
# For demonstration purposes, we create a dummy LLM:
class MyLLM:
    def generate(self, prompt: str) -> str:
        co = cohere.ClientV2(api_key="2YR7FbBNMB1pnucs4Jqc4sZlfF6G9Z65w2al0K6I")
        res = co.chat(
            model="command-a-03-2025",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return res.message.content[0].text


# Create an instance of your dummy LLM and the QO class
llm = MyLLM()
qo = QO(embedding_model_name="sentence-transformers/all-mpnet-base-v1", llm=llm)

# Example query needs optimization
# query = "When was Hogwarts founded and who founded it?"
# query = 'Which of the following is NOT one of Albus Dumbledore\'s titles?'
query = 'has Dumbledore visited Dursleys and sat down?'

# --- Query Transformation ---
# 1. Use HyDE to generate an embedding from a hypothetical document
try:
    embedding = qo.hyde(query)
    print("HyDE Embedding:", embedding)
    print('Embedding Successful!')
except ValueError as e:
    print(e)

# 2. Use Step-Back Prompting to generate a more abstract version of the query
abstract_query = qo.step_back(query)
print("Abstract Query:", abstract_query)

# --- Query Routing ---
# Define available sources with metadata keywords
sources = {
    "harry_potter": {"keywords": ["hogwarts", "harry potter", "wizard"]},
    "wikipedia": {"keywords": []}  # default/general source
}
selected_source = qo.semantic_routing(query, sources)
print("Selected Source:", selected_source)

# --- Metadata Filtering ---
# Define available metadata for filtering (for example, mapping book IDs to titles)
available_meta = {
    "book": {
        1: "Harry Potter and the Philosopher's Stone",
        2: "Harry Potter and the Chamber of Secrets",
        3: "Harry Potter and the Prisoner of Azkaban",
        4: "Harry Potter and the Goblet of Fire",
        5: "Harry Potter and the Order of the Phoenix",
        6: "Harry Potter and the Half-Blood Prince",
        7: "Harry Potter and the Deathly Hallows"
    }
}
filters = qo.metadata_filter(query, available_meta)
print("Metadata Filters:", filters)

# --- Query Expansion ---
# 1. Generate multiple query variants
query_variants = qo.multi_query(query)
# query_variants = qo.multi_query(abstract_query)
print("Query Variants:")
for variant in query_variants:
    print(" -", variant)

# 2. Decompose the query into sub-questions if it is compound
# sub_questions = qo.sub_questions(query)
# sub_questions = qo.sub_questions(abstract_query)

# Split by lines and search for the first variant that starts with "1."
# lines = query_variants.splitlines()
# first_variant_line = next(line for line in lines if line.strip().startswith("1."))
# Remove the numbering and quotes if needed:
# first_variant = first_variant_line.split("1.", 1)[1].strip().strip('"')
# print(first_variant)

variants_str = "\n".join(query_variants)
lines = variants_str.splitlines()
print(lines[3][2:])

sub_questions = qo.sub_questions(lines[3][2:])

print("Sub-Questions:")
for subq in sub_questions:
    print(" -", subq)

# The optimized queries or embeddings can now be used by the QA system (QA.py)
# or to run NER tasks (NER.py), further integrating with the overall system.
