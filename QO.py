# Import the QO class from utils.py
from utils import QO

# Assume we have an LLM instance that supports a .generate() method.
# This could be any language model interface you have implemented.
# For demonstration purposes, we create a dummy LLM:
class DummyLLM:
    def generate(self, prompt: str) -> str:
        # In a real scenario, replace this with an actual call to your LLM
        return f"Generated text based on: {prompt}"

# Create an instance of your dummy LLM and the QO class
llm = DummyLLM()
qo = QO(embedding_model_name="sentence-transformers/all-mpnet-base-v1", llm=llm)

# Example query needs optimization
query = "When was Hogwarts founded and who founded it?"

# --- Query Transformation ---
# 1. Use HyDE to generate an embedding from a hypothetical document
try:
    embedding = qo.hyde(query)
    print("HyDE Embedding:", embedding)
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
        1: "Philosopher's Stone",
        2: "Chamber of Secrets",
        3: "Prisoner of Azkaban"
        # more books...
    }
}
filters = qo.metadata_filter(query, available_meta)
print("Metadata Filters:", filters)

# --- Query Expansion ---
# 1. Generate multiple query variants
query_variants = qo.multi_query(query)
print("Query Variants:")
for variant in query_variants:
    print(" -", variant)

# 2. Decompose the query into sub-questions if it is compound
sub_questions = qo.sub_questions(query)
print("Sub-Questions:")
for subq in sub_questions:
    print(" -", subq)

# The optimized queries or embeddings can now be used by the QA system (QA.py)
# or to run NER tasks (NER.py), further integrating with the overall system.
