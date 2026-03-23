from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv

load_dotenv()

# ==============================
# 1. Load Vector DB
# ==============================
persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# ==============================
# 2. Retriever
# ==============================
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.3}
)

# ==============================
# 3. Load FLAN-T5 directly (no pipeline task string needed)
# ==============================
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==============================
# 4. Query
# ==============================
# query = "What was NVIDIA's first graphics accelerator called?"
query = "What was Microsoft's first hardware product release?"
print(f"\nUser Query: {query}")

# ==============================
# 5. Retrieval + Fallback
# ==============================
docs = retriever.invoke(query)

if not docs:
    print("⚠️ No docs found above threshold, using fallback...")
    retriever_fallback = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever_fallback.invoke(query)

if not docs:
    print("❌ No documents found at all. Check your vector DB.")
    exit()

# ==============================
# 6. Build Context
# ==============================
context = "\n\n".join([doc.page_content for doc in docs])
context = context[:1500]

print("\n--- Retrieved Context (Debug) ---")
print(context[:500])

# ==============================
# 7. Prompt
# ==============================
final_prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

# ==============================
# 8. Generate Answer
# ==============================
response = generate_answer(final_prompt)

print("\n--- Generated Answer ---")
print(response if response else "⚠️ Model returned empty string")

# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"