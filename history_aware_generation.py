from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

load_dotenv()

# ==============================
# 1. Connect to Vector DB
# ==============================
persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# ==============================
# 2. Load FLAN-T5
# ==============================
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_answer(prompt, max_length=512, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==============================
# 3. Chat History (simple list of dicts)
# ==============================
chat_history = []  # each entry: {"role": "user"/"assistant", "content": "..."}

def build_history_string():
    """Convert chat history to a plain text string for FLAN-T5"""
    if not chat_history:
        return ""
    lines = []
    for msg in chat_history[-6:]:  # only last 3 exchanges to stay within token limit
        prefix = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{prefix}: {msg['content']}")
    return "\n".join(lines)

# ==============================
# 4. Core Logic
# ==============================
def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Rewrite question to be standalone if there's history
    if chat_history:
        history_str = build_history_string()
        rewrite_prompt = (
            f"Chat history:\n{history_str}\n\n"
            f"New question: {user_question}\n\n"
            f"Rewrite the new question to be fully standalone and searchable without needing the chat history. "
            f"Return only the rewritten question."
        )
        search_question = generate_answer(rewrite_prompt, max_new_tokens=80).strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Retrieve relevant docs
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.3}
    )
    docs = retriever.invoke(search_question)

    # Fallback if nothing found above threshold
    if not docs:
        print("⚠️ No docs above threshold, using fallback...")
        retriever_fallback = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever_fallback.invoke(search_question)

    if not docs:
        print("❌ No documents found at all.")
        return "I don't have enough information to answer that question."

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"  Doc {i}: {preview}...")

    # Step 3: Build context from docs
    context = "\n\n".join([doc.page_content for doc in docs])
    context = context[:1500]  # stay within token limits

    # Step 4: Build final prompt with history + context
    history_str = build_history_string()
    if history_str:
        final_prompt = (
            f"Chat history:\n{history_str}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_question}\n"
            f"Answer based only on the context above. If the answer is not in the context, say 'I don't have enough information'.\n"
            f"Answer:"
        )
    else:
        final_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {user_question}\n"
            f"Answer based only on the context above. If the answer is not in the context, say 'I don't have enough information'.\n"
            f"Answer:"
        )

    # Step 5: Generate answer
    answer = generate_answer(final_prompt, max_new_tokens=200).strip()

    if not answer:
        answer = "I don't have enough information to answer that question based on the provided documents."

    # Step 6: Save to chat history
    chat_history.append({"role": "user", "content": user_question})
    chat_history.append({"role": "assistant", "content": answer})

    print(f"Answer: {answer}")
    return answer

# ==============================
# 5. Chat Loop
# ==============================
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)

if __name__ == "__main__":
    start_chat()


# Sythetic question
"What was Microsoft's first hardware product release?"
# what does it do?