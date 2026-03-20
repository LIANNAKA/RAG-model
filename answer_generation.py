from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

retriever = db.as_retriever(search_kwargs={"k": 5})

pipe = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200
)

# Wrap in LangChain
llm = HuggingFacePipeline(pipeline=pipe)
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the context below.

Context:
{context}

Question:
{question}

Answer:
""")

# LCEL Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

# Search for relevant documents
# query = "How much did Microsoft pay to acquire GitHub?"
query = "What was NVIDIA's first graphics accelerator called?"

print(f"\nUser Query: {query}")

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

response = rag_chain.invoke(query)

print("\n--- Generated Answer ---")
print(response["text"] if isinstance(response, dict) else response)

# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"