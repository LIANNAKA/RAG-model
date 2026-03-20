import os
import re
#  read text files, docx files
from langchain_community.document_loaders import TextLoader, DirectoryLoader
#  for chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter
#  embedding model
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings
#  vector database - chroma db to host it locally
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

#  cleaning function
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove citations like [1], [23]
    text = re.sub(r'\[\d+\]', '', text)

    # Remove "Archived", "Retrieved", etc.
    text = re.sub(r'Archived.*?Retrieved.*?\.', '', text, flags=re.DOTALL)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# load documents
def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"{docs_path} does not exist.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError("No .txt files found.")

    # Clean + add metadata
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

        # Extract company name from filename
        filename = os.path.basename(doc.metadata["source"])
        company = filename.split(".")[0]

        doc.metadata["company"] = company

    print(f"Loaded {len(documents)} documents")
    return documents

def split_documents(documents):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,          # smaller = better retrieval
        chunk_overlap=100        # VERY IMPORTANT
    )
    
    chunks = text_splitter.split_documents(documents)

    
    print(f"Created {len(chunks)} chunks")
    return chunks
    
    # if chunks:
    
        # for i, chunk in enumerate(chunks[:5]):
        #     print(f"\n--- Chunk {i+1} ---")
        #     print(f"Source: {chunk.metadata['source']}")
        #     print(f"Length: {len(chunk.page_content)} characters")
        #     print(f"Content:")
        #     print(chunk.page_content)
        #     print("-" * 50)
        
        # if len(chunks) > 5:
        #     print(f"\n... and {len(chunks) - 5} more chunks")
    
    # return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
        
        # embedding model
        
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    # embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main(): 
    print("Main Function")

    #1. Loading the files
    documents = load_documents(docs_path="docs")  
    #2. Chunking the fiels
    chunks = split_documents(documents)
    #3. Embedding and storing in vector db
    vectorstore = create_vector_store(chunks)
if __name__ == "__main__":
    main()


# documents = [
#    Document(
#        page_content="Google LLC is an American multinational corporation and technology company focusing on online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, consumer electronics, and artificial intelligence (AI).",
#        metadata={'source': 'docs/google.txt'}
#    ),
#    Document(
#        page_content="Microsoft Corporation is an American multinational corporation and technology conglomerate headquartered in Redmond, Washington.",
#        metadata={'source': 'docs/microsoft.txt'}
#    ),
#    Document(
#        page_content="Nvidia Corporation is an American technology company headquartered in Santa Clara, California.",
#        metadata={'source': 'docs/nvidia.txt'}
#    ),
#    Document(
#        page_content="Space Exploration Technologies Corp., commonly referred to as SpaceX, is an American space technology company headquartered at the Starbase development site in Starbase, Texas.",
#        metadata={'source': 'docs/spacex.txt'}
#    ),
#    Document(
#        page_content="Tesla, Inc. is an American multinational automotive and clean energy company headquartered in Austin, Texas.",
#        metadata={'source': 'docs/tesla.txt'}
#    )
# ]