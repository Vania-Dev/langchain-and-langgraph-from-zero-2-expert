# Docker command to run PostgreSQL with pgvector extension
# This creates a container with vector database capabilities
# docker run \
# --name pgvector-container \
# -e POSTGRES_USER=langchain \
# -e POSTGRES_PASSWORD=langchain \
# -e POSTGRES_DB=langchain \
# -p 6024:5432 \
# -d pgvector/pgvector:pg16

# Import required libraries for document processing and vector storage
from langchain_community.document_loaders import TextLoader  # Loads text files
from langchain_ollama import OllamaEmbeddings  # Creates embeddings using Ollama models
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits text into chunks
from langchain_postgres import PGVectorStore, PGEngine  # PostgreSQL vector database components
from langchain_core.documents import Document
import uuid

# STEP 1: Load and prepare documents
raw_documents = TextLoader('files/test.txt').load()  # Load the text file into memory
text_splitter = RecursiveCharacterTextSplitter(  # Create a text splitter
    chunk_size=1000,  # Each chunk will be max 1000 characters
    chunk_overlap=200  # 200 characters overlap between chunks for context
)
documents = text_splitter.split_documents(raw_documents)  # Split the document into smaller chunks
print(documents)  # Display the chunks that were created

# STEP 2: Set up embedding model and database connection
embeddings_model = OllamaEmbeddings(model="llama3.2:3b")  # Initialize Ollama embedding model
connection_string = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'  # Database connection URL
engine = PGEngine.from_connection_string(url=connection_string)  # Create database engine

# STEP 3: Initialize database table (only if it doesn't exist)
try:
    engine.init_vectorstore_table(  # Try to create the vector table
        table_name="documents_v2",  # Name of the table to store vectors
        vector_size=3072,  # Size of vectors (must match Ollama model output)
    )
except Exception:
    pass  # If table already exists, ignore the error and continue

# STEP 4: Create vector store instance
store = PGVectorStore.create_sync(  # Create a vector store object
    engine=engine,  # Use the database engine we created
    table_name="documents_v2",  # Connect to our table
    embedding_service=embeddings_model,  # Use Ollama for creating embeddings
)

# STEP 5: Add documents to vector store
store.add_documents(documents)  # Convert text chunks to vectors and store in database
print(f"Added {len(documents)} documents to vector store")  # Confirm how many documents were added

# STEP 6: Test similarity search
print(store.similarity_search("query", k=4))  # Search for similar documents to "query" and return top 4 results

# STEP 7: Adding more documents with specific IDs
ids = [str(uuid.uuid4()), str(uuid.uuid4())]  # Generate proper UUID strings
store.add_documents(
    [
        Document(
            page_content="there are cats in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
        Document(
            page_content="ducks are also found in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
    ],
    ids=ids,  # Use the UUID strings we generated
)

# STEP 8: Delete documents using proper UUID format
store.delete(ids=[ids[0]])  # Delete the first document using its UUID