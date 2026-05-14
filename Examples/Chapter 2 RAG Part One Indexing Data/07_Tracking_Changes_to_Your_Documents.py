# Import required libraries for document indexing and vector storage
from langchain_classic.indexes import SQLRecordManager, index  # Track indexed documents
from langchain_postgres import PGVectorStore, PGEngine  # PostgreSQL vector database
from langchain_ollama import OllamaEmbeddings  # Ollama embeddings model
from langchain_core.documents import Document  # Document structure
import hashlib
import uuid

# SHA-256 encoder for secure document hashing (replaces insecure SHA-1)
def sha256_encoder(doc: Document) -> str:
    hash_hex = hashlib.sha256(doc.page_content.encode()).hexdigest()
    return str(uuid.UUID(hash_hex[:32]))

# Database connection and configuration
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "my_docs"
embeddings_model = OllamaEmbeddings(model="llama3.2:3b")
namespace = "my_docs_namespace"  # Namespace for record manager

# Initialize database engine
engine = PGEngine.from_connection_string(url=connection)

# Initialize the vectorstore table with langchain_id column
try:
    engine.init_vectorstore_table(
        table_name=collection_name,
        vector_size=3072,  # Vector size for llama3.2:3b embeddings
    )
except Exception:
    pass  # Table already exists

# Create vectorstore instance
vectorstore = PGVectorStore.create_sync(
    engine=engine,
    table_name=collection_name,
    embedding_service=embeddings_model,
)

# Create a record manager to track which documents have been indexed
record_manager = SQLRecordManager(
    namespace,
    db_url="postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
)

# Create the schema if it doesn't exist
record_manager.create_schema()

# Create sample documents
docs = [
    Document(page_content='there are cats in the pond', metadata={
        "id": 1, "source": "cats.txt"}),
    Document(page_content='ducks are also found in the pond', metadata={
        "id": 2, "source": "ducks.txt"}),
]

# First indexing attempt - adds new documents
index_1 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",  # Prevent duplicate documents
    source_id_key="source",  # Use the source field as the source_id
    key_encoder=sha256_encoder,  # Use SHA-256 for secure hashing
)
print("Index attempt 1:", index_1)

# Second indexing attempt - no changes, documents already indexed
index_2 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
    key_encoder=sha256_encoder,
)
print("Index attempt 2:", index_2)

# Modify a document - the new version will be written and old version deleted
docs[0].page_content = "I just modified this document!"
index_3 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
    key_encoder=sha256_encoder,
)
print("Index attempt 3:", index_3)