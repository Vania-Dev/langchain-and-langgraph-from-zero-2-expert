from langchain_community.document_loaders import TextLoader  # Loads text files
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits text into chunks
from langchain_ollama import OllamaEmbeddings  # Ollama embeddings model
from langchain_postgres import PGVectorStore, PGEngine  # PostgreSQL vector database
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.documents import Document  # Document structure
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.stores import InMemoryStore
import uuid

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "summaries"
embeddings_model = OllamaEmbeddings(model="llama3.2:3b")

# Load the document
loader = TextLoader("files/test.txt", encoding="utf-8")
docs = loader.load()
print("length of loaded docs: ", len(docs[0].page_content))

# Split the document
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# The rest of your code remains the same, starting from:
prompt_text = "Summarize the following document:\n\n{doc}"
prompt = ChatPromptTemplate.from_template(prompt_text)
llm = ChatOllama(temperature=0, model="llama3.2:3b")
summarize_chain = {"doc": lambda x: x.page_content} | prompt | llm | StrOutputParser()

# batch the chain across the chunks
summaries = summarize_chain.batch(chunks, {"max_concurrency": 5})

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

# The vectorstore to use to index the child chunks
vectorstore = PGVectorStore.create_sync(
    engine=engine,
    table_name=collection_name,
    embedding_service=embeddings_model,
)
# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"
# indexing the summaries in our vector store, whilst retaining the original
# documents in our document store:
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)
# Changed from summaries to chunks since we need same length as docs
doc_ids = [str(uuid.uuid4()) for _ in chunks]
# Each summary is linked to the original document by the doc_id
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]
# Add the document summaries to the vector store for similarity search
retriever.vectorstore.add_documents(summary_docs)
# Store the original documents in the document store, linked to their summaries
# via doc_ids
# This allows us to first search summaries efficiently, then fetch the full
# docs when needed
retriever.docstore.mset(list(zip(doc_ids, chunks)))
# vector store retrieves the summaries
sub_docs = retriever.vectorstore.similarity_search("chapter on philosophy", k=2)
# Whereas the retriever will return the larger source document chunks:
retrieved_docs = retriever.invoke("chapter on philosophy")
print(retrieved_docs)