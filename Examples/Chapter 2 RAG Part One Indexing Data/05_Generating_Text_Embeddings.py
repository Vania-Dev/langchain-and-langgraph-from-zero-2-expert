from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Usa cualquier modelo compatible con embeddings, por ejemplo nomic-embed-text
emb_model = OllamaEmbeddings(model="llama3.2:3b")

embeddings = emb_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!",
    ]
)

print(embeddings)


loader = TextLoader("files/test.txt")
doc = loader.load()
## Split the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
)
chunks = text_splitter.split_documents(doc)
## Generate embeddings
embeddings_model = OllamaEmbeddings(model="llama3.2:3b")
embeddings = embeddings_model.embed_documents([chunk.page_content for chunk in chunks])

print(embeddings)
