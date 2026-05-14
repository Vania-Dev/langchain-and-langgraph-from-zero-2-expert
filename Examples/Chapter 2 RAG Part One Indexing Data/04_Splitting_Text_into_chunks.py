from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
# Importamos la clase TextLoader para cargar archivos de texto
from langchain_community.document_loaders import TextLoader

loader = TextLoader("files/test.txt") # or any other loader
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
splitted_docs = splitter.split_documents(docs)
print(splitted_docs)

python_code = """
    def hello_world():
        print("Hello, World!")

    # Call the function
    hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([python_code])

print(python_docs)

markdown_text = """
    # LangChain
    ⚡ Building applications with LLMs through composability ⚡
    ## Quick Install
    ```bash
    pip install langchain
    ```
    As an open source project in a rapidly developing field, we are extremely open
    to contributions.
"""

md_splitter = RecursiveCharacterTextSplitter.from_language(
language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0
)
md_docs = md_splitter.create_documents([markdown_text],
[{"source": "https://www.langchain.com"}])

print(md_docs)