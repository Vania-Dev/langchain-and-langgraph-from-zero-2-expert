# Importamos la clase TextLoader para cargar archivos de texto
from langchain_community.document_loaders import TextLoader

# Creamos un cargador de documentos que apunta a nuestro archivo de texto
# El archivo debe existir en la ruta especificada
loader = TextLoader("./files/test.txt")

# Cargamos el documento - esto lee el archivo y lo convierte en formato LangChain
loader.load()

# Imprimimos el contenido cargado para ver el resultado
# El documento se convierte en un objeto Document con metadatos
print(loader.load())