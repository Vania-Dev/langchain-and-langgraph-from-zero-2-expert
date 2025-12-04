# REQUISITO: Instalar beautifulsoup4 con: pip install beautifulsoup4
# Esta librería es necesaria para extraer contenido de páginas web

# Importamos WebBaseLoader para cargar contenido desde sitios web
from langchain_community.document_loaders import WebBaseLoader

# Creamos un cargador que apunta a una URL específica
# Cambia la URL por cualquier página web que quieras procesar
loader = WebBaseLoader("https://www.langchain.com/")

# Cargamos el contenido de la página web
# Esto descarga y extrae el texto de la página HTML
loader.load()

# Mostramos el contenido extraído
# El texto se limpia automáticamente de etiquetas HTML
print(loader.load())