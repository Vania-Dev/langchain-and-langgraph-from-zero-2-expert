# REQUISITO: Instalar la librería para procesar PDFs
# Ejecuta en terminal: pip install pypdf

# Importamos PyPDFLoader para trabajar con archivos PDF
from langchain_community.document_loaders import PyPDFLoader

# Creamos un cargador que apunta a nuestro archivo PDF
# Asegúrate de que el archivo PDF existe en la ruta especificada
loader = PyPDFLoader("./files/OCR.pdf")

# Cargamos todas las páginas del PDF
# Cada página se convierte en un documento separado
pages = loader.load()

# Imprimimos todas las páginas procesadas
# Cada elemento de la lista representa una página con su texto extraído
print(pages)