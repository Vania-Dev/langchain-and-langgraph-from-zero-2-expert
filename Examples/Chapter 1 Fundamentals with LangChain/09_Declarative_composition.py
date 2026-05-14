# Importamos ChatOllama para interactuar con modelos de Ollama
from langchain_ollama import ChatOllama
# Importamos ChatPromptTemplate para crear plantillas de conversación
from langchain_core.prompts import ChatPromptTemplate

# BLOQUES DE CONSTRUCCIÓN (Building Blocks)
# Creamos una plantilla de chat simple con un mensaje de sistema y uno humano
template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant.'),
    ('human', '{question}'),
])

# Definimos el modelo que vamos a usar
model_name = "llama3.2:3b"
# Creamos una instancia del modelo ChatOllama
# streaming=True permite que las respuesta se genere y se imprima en tiempo real.
model = ChatOllama(model=model_name, streaming=True)  # Habilita streaming

chatbot = template | model

response = chatbot.invoke({"question": "Which model providers offer LLMs?"})

print(response)