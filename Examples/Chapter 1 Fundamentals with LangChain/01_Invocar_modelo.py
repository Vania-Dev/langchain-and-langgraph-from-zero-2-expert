# Importamos ChatOllama desde langchain_ollama
# ChatOllama es un modelo de chat basado en Ollama que permite enviar mensajes tipo chat.
from langchain_ollama import ChatOllama

# Nombre del modelo que vamos a usar
# "llama3.2:3b" es un modelo LLaMA 3 instructivo de 3B parámetros.
model_name = "llama3.2:3b"

# Creamos una instancia del modelo ChatOllama
# streaming=True permite que la respuesta se genere y se imprima en tiempo real.
model = ChatOllama(model=model_name, streaming=True)  # Habilita streaming

# Enviamos un mensaje al modelo usando invoke()
# Aquí le enviamos la cadena "Hola" y obtenemos la respuesta.
# .content extrae solo el texto de la respuesta del modelo.
print(model.invoke("Hola").content)
