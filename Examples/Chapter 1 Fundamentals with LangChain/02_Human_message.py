# Importamos ChatOllama desde langchain_ollama
# ChatOllama permite interactuar con modelos de Ollama usando un estilo de chat.
from langchain_ollama import ChatOllama

# Importamos HumanMessage desde langchain.messages
# HumanMessage representa un mensaje enviado por el usuario al modelo.
from langchain.messages import HumanMessage

# Definimos el nombre del modelo que vamos a usar
# "llama3.2:3b" es un modelo instructivo de LLaMA 3 de 3B parámetros.
model_name = "llama3.2:3b"

# Creamos una instancia del modelo ChatOllama
# streaming=True permite que la respuesta se genere en tiempo real.
model = ChatOllama(model=model_name, streaming=True)  # Habilita streaming

# Creamos el mensaje que vamos a enviar al modelo
# Aquí usamos HumanMessage para indicar que este mensaje viene del usuario.
prompt = [HumanMessage("Cual es la capital de mexico?")]

# Invocamos el modelo con la lista de mensajes
# invoke() reemplaza el uso de __call__, y devuelve un objeto de respuesta.
# .content extrae únicamente el texto generado por el modelo.
print(model.invoke(prompt).content)
