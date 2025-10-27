# Importamos ChatOllama desde langchain_ollama
# ChatOllama permite interactuar con modelos de Ollama usando un estilo de chat con roles de mensajes.
from langchain_ollama import ChatOllama

# Importamos HumanMessage y SystemMessage desde langchain.messages
# HumanMessage representa un mensaje enviado por el usuario.
# SystemMessage representa un mensaje de sistema que define el comportamiento o rol del asistente.
from langchain.messages import HumanMessage, SystemMessage

# Definimos el nombre del modelo que vamos a usar
# "llama3.2:3b" es un modelo instructivo de LLaMA 3 de 3B parámetros.
model_name = "llama3.2:3b"

# Creamos una instancia del modelo ChatOllama
# streaming=True permite que la respuesta se genere en tiempo real si se usan callbacks.
model = ChatOllama(
    model=model_name,
    streaming=True
)

# Creamos la lista de mensajes de la conversación
# El primer mensaje es un SystemMessage que define el rol del asistente.
# El segundo mensaje es un HumanMessage que representa la pregunta del usuario.
messages = [
    SystemMessage(content="Eres un asistente que ayuda a contestar las preguntas con signos de exclamación al inicio y final."),
    HumanMessage(content="Cual es la capital de mexico")
]
1
# Invocamos el modelo con la lista de mensajes
# invoke() envía los mensajes al modelo y devuelve un objeto de respuesta.
response = model.invoke(messages)

# Imprimimos la respuesta final generada por el modelo
print("\nRespuesta final:")
print(response.content)
