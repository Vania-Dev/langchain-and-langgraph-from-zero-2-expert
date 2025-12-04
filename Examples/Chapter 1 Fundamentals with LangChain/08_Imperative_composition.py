# Importamos ChatOllama para interactuar con modelos de Ollama
from langchain_ollama import ChatOllama
# Importamos ChatPromptTemplate para crear plantillas de conversación
from langchain_core.prompts import ChatPromptTemplate
# Importamos el decorador chain para convertir funciones en Runnables
from langchain_core.runnables import chain

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

# COMPOSICIÓN IMPERATIVA: Combinamos los componentes en una función
# El decorador @chain convierte cualquier función en un Runnable
# Esto significa que la función tendrá métodos como .invoke(), .batch(), .stream()
@chain
def chatbot(values):
    # Paso 1: Generamos el prompt usando la plantilla
    prompt = template.invoke(values)
    # Paso 2: Enviamos el prompt al modelo y devolvemos la respuesta
    return model.invoke(prompt)

# USANDO EL CHATBOT: Ahora podemos usar chatbot.invoke() como cualquier Runnable
response = chatbot.invoke({"question": "Which model providers offer LLMs?"})
print(response.content)

print("\n=== STREAMING VERSION ===")

# VERSIÓN CON STREAMING: Creamos una versión que soporta streaming
# yield permite que la función genere tokens uno por uno
@chain
def chatbot_stream(values):
    # Generamos el prompt
    prompt = template.invoke(values)
    # Iteramos sobre cada token del stream y lo devolvemos
    for token in model.stream(prompt):
        yield token

# Usamos el método .stream() del chatbot para obtener respuestas en tiempo real
for part in chatbot_stream.stream({
    "question": "Which model providers offer LLMs?"
}):
    print(part.content, end="", flush=True)

# Asincrono
@chain
async def asin_chatbot(values):
    prompt = await template.ainvoke(values)
    return await model.ainvoke(prompt)
response = await asin_chatbot.ainvoke({"question": "Which model providers offer LLMs?"})
print(response)

# > AIMessage(content="""Hugging Face's `transformers` library, OpenAI using    the `openai` library, and Cohere using the `cohere` library offer LLMs.""")