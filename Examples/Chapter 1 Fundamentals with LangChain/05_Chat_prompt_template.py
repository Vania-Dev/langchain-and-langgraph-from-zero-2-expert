# Importamos ChatOllama para interactuar con modelos de Ollama
from langchain_ollama import ChatOllama
# Importamos ChatPromptTemplate para crear plantillas de conversación estructuradas
from langchain_core.prompts import ChatPromptTemplate

# Creamos una plantilla de conversación con diferentes roles de mensajes
# ChatPromptTemplate permite definir una secuencia de mensajes con roles específicos
# 'system': Define el comportamiento y las instrucciones para el asistente
# 'human': Representa los mensajes del usuario
template = ChatPromptTemplate.from_messages([
('system', '''Answer the question based on the context below. If the
question cannot be answered using the information provided, answer with
"I don\'t know".'''),

('human', 'Context: {context}'),

('human', 'Question: {question}'),

])

# Definimos el modelo que vamos a usar
model_name = "llama3.2:3b"

# Creamos una instancia del modelo ChatOllama
# streaming=True permite que la respuesta se genere y se imprima en tiempo real.
model = ChatOllama(model=model_name, streaming=True)  # Habilita streaming

# Invocamos la plantilla de chat con valores específicos
# Esto genera una lista de mensajes estructurados con los roles apropiados
# Las variables {context} y {question} son reemplazadas con los valores proporcionados
prompt = template.invoke({
    "context": """The most recent advancements in NLP are being driven by Large
        Language Models (LLMs). These models outperform their smaller
        counterparts and have become invaluable for developers who are creating
        applications with NLP capabilities. Developers can tap into these
        models through Hugging Face's `transformers` library, or by utilizing
        OpenAI and Cohere's offerings through the `openai` and `cohere`
        libraries, respectively.""",
    "question": "Which model providers offer LLMs?"
})

# Enviamos la conversación estructurada al modelo
response = model.invoke(prompt)

# Imprimimos la respuesta completa (incluye metadatos)
# Para obtener solo el texto, usar response.content
print(response)