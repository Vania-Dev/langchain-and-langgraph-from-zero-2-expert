# Importamos ChatOllama para interactuar con modelos de Ollama
from langchain_ollama import ChatOllama
# Importamos PromptTemplate para crear plantillas de prompts reutilizables
from langchain_core.prompts import PromptTemplate

# Creamos una plantilla de prompt con variables dinámicas
# PromptTemplate permite crear prompts estructurados con placeholders {variable}
# Esto es útil para crear prompts consistentes que pueden reutilizarse con diferentes datos
template = PromptTemplate.from_template("""Answer the question based on the
    context below. If the question cannot be answered using the information
    provided, answer with "I don't know".
                                        
    Context: {context}
                                        
    Question: {question}
                                        
    Answer: """)

# Definimos el modelo que vamos a usar
model_name = "llama3.2:3b"

# Creamos una instancia del modelo ChatOllama
# streaming=True permite que la respuesta se genere y se imprima en tiempo real.
model = ChatOllama(model=model_name, streaming=True)  # Habilita streaming

# Invocamos la plantilla con valores específicos para las variables
# template.invoke() reemplaza {context} y {question} con los valores proporcionados
# Esto genera el prompt final que será enviado al modelo
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

# Enviamos el prompt generado al modelo y obtenemos la respuesta
completion = model.invoke(prompt)

# Imprimimos solo el contenido de texto de la respuesta
print(completion.content)