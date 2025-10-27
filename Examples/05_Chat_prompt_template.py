from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
('system', '''Answer the question based on the context below. If the
question cannot be answered using the information provided, answer with
"I don\'t know".'''),

('human', 'Context: {context}'),

('human', 'Question: {question}'),

])

model_name = "llama3.2:3b"

# Creamos una instancia del modelo ChatOllama
# streaming=True permite que la respuesta se genere y se imprima en tiempo real.
model = ChatOllama(model=model_name, streaming=True)  # Habilita streaming


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

response = model.invoke(prompt)

print(response)