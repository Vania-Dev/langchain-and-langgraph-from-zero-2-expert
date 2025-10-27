from langchain_ollama import ChatOllama
from pydantic import BaseModel

class AnswerWithJustification(BaseModel):
    '''An answer to the user's question along with justification for the
    answer.'''
    answer: str
    '''The answer to the user's question'''
    justification: str
    '''Justification for the answer'''

model_name = "llama3.2:3b"

# Creamos una instancia del modelo ChatOllama
# streaming=True permite que la respuesta se genere y se imprima en tiempo real.
model = ChatOllama(model=model_name, streaming=True)  # Habilita streaming

structured_llm = model.with_structured_output(AnswerWithJustification)

response = structured_llm.invoke("""What weighs more, a pound of bricks or a pound
of feathers""")
print(response)