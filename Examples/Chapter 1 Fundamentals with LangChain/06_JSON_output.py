# Importamos ChatOllama para interactuar con modelos de Ollama
from langchain_ollama import ChatOllama
# Importamos BaseModel de Pydantic para definir esquemas de datos estructurados
from pydantic import BaseModel

# Definimos un modelo Pydantic que especifica la estructura de la respuesta
# Esto garantiza que el modelo devuelva datos en un formato consistente y tipado
class AnswerWithJustification(BaseModel):
    '''An answer to the user's question along with justification for the
    answer.'''
    answer: str
    '''The answer to the user's question'''
    justification: str
    '''Justification for the answer'''

# Definimos el modelo que vamos a usar
model_name = "llama3.2:3b"

# Creamos una instancia del modelo ChatOllama
# streaming=True permite que la respuesta se genere y se imprima en tiempo real.
model = ChatOllama(model=model_name, streaming=True)  # Habilita streaming

# Configuramos el modelo para que devuelva respuestas estructuradas
# with_structured_output() fuerza al modelo a seguir el esquema definido en AnswerWithJustification
# Esto es útil para obtener respuestas consistentes y procesables programáticamente
structured_llm = model.with_structured_output(AnswerWithJustification)

# Hacemos una pregunta al modelo estructurado
# La respuesta será un objeto AnswerWithJustification con campos 'answer' y 'justification'
response = structured_llm.invoke("""What weighs more, a pound of bricks or a pound
of feathers""")

# Imprimimos la respuesta estructurada
# Podemos acceder a response.answer y response.justification individualmente
print(response)