# Importamos ChatOllama para interactuar con modelos de Ollama
from langchain_ollama import ChatOllama

# Definimos el modelo que vamos a usar
model_name = "llama3.2:3b"

# Creamos una instancia del modelo ChatOllama
# streaming=True permite que las respuesta se genere y se imprima en tiempo real.
model = ChatOllama(model=model_name, streaming=True)  # Habilita streaming

# INVOKE: Método para enviar un único mensaje y obtener una respuesta completa
# Es el método más básico de la interfaz Runnable
completion = model.invoke('Hi there!')
print("Invoke", completion.content)

# BATCH: Método para procesar múltiples mensajes de una vez
# Útil para procesar lotes de datos de manera eficiente
# Devuelve una lista de respuestas en el mismo orden que los inputs
completions = model.batch(['Hi there!', 'Bye!'])
print("\nBatch", completions)

# STREAM: Método para obtener la respuesta token por token en tiempo real
# Permite mostrar la respuesta mientras se genera, mejorando la experiencia del usuario
# Cada token es un fragmento pequeño de la respuesta completa
print("\nStreaming tokens:")
for token in model.stream('Bye!'):
    print("\n", token)
    # Cada token contiene una parte de la respuesta:
    # Token 1: "Good"
    # Token 2: "bye" 
    # Token 3: "!"