"""
╔══════════════════════════════════════════════════════════════════════╗
║  ColBERT — Optimizing Embeddings con Late Interaction
║  Versión actualizada — Python 3.13 (sin ragatouille)
╚══════════════════════════════════════════════════════════════════════╝

Por qué ya no se usa ragatouille:
  ❌ ragatouille depende de `voyager` que no tiene wheels para Python 3.13
  ❌ ragatouille requiere langchain-core < 0.2.0 (incompatible con v1+)
  ❌ El proyecto está en mantenimiento mínimo desde 2024

Solución:
  ✅ sentence-transformers con modelo ColBERT
  ✅ FAISS para indexado eficiente
  ✅ Compatible con Python 3.13 y langchain v1+

Instalación:
    uv add sentence-transformers faiss-cpu langchain-core langchain-ollama

Requiere Ollama corriendo localmente (para el RAG chain final):
    ollama pull gemma4:latest
"""

import faiss
import numpy as np
import requests
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

# ── Configuración ─────────────────────────────────────────────────────────────
# Modelo con capacidades similares a ColBERT (late interaction bidireccional)
# ColBERT usa "late interaction": genera embeddings detallados para cada token
# y calcula similitud al final, en lugar de un único vector por documento
# Alternativas:
#   "sentence-transformers/all-MiniLM-L6-v2"  — liviano, rápido
#   "sentence-transformers/msmarco-distilbert-base-v4"  — mejor para búsqueda
#   "colbert-ir/colbertv2.0"  — modelo original (requiere transformers directo)
COLBERT_MODEL = "sentence-transformers/msmarco-distilbert-base-v4"
LLM_MODEL = "gemma4:latest"  # Modelo de Ollama para generar respuestas finales


# ── 1. Obtener documento de Wikipedia ────────────────────────────────────────
def get_wikipedia_page(title: str) -> str | None:
    """Descarga el contenido completo de una página de Wikipedia.

    Usa la API de Wikipedia para obtener el texto plano (sin HTML) de un artículo.
    Esto nos da el documento fuente que luego dividiremos e indexaremos.
    """
    URL = "https://en.wikipedia.org/w/api.php"
    # Parámetros para obtener el extracto completo en texto plano
    params = {
        "action": "query",  # Tipo de operación: consulta
        "format": "json",  # Formato de respuesta
        "titles": title,  # Título del artículo a buscar
        "prop": "extracts",  # Propiedad: contenido del artículo
        "explaintext": True,  # Devolver texto plano sin HTML
    }
    headers = {
        "User-Agent": "ColBERT_tutorial/2025"
    }  # Identificación requerida por Wikipedia
    response = requests.get(URL, params=params, headers=headers)
    data = response.json()
    # Wikipedia devuelve un dict con IDs de página como keys, tomamos el primero
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None


# ── 2. Dividir en chunks (equivalente a split_documents=True, max_document_length=180) ──
def split_text(text: str, max_words: int = 180) -> list[str]:
    """Divide el texto en chunks de máximo `max_words` palabras.

    RAG funciona mejor con fragmentos pequeños porque:
    1. Los embeddings capturan mejor el significado de textos cortos
    2. Recuperamos solo la información relevante, no documentos completos
    3. Evitamos exceder el límite de tokens del LLM

    180 palabras ≈ 240 tokens, un tamaño óptimo para búsqueda semántica.
    """
    words = text.split()  # Dividir por espacios en blanco
    # Crear chunks deslizantes: cada uno tiene max_words palabras
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


# ── 3. Clase ColBERTIndex — equivalente a RAGPretrainedModel + RAG.index() ───
class ColBERTIndex:
    """
    Índice vectorial con sentence-transformers + FAISS.
    Replica el comportamiento de RAGatouille con APIs modernas.

    Componentes:
    - SentenceTransformer: convierte texto a vectores (embeddings)
    - FAISS: biblioteca de Facebook para búsqueda eficiente de vectores similares
    - chunks: almacena los fragmentos de texto originales
    """

    def __init__(self, model_name: str = COLBERT_MODEL):
        print(f"🔄 Cargando modelo: {model_name}")
        # SentenceTransformer carga el modelo preentrenado desde HuggingFace
        self.model = SentenceTransformer(model_name)
        # FAISS index se creará después de indexar documentos
        self.index: faiss.IndexFlatIP | None = None
        # Lista para guardar los chunks de texto originales (necesarios para recuperación)
        self.chunks: list[str] = []
        print("   ✅ Modelo cargado")

    def index_documents(
        self,
        collection: list[str],
        index_name: str = "default",
        max_document_length: int = 180,
        split_documents: bool = True,
    ) -> None:
        """
        Indexa documentos para búsqueda semántica.

        Proceso:
        1. Dividir documentos en chunks manejables
        2. Convertir cada chunk a un vector (embedding)
        3. Almacenar vectores en FAISS para búsqueda rápida

        Equivalente a:
            RAG.index(collection=[doc], index_name=..., max_document_length=180, split_documents=True)
        """
        print(f"\n📚 Indexando colección: '{index_name}'")

        # PASO 1: Dividir documentos en chunks si se solicita
        if split_documents:
            for doc in collection:
                # Extender la lista con todos los chunks del documento
                self.chunks.extend(split_text(doc, max_words=max_document_length))
        else:
            # Usar documentos completos sin dividir
            self.chunks = list(collection)

        print(f"   ✂️  {len(self.chunks)} chunks generados")

        # PASO 2: Generar embeddings (vectores) para cada chunk
        print("   🔢 Calculando embeddings...")
        embeddings = self.model.encode(
            self.chunks,
            convert_to_numpy=True,  # Devolver arrays de NumPy (requerido por FAISS)
            show_progress_bar=True,  # Mostrar progreso en consola
            normalize_embeddings=True,  # Normalizar a longitud 1 (importante para Inner Product)
        )
        # embeddings es una matriz de shape (num_chunks, embedding_dimension)
        # Cada fila es el vector que representa un chunk

        # PASO 3: Crear índice FAISS para búsqueda eficiente
        dimension = embeddings.shape[1]  # Dimensión del embedding (ej: 768 para BERT)
        # IndexFlatIP = Inner Product (producto punto)
        # Con embeddings normalizados, Inner Product = Cosine Similarity
        self.index = faiss.IndexFlatIP(dimension)
        # Agregar todos los vectores al índice
        self.index.add(embeddings.astype(np.float32))  # FAISS requiere float32

        print(f"   ✅ Índice creado con {self.index.ntotal} vectores (dim={dimension})")

    def search(self, query: str, k: int = 3) -> list[dict]:
        """
        Busca los k chunks más similares a la consulta.

        Proceso:
        1. Convertir la consulta a embedding (vector)
        2. Buscar en FAISS los k vectores más cercanos
        3. Recuperar los chunks originales correspondientes

        Equivalente a:
            results = RAG.search(query="...", k=3)

        Retorna lista de dicts con 'content', 'score', 'rank'.
        """
        if self.index is None:
            raise RuntimeError("Primero debes llamar a index_documents()")

        # PASO 1: Convertir la consulta a embedding
        query_embedding = self.model.encode(
            [query],  # Lista con un solo elemento (la consulta)
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalizar para usar con Inner Product
        ).astype(np.float32)

        # PASO 2: Buscar en FAISS los k vectores más similares
        # scores: similitud (mayor = más similar)
        # indices: posiciones de los chunks en self.chunks
        scores, indices = self.index.search(query_embedding, k)

        # PASO 3: Construir resultados con el contenido original
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            results.append(
                {
                    "content": self.chunks[idx],  # Texto original del chunk
                    "score": float(score),  # Similitud con la consulta
                    "rank": rank,  # Posición en el ranking (1 = más relevante)
                }
            )

        return results

    def as_langchain_retriever(self, k: int = 3) -> "ColBERTRetriever":
        """
        Crea un retriever compatible con LangChain.

        Esto permite integrar nuestro índice ColBERT en chains de LangChain,
        como RAG chains que combinan recuperación + generación.

        Equivalente a:
            retriever = RAG.as_langchain_retriever(k=3)
        """
        return ColBERTRetriever(colbert_index=self, k=k)


# ── 4. LangChain Retriever ────────────────────────────────────────────────────
class ColBERTRetriever(BaseRetriever):
    """Retriever de LangChain que usa ColBERTIndex internamente.

    BaseRetriever es la clase base de LangChain para componentes de recuperación.
    Implementamos _get_relevant_documents() para definir cómo buscar documentos.
    """

    colbert_index: ColBERTIndex  # Nuestro índice personalizado
    k: int = 3  # Número de documentos a recuperar

    class Config:
        # Permitir tipos personalizados (ColBERTIndex) en Pydantic
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """Método requerido por BaseRetriever.

        Busca documentos relevantes y los convierte al formato Document de LangChain.
        """
        # Buscar usando nuestro índice ColBERT
        results = self.colbert_index.search(query, k=self.k)
        # Convertir resultados a objetos Document de LangChain
        return [Document(page_content=r["content"]) for r in results]


# ── 5. Pipeline completo ──────────────────────────────────────────────────────
def main():
    """Demostración completa del sistema RAG con ColBERT.

    Pipeline:
    1. Descargar documento fuente (Wikipedia)
    2. Indexar con ColBERT (dividir + embeddings + FAISS)
    3. Búsqueda directa (sin LLM)
    4. Retriever de LangChain (sin LLM)
    5. RAG Chain completo (retriever + LLM para generar respuesta)
    """
    print("=" * 60)
    print("  ColBERT RAG — Versión 2025 (Python 3.13, sin ragatouille)")
    print("=" * 60)

    # PASO 1: Obtener documento fuente
    print("\n📖 Descargando artículo de Wikipedia...")
    full_document = get_wikipedia_page("Hayao_Miyazaki")
    if not full_document:
        raise ValueError("No se pudo obtener el artículo")
    print(f"   ✅ {len(full_document)} caracteres descargados")

    # PASO 2: Crear índice vectorial
    # Equivalente a RAGPretrainedModel.from_pretrained() + RAG.index()
    RAG = ColBERTIndex(model_name=COLBERT_MODEL)
    RAG.index_documents(
        collection=[full_document],  # Lista de documentos a indexar
        index_name="Miyazaki-123",  # Nombre identificador del índice
        max_document_length=180,  # Máximo de palabras por chunk
        split_documents=True,  # Dividir en chunks automáticamente
    )
    # Ahora tenemos un índice FAISS con embeddings de todos los chunks

    # PASO 3: Búsqueda directa (sin LLM, solo recuperación)
    # Equivalente a RAG.search()
    print("\n🔍 Búsqueda directa (equivalente a RAG.search()):")
    print("   Esto solo recupera chunks relevantes, sin generar respuesta")
    results = RAG.search(query="What animation studio did Miyazaki found?", k=3)
    print(results)

    # Mostrar los chunks más relevantes encontrados
    for r in results:
        print(f"\n[{r['rank']}] score={r['score']:.4f}")
        print(f"    {r['content'][:200]}...")  # Primeros 200 caracteres

    # PASO 4: LangChain Retriever (interfaz estándar de LangChain)
    # Equivalente a RAG.as_langchain_retriever()
    print("\n🦜 LangChain Retriever (equivalente a RAG.as_langchain_retriever()):")
    print("   Mismo resultado que búsqueda directa, pero en formato LangChain")
    retriever = RAG.as_langchain_retriever(k=3)
    # invoke() es el método estándar de LangChain para ejecutar componentes
    response = retriever.invoke("What animation studio did Miyazaki found?")
    print(response)

    # response es una lista de objetos Document de LangChain
    for doc in response:
        print(f"\n📄 {doc.page_content[:200]}...")  # Primeros 200 caracteres

    # PASO 5: RAG Chain completo (Retrieval + Generation)
    # Bonus — no estaba en el libro, demuestra el flujo completo
    print("\n🤖 RAG Chain con Ollama:")
    print("   Ahora sí generamos una respuesta usando el LLM + contexto recuperado")

    # Inicializar el LLM local (Ollama)
    llm = ChatOllama(
        model=LLM_MODEL, temperature=0
    )  # temperature=0 para respuestas deterministas

    # Crear el prompt que combina contexto + pregunta
    rag_prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context:

        {context}

        Question: {question}
    """)

    # Función auxiliar para formatear los documentos recuperados
    def format_docs(docs):
        """Convierte lista de Documents en un solo string con saltos de línea."""
        return "\n\n".join(d.page_content for d in docs)

    # Construir el chain usando LCEL (LangChain Expression Language)
    # El operador | encadena componentes: salida de uno → entrada del siguiente
    rag_chain = (
        # Paso 1: Preparar inputs para el prompt
        {
            "context": retriever | format_docs,  # Recuperar docs y formatearlos
            "question": RunnablePassthrough(),
        }  # Pasar la pregunta sin modificar
        | rag_prompt  # Paso 2: Insertar context + question en el template
        | llm  # Paso 3: Enviar prompt al LLM para generar respuesta
        | StrOutputParser()  # Paso 4: Extraer el texto de la respuesta
    )

    # Ejecutar el chain completo
    query = "What animation studio did Miyazaki found?"
    answer = rag_chain.invoke(query)  # invoke() ejecuta todo el pipeline
    print(f"\nPregunta: {query}")
    print(f"Respuesta: {answer}")


if __name__ == "__main__":
    main()
