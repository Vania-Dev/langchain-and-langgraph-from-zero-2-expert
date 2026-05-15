"""
╔══════════════════════════════════════════════════════════════════════╗
║  RAPTOR - Recursive Abstractive Processing for Tree-Organized Retrieval
║  Versión actualizada 2025 — Python 3.13 + LangChain v1 + Ollama
╚══════════════════════════════════════════════════════════════════════╝

Instalación:
    uv add langchain langchain-ollama langchain-community
    uv add faiss-cpu sentence-transformers
    uv add umap-learn scikit-learn numpy

Requiere Ollama corriendo localmente:
    ollama pull gemma4:latest
    ollama pull nomic-embed-text
"""

from typing import Optional

import numpy as np
import requests
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configuración ─────────────────────────────────────────────────────────────
# Definimos los modelos que usaremos de Ollama (deben estar instalados localmente)
LLM_MODEL = "gemma4:latest"  # Modelo de lenguaje para generar resúmenes y respuestas
EMBED_MODEL = "nomic-embed-text"  # Modelo para convertir texto en vectores numéricos
MAX_LEVELS = 3  # Profundidad máxima del árbol: cuántas veces resumiremos los resúmenes

# ── 1. LLM y Embeddings (Ollama local, sin API key) ──────────────────────────
# Inicializamos el modelo de lenguaje con temperatura 0 (respuestas determinísticas)
llm = ChatOllama(model=LLM_MODEL, temperature=0)
# Inicializamos el modelo de embeddings para crear representaciones vectoriales del texto
embeddings = OllamaEmbeddings(model=EMBED_MODEL)


# ── 2. Obtener documento de Wikipedia ────────────────────────────────────────
def get_wikipedia_page(title: str) -> Optional[str]:
    """Descarga el contenido de texto de un artículo de Wikipedia."""
    # URL de la API de Wikipedia
    URL = "https://en.wikipedia.org/w/api.php"
    # Parámetros para solicitar el texto plano del artículo
    params = {
        "action": "query",        # Acción: consultar
        "format": "json",         # Formato de respuesta: JSON
        "titles": title,          # Título del artículo a buscar
        "prop": "extracts",       # Propiedad: extraer contenido
        "explaintext": True,      # Obtener texto plano (sin HTML)
    }
    headers = {"User-Agent": "RAPTOR_tutorial/2025"}  # Identificarnos ante Wikipedia
    # Hacemos la petición HTTP
    response = requests.get(URL, params=params, headers=headers)
    data = response.json()
    # Extraemos la primera página del resultado
    page = next(iter(data["query"]["pages"].values()))
    # Retornamos el texto si existe, sino None
    return page["extract"] if "extract" in page else None


# ── 3. Dividir texto en chunks (nodos hoja — Nivel 0) ────────────────────────
def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Divide el texto largo en fragmentos más pequeños (chunks) con solapamiento."""
    # RecursiveCharacterTextSplitter divide el texto respetando párrafos y oraciones
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,    # Tamaño máximo de cada fragmento en caracteres
        chunk_overlap=overlap,    # Caracteres que se solapan entre chunks consecutivos
    )
    # Retorna una lista de strings (cada uno es un chunk)
    return splitter.split_text(text)


# ── 4. Clustering con UMAP + Gaussian Mixture Model ──────────────────────────
def cluster_texts(
    texts: list[str], n_clusters: Optional[int] = None
) -> dict[int, list[str]]:
    """
    Agrupa textos por similitud semántica usando UMAP + GMM.
    Retorna un dict {cluster_id: [textos]}.
    """
    import umap
    from sklearn.mixture import GaussianMixture

    # PASO 1: Convertir cada texto en un vector numérico (embedding)
    # Esto permite comparar textos matemáticamente
    vecs = np.array(embeddings.embed_documents(texts))

    # PASO 2: Reducir dimensionalidad con UMAP (de ~768 dimensiones a 2)
    # Esto facilita el clustering y visualización
    n_neighbors = max(2, min(15, len(texts) - 1))  # Vecinos para UMAP
    reducer = umap.UMAP(
        n_components=2,           # Reducir a 2 dimensiones
        n_neighbors=n_neighbors,  # Cuántos vecinos considerar
        metric="cosine",          # Métrica de similitud
        random_state=42           # Semilla para reproducibilidad
    )
    reduced = reducer.fit_transform(vecs)  # Aplicar la reducción

    # PASO 3: Determinar número óptimo de clusters automáticamente con BIC
    # (Bayesian Information Criterion - menor es mejor)
    if n_clusters is None:
        max_k = min(10, len(texts) // 2)  # Máximo de clusters a probar
        best_bic = np.inf  # Inicializar con infinito
        best_k = 2         # Mínimo 2 clusters
        # Probar diferentes números de clusters
        for k in range(2, max_k + 1):
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(reduced)
            bic = gmm.bic(reduced)  # Calcular BIC
            if bic < best_bic:      # Si es mejor, guardarlo
                best_bic = bic
                best_k = k
        n_clusters = best_k  # Usar el mejor número encontrado

    # PASO 4: Aplicar Gaussian Mixture Model para agrupar
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = gmm.fit_predict(reduced)  # Asignar cada texto a un cluster

    # PASO 5: Organizar los textos por cluster en un diccionario
    clusters: dict[int, list[str]] = {}
    for text, label in zip(texts, labels):
        clusters.setdefault(int(label), []).append(text)

    return clusters  # {0: [texto1, texto2], 1: [texto3], ...}


# ── 5. Resumir cada cluster con el LLM ───────────────────────────────────────
# Creamos una plantilla de prompt para pedirle al LLM que resuma
summary_prompt = ChatPromptTemplate.from_template(
    "You are an expert summarizer. Write a concise summary of the following text:\n\n{text}"
)
# Creamos una cadena (chain) que conecta: prompt → LLM → parser de texto
# El operador | (pipe) conecta los componentes secuencialmente
summarize_chain = summary_prompt | llm | StrOutputParser()


def summarize_cluster(texts: list[str]) -> str:
    """Combina múltiples textos de un cluster y genera un resumen único."""
    # Unir todos los textos del cluster con doble salto de línea
    combined = "\n\n".join(texts)
    # Truncar si es muy largo para el contexto del modelo (evitar errores)
    if len(combined) > 6000:
        combined = combined[:6000] + "..."
    # Invocar la cadena de resumen y retornar el resultado
    return summarize_chain.invoke({"text": combined})


# ── 6. RAPTOR: construcción recursiva del árbol ───────────────────────────────
def build_raptor_tree(texts: list[str], level: int = 0) -> list[str]:
    """
    Construye el árbol RAPTOR recursivamente.
    RAPTOR = Recursive Abstractive Processing for Tree-Organized Retrieval
    
    Proceso:
    1. Nivel 0: Chunks originales (hojas del árbol)
    2. Nivel 1: Agrupar chunks similares y resumir cada grupo
    3. Nivel 2: Agrupar resúmenes y resumir nuevamente
    4. Nivel N: Continuar hasta MAX_LEVELS o quedar con pocos textos
    
    Retorna TODOS los nodos del árbol (hojas + resúmenes de todos los niveles)
    """
    print(f"\n🌲 Nivel {level}: {len(texts)} nodos")

    # CONDICIÓN DE PARADA: si quedan pocos textos o llegamos al nivel máximo
    if len(texts) <= 2 or level >= MAX_LEVELS:
        print(f"   ✅ Fin de recursión")
        return texts  # Retornar los textos actuales sin procesar más

    # PASO 1: Agrupar textos similares en clusters
    clusters = cluster_texts(texts)
    print(f"   📦 {len(clusters)} clusters encontrados")

    # PASO 2: Resumir cada cluster
    summaries = []
    for cluster_id, cluster_docs in clusters.items():
        print(f"   ✍️  Resumiendo cluster {cluster_id} ({len(cluster_docs)} textos)...")
        # Generar un resumen que representa todo el cluster
        summary = summarize_cluster(cluster_docs)
        summaries.append(summary)

    # PASO 3: RECURSIÓN - Procesar los resúmenes como si fueran nuevos textos
    # Esto crea el siguiente nivel del árbol
    higher_summaries = build_raptor_tree(summaries, level + 1)

    # PASO 4: Retornar TODOS los nodos: originales + resúmenes de este nivel + niveles superiores
    # Esto permite buscar en cualquier nivel de abstracción
    return texts + higher_summaries


# ── 7. Indexar todos los nodos en FAISS ──────────────────────────────────────
def build_index(all_texts: list[str]) -> FAISS:
    """Crea un índice vectorial FAISS con todos los nodos del árbol RAPTOR."""
    print(f"\n📚 Indexando {len(all_texts)} nodos en FAISS...")
    # Convertir cada texto en un Document de LangChain
    docs = [Document(page_content=t) for t in all_texts]
    # Crear el vectorstore: convierte cada documento en embedding y los indexa
    # FAISS permite búsquedas rápidas por similitud de vectores
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("   ✅ Índice FAISS creado")
    return vectorstore


# ── 8. Retriever de LangChain ─────────────────────────────────────────────────
class RAPTORRetriever(BaseRetriever):
    """Retriever personalizado que busca en el índice FAISS del árbol RAPTOR."""
    vectorstore: FAISS  # El índice vectorial donde buscar
    k: int = 3          # Número de documentos a recuperar

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """Busca los k documentos más similares a la consulta."""
        # similarity_search convierte la query en embedding y busca los más cercanos
        return self.vectorstore.similarity_search(query, k=self.k)


# ── 9. Pipeline completo ──────────────────────────────────────────────────────
def main():
    """Función principal que ejecuta todo el pipeline RAPTOR."""
    print("=" * 60)
    print("  RAPTOR RAG — Versión 2025 (Python 3.13 + Ollama)")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 1: OBTENER DOCUMENTO
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n📖 Descargando artículo de Wikipedia...")
    document = get_wikipedia_page("Hayao_Miyazaki")
    if not document:
        raise ValueError("No se pudo obtener el artículo de Wikipedia")
    print(f"   ✅ {len(document)} caracteres obtenidos")

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 2: DIVIDIR EN CHUNKS (Nivel 0 del árbol - hojas)
    # ═══════════════════════════════════════════════════════════════════════════
    chunks = split_into_chunks(document, chunk_size=500)
    print(f"\n✂️  {len(chunks)} chunks creados (nodos hoja)")

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 3: CONSTRUIR ÁRBOL RAPTOR (clustering + resúmenes recursivos)
    # ═══════════════════════════════════════════════════════════════════════════
    all_nodes = build_raptor_tree(chunks)
    print(f"\n🌳 Árbol completo: {len(all_nodes)} nodos totales")
    print(f"   ({len(chunks)} hojas + {len(all_nodes) - len(chunks)} resúmenes)")

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 4: INDEXAR TODOS LOS NODOS EN FAISS
    # ═══════════════════════════════════════════════════════════════════════════
    vectorstore = build_index(all_nodes)

    # ═══════════════════════════════════════════════════════════════════════════
    # DEMOSTRACIÓN 1: Búsqueda directa en el vectorstore
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n🔍 Búsqueda directa:")
    query = "What animation studio did Miyazaki found?"
    results = vectorstore.similarity_search(query, k=3)
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc.page_content[:200]}...")

    # ═══════════════════════════════════════════════════════════════════════════
    # DEMOSTRACIÓN 2: Usando LangChain Retriever (interfaz estándar)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n🦜 LangChain Retriever:")
    retriever = RAPTORRetriever(vectorstore=vectorstore, k=3)
    docs = retriever.invoke(query)
    for i, doc in enumerate(docs, 1):
        print(f"\n[{i}] {doc.page_content[:200]}...")

    # ═══════════════════════════════════════════════════════════════════════════
    # DEMOSTRACIÓN 3: RAG Chain completo (Retrieval + Generation)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n🤖 RAG Chain completo:")
    
    # Plantilla de prompt para el RAG: contexto + pregunta
    rag_prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context:

        {context}

        Question: {question}
    """)

    # Función auxiliar para formatear los documentos recuperados
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    from langchain_core.runnables import RunnablePassthrough

    # Construir la cadena RAG completa:
    # 1. Recuperar documentos relevantes y formatearlos como contexto
    # 2. Pasar la pregunta tal cual (RunnablePassthrough)
    # 3. Insertar contexto y pregunta en el prompt
    # 4. Enviar al LLM
    # 5. Parsear la respuesta como string
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # Ejecutar la cadena completa
    answer = rag_chain.invoke(query)
    print(f"\nPregunta: {query}")
    print(f"Respuesta: {answer}")


if __name__ == "__main__":
    main()
