import os
import fitz
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
import chromadb

## Extract text from PDF
for filename in os.listdir('data'):
    if filename.endswith('.pdf'):
        txt_filename = filename.replace('.pdf', '.txt')
        txt_path = f'data/{txt_filename}'

        if not os.path.exists(txt_path):
            print(f'Extracting text from {filename}...')
            doc = fitz.open(f'data/{filename}')
            with open(txt_path, 'w') as f:
                for page in doc:
                    f.write(page.get_text())
            print(f'Done! {len(doc)} pages extracted')
        else:
            print(f'Text already extracted for {filename}, skipping...')


## Set up models
llm = Ollama(model="llama3.2", request_timeout=300.0)
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
embed_model = OllamaEmbedding(model_name="qwen3-embedding:0.6b")


## Load and index
documents = SimpleDirectoryReader("data", required_exts=[".txt"]).load_data()

if os.path.exists('./storage') and os.listdir('./storage'):
    print('Loading existing index...')
    ## Set up ChromaDB
    chroma_client = chromadb.PersistentClient(path="./storage")
    chroma_collection = chroma_client.get_or_create_collection("papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )
else:
    print('Building new index...')
    ## Set up ChromaDB
    chroma_client = chromadb.PersistentClient(path="./storage")
    chroma_collection = chroma_client.get_or_create_collection("papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    documents = SimpleDirectoryReader('data', required_exts=['.txt']).load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=50)],
        show_progress=True
    )


## Query
qa_prompt = PromptTemplate(
    "You are a research assistant. Only use information from the provided context to answer. "
    "Do not make up facts, statistics, or numbers not present in the context. "
    "Do not reference file names or file paths in your answer. "
    "You may synthesise and summarise information from the context to answer the question. "
    "If the answer is truly not in the context, say 'I cannot find this in the document'.\n\n"
    "Context: {context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer: "
)


query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=5,
    text_qa_template=qa_prompt
)


question = "What is the abstract of the PanTS paper?"
response = query_engine.query(question)
print('Question: ')
print(question)
print('Response: ')
print(response)
print('Sources:')
seen = set()
for node in response.source_nodes:
    filename = node.metadata.get('file_name', 'Unknown')
    if filename not in seen:
        print(f' - {filename}')
        seen.add(filename)
