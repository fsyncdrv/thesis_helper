import os
import fitz
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
import chromadb
import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.spinner import Spinner
from rich.live import Live


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


llm = Ollama(model="llama3.2", request_timeout=300.0)
embed_model = OllamaEmbedding(model_name="qwen3-embedding:0.6b")


documents = SimpleDirectoryReader("data", required_exts=[".txt"]).load_data()

if os.path.exists('./storage') and os.listdir('./storage'):
    print('Loading existing index...')
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


console = Console()

BANNER = r"""

         ______________
        /             /|
       /             / |
      /____________ /  |
     | ___________ |   |   ________              _         __  __     __
     ||           ||   |  /_  __/ /_  ___  _____(_)____   / / / /__  / /___  ___  _____
     ||  Hello!   ||   |   / / / __ \/ _ \/ ___/ / ___/  / /_/ / _ \/ / __ \/ _ \/ ___/
     ||      + +  ||   |  / / / / / /  __(__  ) (__  )  / __  /  __/ / /_/ /  __/ /
     ||___________||   | /_/ /_/ /_/\___/____/_/____/  /_/ /_/\___/_/ .___/\___/_/
     |   _______   |  /                                          /_//_/
    /|  (_______)  | /
   ( |_____________|/
   .=======================.
   | ::::::::::::::::  ::: |
   | ::::::::::::::[]  ::: |
   |   -----------     ::: |
   `-----------------------'

"""

def print_banner():
    console.print(BANNER, style="bold cyan")
    console.print(Panel(
        "[bold white]Ask questions about your loaded research papers[/bold white]\n[dim]Type 'quit' to exit[/dim]",
        style="cyan",
        padding=(1, 4)
    ))

def print_answer(response):
    console.print(Panel(
        response.response,
        title="[bold green]💡 Answer[/bold green]",
        style="green",
        padding=(1, 2)
    ))

def print_sources(response):
    seen = set()
    for node in response.source_nodes:
        filename = node.metadata.get('file_name', 'Unknown')
        if filename not in seen:
            console.print(Panel(
                f"[dim]{node.text[:300]}...[/dim]",
                title=f"[bold yellow]📄 {filename}[/bold yellow]",
                style="yellow",
                padding=(1, 2)
            ))
            seen.add(filename)

def ask_question(question):
    with Live(Spinner("dots", text="[cyan]Thinking...[/cyan]"), refresh_per_second=10):
        response = query_engine.query(question)
    return response

print_banner()

while True:
    console.print("\n[bold cyan]❓ Question:[/bold cyan] ", end="")
    question = input().strip()

    if not question:
        continue

    if question.lower() == 'quit':
        console.print("\n[bold cyan]Goodbye! 👋[/bold cyan]\n")
        sys.exit(0)

    response = ask_question(question)
    print_answer(response)
    print_sources(response)
    console.print("[dim]" + "─" * 60 + "[/dim]")
