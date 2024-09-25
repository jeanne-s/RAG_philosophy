from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


def load_documents(data_path="data"):
    """Load documents from a directory containing PDFs."""
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()


def split_documents(documents: list[Document]):
    """Split the loaded documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, 
                                                   chunk_overlap=20,
                                                   length_function=len,
                                                   is_separator_regex=False)
    return text_splitter.split_documents(documents)


def embedding_function():
    """Initialize and return the Ollama embedding model."""
    embeddings = OllamaEmbeddings(model="qwen2.5:0.5b")
    return embeddings


def store_in_db(chunks: list[Document],
                persist_directory: str = "./chroma_db"):
    """Store document chunks in the Chroma database, skipping existing chunks."""

    database = Chroma(embedding_function=embedding_function(), 
                      persist_directory=persist_directory)
    
    chunks_with_ids = get_chunk_ids(chunks)

    existing_items = database.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the db
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        database.add_documents(new_chunks, ids=new_chunk_ids)
        database.persist()
    else:
        print("No new documents to add")
    
    return


def get_chunk_ids(chunks):
    """Assign unique IDs to document chunks based on their source, page, and chunk index."""

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks
    

if __name__ == "__main__":
    doc = load_documents()
    split = split_documents(doc)
    store_in_db(split)