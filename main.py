#%%
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.documents import Document
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s" 
)

US_TOPICS = [
    "Politics of the United States",
    "Federal government of the United States",
    "United States Congress",
    "Supreme Court of the United States",

    "President of the United States",
    "Vice President of the United States",
    "Joe Biden",
    "Kamala Harris",
    "Donald Trump",

    "2020 United States presidential election",
    "2024 United States presidential election",
    "United States presidential election",

    "Democratic Party (United States)",
    "Republican Party (United States)",

    # Policy domains
    "Immigration policy of the United States",
    "Healthcare reform in the United States",
    "Climate change policy of the United States",
    "United States foreign policy",

    # contemporary issues
    "Russian invasion of Ukraine",
    "Israel–Hamas war",
    "United States–China relations",
    "Epstein files",
    
    # Historical context
    "Barack Obama",
    "George W. Bush",
    "History of the United States Republican Party",
    "History of the Democratic Party (United States)",
    "List of presidents of the United States"
]

GLOBAL_TOPICS = [
    "Politics of the United Kingdom",
    "Politics of Germany",
    "Politics of France",
    "Politics of China",
    "Politics of India",

    "Rishi Sunak",
    "Olaf Scholz",
    "Emmanuel Macron",
    "Xi Jinping",
    "Narendra Modi",

    "NATO",
    "United Nations",
    "G7",
    "G20"
]


def load_wikipedia_topic_content(
    topics: List[str], 
    load_max_docs: int = 10, 
    doc_content_max_chars: int = 10000
) -> List[Document]:
    
    documents = []
    
    for topic in topics:
        logging.info('Loading topic: %s', topic)
        loader = WikipediaLoader(
            query = topic,
            load_max_docs = load_max_docs,
            doc_content_max_chars = doc_content_max_chars)
        
        docs = loader.load()
        logging.info('Loaded %d documents for topic: %s', len(docs), topic)
        
        documents.extend(docs)
        
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> List[Document]:
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    
    return text_splitter.split_documents(documents)
        

def main():
    logging.info("Start script")
    
    wikipedia_documents = load_wikipedia_topic_content(GLOBAL_TOPICS)
    logging.info("Succesfully retrieved %d documents", len(wikipedia_documents))
    
    chunked_documents = chunk_documents(wikipedia_documents)
    logging.info("Succesfully chunked into %d chunks", len(chunked_documents))
    
    embeddings = OllamaEmbeddings(model = "nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents = chunked_documents,
        embeddings = embeddings,
        persist_directory = "./wiki_db"
    )
    
if __name__ == "main":
    main()