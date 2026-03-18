"""
PDF loading and text chunking utilities.

Loads all PDFs from the data_pdf directory, extracts text,
and splits into overlapping chunks with source metadata.
"""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from backend.config import DATA_PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def load_all_pdfs(pdf_dir: Path = DATA_PDF_DIR) -> List[Document]:
    """Load every PDF in *pdf_dir* and return a flat list of page-level Documents."""
    documents: List[Document] = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        return documents

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            # Attach a cleaner source name
            for page in pages:
                page.metadata["source"] = pdf_path.name
            documents.extend(pages)
            logger.info("Loaded %d pages from %s", len(pages), pdf_path.name)
        except Exception as exc:
            logger.error("Failed to load %s: %s", pdf_path.name, exc)

    logger.info("Total documents loaded: %d pages from %d PDFs", len(documents), len(pdf_files))
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d pages into %d chunks", len(documents), len(chunks))
    return chunks


def load_and_split_pdfs() -> List[Document]:
    """Convenience helper: load all PDFs then split into chunks."""
    docs = load_all_pdfs()
    return split_documents(docs)
