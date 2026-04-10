from typing import List
import tiktoken
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def num_tokens_from_string(string: str, model: str = "qwen3:4b") -> int:
    """Cuenta el número de tokens en un string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(string))


def chunk_text(
        text: str,
        chunk_size: int,
        overlap: int
) -> List[str]:
    """
    Divide el texto en chunks.
    Primero separa el texto según la jerarquía de encabezados markdown y, posteriormente se
    hace un chunking recursivo por párrafos, oraciones ...

    Args:
        text: Texto a dividir
        chunk_size: Tamaño máximo de cada chunk en caracteres
        overlap: Número de caracteres de overlap entre chunks
        split_on_whitespace_only: Si es True, solo divide en espacios

    Returns:
        Lista de chunks de texto con los metadatos del animal y sección concreta a la que pertenecen
    """

    headers_to_split_on = [
        ("#", "Animal"),
        ("##", "Section"),
        ("###", "Subsection"),
        ("####", "Subsubsection")
    ]

    levels = [
        ("Animal", "Animal"),
        ("Section", "Section"),
        ("Subsection", "Subsection"),
        ("Subsubsection", "Subsubsection")
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", r"(?<=\.) ", " "], 
        is_separator_regex=True,
        length_function=len,
    )

    splits = text_splitter.split_documents(md_header_splits)

    chunks = []

    for doc in splits:
        header_lines = [
            f"{label}: {doc.metadata[key]}"
            for key, label in levels
            if key in doc.metadata and doc.metadata[key] is not None
        ]

        text = "\n".join(header_lines) + "\n" + doc.page_content

        new_doc = Document(
            page_content=text,
            metadata=doc.metadata
        )
        chunks.append(new_doc)

    return chunks
    
