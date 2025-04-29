from typing import Dict, Any, List
from graph_definition import GraphState
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter # Example splitters

# --- Configuration ---
CHUNK_STRATEGY = "recursive" # Options: "recursive", "markdown", "semantic" (requires embedding model)
CHUNK_SIZE = 1000 # Target size for chunks (in characters for recursive/markdown)
CHUNK_OVERLAP = 150 # Overlap between chunks
# For semantic chunking (if implemented):
# EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Example sentence-transformer model
# SEMANTIC_THRESHOLD = 0.85 # Example percentile threshold

# Initialize text splitters
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n", "\n", ". ", ", ", " ", ""] # Common separators
)

markdown_splitter = MarkdownTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Placeholder for semantic chunking setup
# if CHUNK_STRATEGY == "semantic":
#     try:
#         from langchain_experimental.text_splitter import SemanticChunker
#         from langchain_community.embeddings import OllamaEmbeddings # Or SentenceTransformerEmbeddings
#
#         embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=os.getenv("OLLAMA_BASE_URL"))
#         # embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
#
#         semantic_splitter = SemanticChunker(
#             embeddings=embeddings,
#             breakpoint_threshold_type="percentile", # Or "standard_deviation", "interquartile"
#             breakpoint_threshold_amount=SEMANTIC_THRESHOLD
#         )
#         print("Initialized Semantic Chunker.")
#     except ImportError:
#         print("Semantic Chunker or embedding model not available. Falling back to recursive.")
#         CHUNK_STRATEGY = "recursive"
#     except Exception as e:
#         print(f"Error initializing Semantic Chunker: {e}. Falling back to recursive.")
#         CHUNK_STRATEGY = "recursive"


def create_chunks(state: GraphState) -> Dict[str, Any]:
    """
    Agent 7: Chunks the synthesized content into meaningful segments.

    Args:
        state: The current graph state with synthesized_content.

    Returns:
        A dictionary with the updated 'final_chunks'.
    """
    print(f"Chunking synthesized content using strategy: {CHUNK_STRATEGY}...")
    synthesized_content = state.get("synthesized_content", [])
    final_chunks = []
    doc_metadata = state.get("metadata", {}) # Get overall doc metadata

    if not synthesized_content:
        print("No synthesized content to chunk.")
        return {}

    # Combine content into a single string or process element by element?
    # Option 1: Combine all text-based content into one large document string
    # Option 2: Chunk element by element, preserving element type in metadata

    # Let's try Option 2: Chunk element by element, adding type info
    chunk_index = 0
    for element in synthesized_content:
        element_type = element.get("type", "unknown")
        content_to_chunk = ""
        base_metadata = {
            **doc_metadata, # Add doc-level metadata (like source file)
            **element.get("metadata", {}), # Add element specific metadata (page, bbox etc)
            "element_type": element_type
        }

        # Extract content based on type
        if element_type == "text":
            content_to_chunk = element.get("text", "")
        elif element_type == "image_summary":
            # Combine description and OCR text if available
            desc = element.get("description", "")
            ocr = element.get("ocr_text")
            content_to_chunk = f"Image Description: {desc}"
            if ocr:
                content_to_chunk += f"\n\nOCR Text:\n{ocr}"
            base_metadata["image_ref"] = element.get("image_ref")
        elif element_type == "chart_summary":
            content_to_chunk = f"Chart Summary: {element.get('summary', '')}"
            base_metadata["chart_ref"] = element.get("chart_ref")
        elif element_type == "table_processed":
            table_fmt = element.get("format", "unknown")
            table_data = element.get("data", "")
            content_to_chunk = f"Table (Format: {table_fmt}):\n{table_data}"
            base_metadata["table_ref"] = element.get("table_ref")
            base_metadata["table_format"] = table_fmt
        else:
            # Skip unknown types or try to convert to string
            content_to_chunk = str(element.get("content", "")) # Fallback

        if not content_to_chunk.strip():
            continue # Skip empty content

        # Apply selected chunking strategy
        try:
            chunks = []
            if CHUNK_STRATEGY == "markdown":
                # Use Markdown splitter if content is likely Markdown (tables, maybe LLM-cleaned text)
                if element_type == "table_processed" and base_metadata.get("table_format") == "llm_markdown":
                     chunks = markdown_splitter.split_text(content_to_chunk)
                else:
                     # Use recursive for general text or non-markdown tables/summaries
                     chunks = recursive_splitter.split_text(content_to_chunk)
            # elif CHUNK_STRATEGY == "semantic" and semantic_splitter:
            #     chunks = semantic_splitter.split_text(content_to_chunk)
            else: # Default to recursive
                chunks = recursive_splitter.split_text(content_to_chunk)

            # Create final chunk dictionaries
            for i, chunk_text in enumerate(chunks):
                chunk_metadata = {
                    **base_metadata,
                    "chunk_index": chunk_index,
                    "part_of_element": i + 1, # Which part of the original element this chunk is
                    "total_parts": len(chunks) # Total parts the element was split into
                }
                final_chunks.append({
                    "content": chunk_text,
                    "metadata": chunk_metadata
                })
                chunk_index += 1

        except Exception as e:
            print(f"Error chunking element ({element_type}): {e}")
            # Add the whole element as a single chunk with error info?
            error_metadata = {**base_metadata, "chunking_error": str(e), "chunk_index": chunk_index}
            final_chunks.append({"content": content_to_chunk, "metadata": error_metadata})
            chunk_index += 1


    print(f"Finished chunking. Generated {len(final_chunks)} final chunks.")
    return {"final_chunks": final_chunks}

