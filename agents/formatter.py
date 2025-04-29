from typing import Dict, Any, List
from graph_definition import GraphState
import json

def format_output(state: GraphState) -> Dict[str, Any]:
    """
    Agent 8: Performs final quality checks and formats the output as JSON.

    Args:
        state: The current graph state containing final_chunks.

    Returns:
        A dictionary containing the final 'final_chunks' list, ready for saving.
        (Technically, LangGraph expects updates, so we return the state dict key).
    """
    print("Formatting final output...")
    final_chunks = state.get("final_chunks", [])

    if not final_chunks:
        print("Warning: No final chunks generated.")
        return {"final_chunks": []} # Return empty list

    # --- Quality Checks (Examples) ---
    checked_chunks = []
    for i, chunk in enumerate(final_chunks):
        # 1. Check for empty content
        if not chunk.get("content", "").strip():
            print(f"  Warning: Removing empty chunk at index {i}.")
            continue

        # 2. Check metadata integrity (basic)
        if not chunk.get("metadata"):
            print(f"  Warning: Chunk {i} missing metadata. Adding default.")
            chunk["metadata"] = {"warning": "Metadata was missing"}
        elif not chunk["metadata"].get("source"):
             # Ensure source file path is present
             chunk["metadata"]["source"] = state.get("pdf_path", "unknown_source")

        # 3. Ensure serializability (convert complex objects if any)
        try:
            # Attempt to serialize metadata to catch issues early
            json.dumps(chunk["metadata"])
        except TypeError as e:
            print(f"  Warning: Metadata in chunk {i} is not JSON serializable ({e}). Attempting conversion.")
            chunk["metadata"] = _make_serializable(chunk["metadata"])
            # Try serializing again after conversion
            try:
                json.dumps(chunk["metadata"])
            except TypeError as final_e:
                 print(f"  Error: Metadata still not serializable after conversion: {final_e}. Replacing metadata.")
                 chunk["metadata"] = {"error": "Metadata serialization failed", "original_keys": list(chunk["metadata"].keys())}

        checked_chunks.append(chunk)

    print(f"Final quality check complete. {len(checked_chunks)} chunks remaining.")

    # The state already holds 'final_chunks', just ensure it's the checked version.
    # LangGraph nodes return the *updates* to the state.
    return {"final_chunks": checked_chunks}


def _make_serializable(obj: Any) -> Any:
    """Recursively converts non-serializable items in dicts/lists to strings."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # Convert anything else to its string representation
        return str(obj)

