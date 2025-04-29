from typing import Dict, Any, List
from graph_definition import GraphState

def synthesize_content(state: GraphState) -> Dict[str, Any]:
    """
    Agent 6: Synthesizes processed content into a logical order.
    Combines text chunks, image descriptions, chart summaries, and table data
    based on their original page and approximate location (if available).

    Args:
        state: The current graph state containing processed elements.

    Returns:
        A dictionary with the updated 'synthesized_content'.
    """
    print("Synthesizing processed content...")

    processed_text = state.get("processed_text_chunks", [])
    image_desc = state.get("image_descriptions", [])
    chart_sum = state.get("chart_summaries", [])
    table_data = state.get("table_data", [])

    # Combine all processed elements into one list
    all_elements = []
    all_elements.extend([{"type": "text", **item} for item in processed_text])
    all_elements.extend([{"type": "image_summary", **item} for item in image_desc])
    # Note: Chart summaries might overlap with image descriptions if charts were treated as images.
    # We might need a way to deduplicate or prioritize. For now, include both.
    all_elements.extend([{"type": "chart_summary", **item} for item in chart_sum])
    all_elements.extend([{"type": "table_processed", **item} for item in table_data])

    # Sort elements primarily by page number, then potentially by bounding box y-coordinate
    # This is a simple sorting approach. More sophisticated layout analysis (Agent 0)
    # would provide a more reliable reading order.
    def get_sort_key(element):
        metadata = element.get("metadata", {})
        page_num = metadata.get("page_number", float('inf')) # Put elements without page num last
        # Use top of bounding box (y0) for vertical sorting within a page
        bbox = metadata.get("bbox")
        y_coord = float('inf') # Default if no bbox
        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
            y_coord = bbox[1] # y0 coordinate
        return (page_num, y_coord)

    try:
        sorted_elements = sorted(all_elements, key=get_sort_key)
        print(f"Synthesized and sorted {len(sorted_elements)} processed elements.")
    except Exception as e:
        print(f"Error sorting elements: {e}. Using original order.")
        # Fallback to the order they were added (less accurate)
        sorted_elements = all_elements


    # Optional: Add relationship extraction logic here if needed
    # e.g., scan text for references like "See Figure 1" or "Table 2 shows..."
    # and link them to the corresponding image/table elements.

    return {"synthesized_content": sorted_elements}
