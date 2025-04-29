from typing import TypedDict, List, Dict, Any, Optional
from agents import parser, language_detector, text_processor, image_analyzer, chart_analyzer, table_analyzer, synthesizer, chunker, formatter

# --- State Definition ---
# This TypedDict defines the structure of the data that flows through the graph.
# Each agent will read from and write to this state.
class GraphState(TypedDict):
    """
    Represents the state of the graph at any point in time.

    Attributes:
        pdf_path: Path to the input PDF file.
        raw_elements: List of raw elements extracted by the parser (text, images, tables).
                      Each element could be a dictionary with 'type', 'content', 'metadata'.
        processed_text_chunks: List of cleaned and potentially enhanced text chunks.
        image_descriptions: List of descriptions generated for images.
        chart_summaries: List of summaries generated for charts.
        table_data: List of processed table data (e.g., as JSON or Markdown).
        synthesized_content: A structured representation of the document content after synthesis.
        final_chunks: The final list of semantic chunks ready for output/embedding.
        error_message: Stores any error message encountered during processing.
        current_agent: Name of the agent currently processing or last processed.
        # Add other relevant fields as needed (e.g., document metadata, language info)
        language: Optional[str] # Detected language (if applicable)
        metadata: Dict[str, Any] # Document-level metadata
    """
    pdf_path: str
    raw_elements: List[Dict[str, Any]]
    processed_text_chunks: List[Dict[str, Any]] # e.g., {'text': '...', 'metadata': {...}}
    image_descriptions: List[Dict[str, Any]] # e.g., {'image_ref': 'img1', 'description': '...', 'metadata': {...}}
    chart_summaries: List[Dict[str, Any]] # e.g., {'chart_ref': 'chart1', 'summary': '...', 'metadata': {...}}
    table_data: List[Dict[str, Any]] # e.g., {'table_ref': 'tbl1', 'data': {...}, 'metadata': {...}}
    synthesized_content: List[Dict[str, Any]] # Combined content in logical order
    final_chunks: List[Dict[str, Any]] # Final output chunks
    error_message: Optional[str]
    current_agent: Optional[str]
    language: Optional[str]
    metadata: Optional[Dict[str, Any]]


# --- Node Creation Function ---
# This function centralizes the creation of node functions for the graph.
def create_graph_nodes() -> Dict[str, callable]:
    """
    Creates and returns a dictionary mapping node names to their corresponding agent functions.
    """
    # Instantiate agents or get their processing functions
    # Note: Agent functions should accept the GraphState dict as input
    # and return a dictionary containing the fields they've updated.

    # Example (replace with actual agent function calls):
    def wrap_agent(agent_func, agent_name):
        def node_func(state: GraphState) -> Dict[str, Any]:
            print(f"--- Running Agent: {agent_name} ---")
            try:
                # Pass the relevant parts of the state to the agent
                # The agent function should know what it needs from the state
                updated_state_parts = agent_func(state)
                updated_state_parts["current_agent"] = agent_name # Update tracker
                updated_state_parts["error_message"] = None # Clear previous errors if successful
                return updated_state_parts
            except Exception as e:
                print(f"!!! Error in Agent {agent_name}: {e} !!!")
                # Log the error traceback if needed
                import traceback
                traceback.print_exc()
                return {"error_message": f"Error in {agent_name}: {str(e)}", "current_agent": agent_name}
        return node_func

    nodes = {
        "parser_agent": wrap_agent(parser.parse_document, "Parser"),
        "language_detection_agent": wrap_agent(language_detector.detect_language, "Language Detector"),
        "text_processor_agent": wrap_agent(text_processor.process_text, "Text Processor"),
        "image_analyzer_agent": wrap_agent(image_analyzer.analyze_images, "Image Analyzer"),
        "chart_analyzer_agent": wrap_agent(chart_analyzer.analyze_charts, "Chart Analyzer"),
        "table_analyzer_agent": wrap_agent(table_analyzer.analyze_tables, "Table Analyzer"),
        "synthesizer_agent": wrap_agent(synthesizer.synthesize_content, "Synthesizer"),
        "chunker_agent": wrap_agent(chunker.create_chunks, "Chunker"),
        "formatter_agent": wrap_agent(formatter.format_output, "Formatter"),
    }
    return nodes

# --- Conditional Edges (Example - Implement as needed) ---
# You might need conditional logic, e.g., skipping image analysis if no images were found.
# def decide_after_parsing(state: GraphState) -> str:
#     """Determines the next step after parsing."""
#     if state.get("error_message"):
#         return END # Go directly to end if parsing failed
#     if any(el['type'] == 'image' for el in state['raw_elements']):
#         # Need to decide how to route - maybe always go to text first?
#         return "text_processor_agent" # Or branch out
#     else:
#         return "text_processor_agent" # Skip image analysis if no images

# --- Graph Assembly (in main.py) ---
# The actual graph (StateGraph instance, adding nodes and edges)
# will be constructed in main.py using the nodes defined here.
