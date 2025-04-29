from typing import TypedDict, List, Dict, Any, Optional
import importlib # Dùng để import động nếu cần, nhưng trực tiếp sẽ rõ hơn
import traceback # Để in lỗi chi tiết hơn

# --- State Definition ---
# Định nghĩa GraphState ở đây, trước khi import các agent cần nó
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
        metadata: Optional[Dict[str, Any]] # Document-level metadata
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
# Di chuyển import vào đây để tránh circular import
def create_graph_nodes() -> Dict[str, callable]:
    """
    Creates and returns a dictionary mapping node names to their corresponding agent functions.
    Imports agent modules only when this function is called.
    """
    # Import agent modules *inside* the function
    try:
        from agents import parser
        from agents import language_detector
        from agents import text_processor
        from agents import image_analyzer
        from agents import chart_analyzer
        from agents import table_analyzer
        from agents import synthesizer
        from agents import chunker
        from agents import formatter
    except ImportError as e:
        print(f"!!! Failed to import agent modules: {e} !!!")
        print("Ensure all agent files exist in the 'agents' directory and have no syntax errors.")
        raise # Re-raise the error to stop execution if imports fail

    # Wrapper function remains the same
    def wrap_agent(agent_func, agent_name):
        def node_func(state: GraphState) -> Dict[str, Any]:
            print(f"--- Running Agent: {agent_name} ---")
            # Check for previous error before running
            if state.get("error_message"):
                 print(f"--- Skipping Agent: {agent_name} due to previous error: {state['error_message']} ---")
                 # Return an empty dict, indicating no changes to the state by this skipped agent
                 # The error message persists from the previous state.
                 return {}

            try:
                # Pass the relevant parts of the state to the agent
                # The agent function should know what it needs from the state
                updated_state_parts = agent_func(state) # Call the actual agent function
                if updated_state_parts is None: # Agent might return None if no changes
                    updated_state_parts = {}

                # Ensure the agent returned a dictionary
                if not isinstance(updated_state_parts, dict):
                     print(f"!!! Warning: Agent {agent_name} did not return a dictionary. Returning empty update. !!!")
                     updated_state_parts = {}


                updated_state_parts["current_agent"] = agent_name # Update tracker
                # Clear previous error message *only if* this agent succeeds *and* returns something
                # We keep the error if the agent returns nothing or fails
                if updated_state_parts: # Check if the agent actually returned updates
                    updated_state_parts["error_message"] = None

                return updated_state_parts
            except Exception as e:
                print(f"!!! Error in Agent {agent_name}: {e} !!!")
                # Log the error traceback if needed
                traceback.print_exc()
                # Return the error message in the state update
                # This will cause subsequent agents to be skipped by the check above
                return {"error_message": f"Error in {agent_name}: {str(e)}", "current_agent": agent_name}
        return node_func

    # Create the dictionary of nodes using the imported agent functions
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
#         # Using END requires importing it: from langgraph.graph import END
#         from langgraph.graph import END
#         return END # Go directly to end if parsing failed
#     # ... rest of the logic ...

# --- Graph Assembly (in main.py) ---
# The actual graph (StateGraph instance, adding nodes and edges)
# will be constructed in main.py using the nodes defined here.
