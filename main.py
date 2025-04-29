import os
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from graph_definition import GraphState, create_graph_nodes # Import state and node creation function
from utils.file_handler import save_json_output
import argparse

# Load environment variables (e.g., OLLAMA_BASE_URL)
load_dotenv()

def run_pipeline(pdf_path: str, output_path: str):
    """
    Initializes and runs the PDF processing pipeline.

    Args:
        pdf_path: Path to the input PDF file.
        output_path: Path to save the final JSON output.
    """
    print(f"--- Starting Pipeline for: {pdf_path} ---")

    # Initial state
    initial_state: GraphState = {
        "pdf_path": pdf_path,
        "raw_elements": [],
        "processed_text_chunks": [],
        "image_descriptions": [],
        "chart_summaries": [],
        "table_data": [],
        "synthesized_content": [],
        "final_chunks": [],
        "error_message": None,
        "current_agent": None # Track the current agent for debugging/logging
    }

    # Create the graph workflow
    workflow = StateGraph(GraphState)

    # --- Define Nodes ---
    # create_graph_nodes returns a dictionary of node_name: node_function
    nodes = create_graph_nodes()
    for name, node_func in nodes.items():
        workflow.add_node(name, node_func)

    # --- Define Edges (Simplified Linear Flow for now) ---
    # This defines the sequence of agents. More complex routing can be added later.
    workflow.set_entry_point("parser_agent")
    workflow.add_edge("parser_agent", "language_detection_agent") # Or directly to text_processor if lang detection is skipped
    workflow.add_edge("language_detection_agent", "text_processor_agent")

    # Branching after text processing to handle different content types in parallel (conceptual)
    # LangGraph conditional edges would be needed here for actual parallel execution logic
    # For simplicity, we run them sequentially first.
    workflow.add_edge("text_processor_agent", "image_analyzer_agent")
    workflow.add_edge("image_analyzer_agent", "chart_analyzer_agent")
    workflow.add_edge("chart_analyzer_agent", "table_analyzer_agent")

    # Join back for synthesis
    workflow.add_edge("table_analyzer_agent", "synthesizer_agent")
    workflow.add_edge("synthesizer_agent", "chunker_agent")
    workflow.add_edge("chunker_agent", "formatter_agent")
    workflow.add_edge("formatter_agent", END) # End of the graph

    # Compile the graph
    app = workflow.compile()

    print("--- Graph Compiled. Running... ---")

    # Run the graph
    final_state = app.invoke(initial_state, config={"recursion_limit": 15}) # Increase recursion limit if needed

    print("--- Pipeline Finished ---")

    # Handle potential errors
    if final_state.get("error_message"):
        print(f"Pipeline finished with error: {final_state['error_message']}")
        # Optionally save partial results or error info
        error_output = {
            "error": final_state['error_message'],
            "last_successful_agent": final_state.get('current_agent', 'Unknown'),
            "partial_state": {k: v for k, v in final_state.items() if k != 'error_message'} # Exclude error itself
        }
        save_json_output(error_output, output_path.replace(".json", "_error.json"))

    elif final_state.get("final_chunks"):
        print(f"Saving output to: {output_path}")
        # Save the final JSON output using the utility function
        save_json_output(final_state["final_chunks"], output_path)
    else:
         print("Pipeline finished, but no final chunks were generated.")
         # Save the final state for debugging
         save_json_output(final_state, output_path.replace(".json", "_empty_state.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Multi-Agent PDF Processing Pipeline.")
    parser.add_argument("pdf_file", help="Path to the input PDF file.")
    parser.add_argument("-o", "--output", default="output.json", help="Path to save the output JSON file (default: output.json).")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_file):
        print(f"Error: Input PDF file not found at {args.pdf_file}")
    else:
        run_pipeline(args.pdf_file, args.output)
