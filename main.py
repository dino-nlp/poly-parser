import os
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
# Ensure GraphState and create_graph_nodes are correctly imported
from graph_definition import GraphState, create_graph_nodes
from utils.file_handler import save_json_output
import argparse

# Load environment variables (e.g., OLLAMA_BASE_URL)
load_dotenv()

def run_pipeline(pdf_path: str, output_path: str, visualize: bool = True, viz_path: str = "workflow_graph.png"):
    """
    Initializes and runs the PDF processing pipeline.

    Args:
        pdf_path: Path to the input PDF file.
        output_path: Path to save the final JSON output.
        visualize: Whether to generate and save a visualization of the graph.
        viz_path: Path to save the graph visualization image.
    """
    print(f"--- Starting Pipeline for: {pdf_path} ---")

    # Initial state definition (ensure all keys from GraphState are present)
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
        "current_agent": None, # Track the current agent for debugging/logging
        "language": None,
        "metadata": None
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
    workflow.add_edge("parser_agent", "language_detection_agent")
    workflow.add_edge("language_detection_agent", "text_processor_agent")

    # Sequential flow for simplicity (can be parallelized with conditional edges)
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

    print("--- Graph Compiled ---")

    # --- Visualize the graph (Optional) ---
    if visualize:
        try:
            print(f"Attempting to visualize graph and save to {viz_path}...")
            # Get the graph object
            graph = app.get_graph()
            # Draw the graph and save as PNG
            # You might need write permissions in the target directory
            graph.draw_mermaid_png(output_file_path=viz_path)
            print(f"Graph visualization saved successfully to {viz_path}")
        except ImportError:
            print("!!! Visualization failed: `pygraphviz` not installed or Graphviz system library not found. !!!")
            print("Install Graphviz (system) and then run: pip install pygraphviz")
        except Exception as viz_error:
            print(f"!!! An error occurred during graph visualization: {viz_error} !!!")
            # Print traceback for detailed debugging if needed
            import traceback
            traceback.print_exc()


    print("--- Running Pipeline... ---")

    # Run the graph
    # Increase recursion limit if the graph is deep or has complex conditional logic
    final_state = app.invoke(initial_state, config={"recursion_limit": 25})

    print("--- Pipeline Finished ---")

    # Handle potential errors during pipeline execution
    if final_state.get("error_message"):
        print(f"Pipeline finished with error: {final_state['error_message']}")
        # Optionally save partial results or error info
        error_output = {
            "error": final_state['error_message'],
            "last_successful_agent": final_state.get('current_agent', 'Unknown'),
            # Ensure all keys from GraphState are present, even if None, for consistency
            "partial_state": {k: final_state.get(k) for k in GraphState.__annotations__ if k != 'error_message'}
        }
        save_json_output(error_output, output_path.replace(".json", "_error.json"))

    elif final_state.get("final_chunks"):
        print(f"Saving output to: {output_path}")
        # Save the final JSON output using the utility function
        # Pass only the final_chunks list to be saved
        save_json_output(final_state["final_chunks"], output_path)
    else:
         print("Pipeline finished, but no final chunks were generated.")
         # Save the final state for debugging, ensuring all keys are present
         final_state_output = {k: final_state.get(k) for k in GraphState.__annotations__}
         save_json_output(final_state_output, output_path.replace(".json", "_empty_state.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Multi-Agent PDF Processing Pipeline.")
    parser.add_argument("pdf_file", help="Path to the input PDF file.")
    parser.add_argument("-o", "--output", default="output.json", help="Path to save the output JSON file (default: output.json).")
    parser.add_argument("--noviz", action="store_true", help="Disable graph visualization generation.")
    parser.add_argument("--vizpath", default="workflow_graph.png", help="Path to save the graph visualization image (default: workflow_graph.png).")

    args = parser.parse_args()

    if not os.path.exists(args.pdf_file):
        print(f"Error: Input PDF file not found at {args.pdf_file}")
    else:
        # Pass visualization flag and path to the function
        run_pipeline(args.pdf_file, args.output, visualize=(not args.noviz), viz_path=args.vizpath)
