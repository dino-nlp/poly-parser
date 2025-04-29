from typing import Dict, Any, List
from graph_definition import GraphState
import json
import pandas as pd # Optional: For structured processing if needed

# --- Configuration ---
# Set to True to use LLM for summarizing or converting tables
USE_LLM_FOR_TABLES = True
# Output format: 'json', 'markdown', 'summary'
TABLE_OUTPUT_FORMAT = 'markdown' # Markdown is often good for LLMs

# Initialize Ollama (if used for tables)
llm_table = None
if USE_LLM_FOR_TABLES:
    try:
        from langchain_community.llms import Ollama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        import os

        llm_table = Ollama(model="llama3", base_url=os.getenv("OLLAMA_BASE_URL")) # Use a general model

        # Prompt for table processing (adjust based on desired output)
        if TABLE_OUTPUT_FORMAT == 'summary':
            prompt_text = ("You are a data analysis assistant. Analyze the following table data "
                           "(provided as a list of lists or HTML) and provide a concise summary "
                           "of its key information or purpose.\n\nTable Data:\n{table_content}\n\nSummary:")
        elif TABLE_OUTPUT_FORMAT == 'markdown':
             prompt_text = ("You are a data formatting assistant. Convert the following table data "
                            "(provided as a list of lists or HTML) into a clean GitHub-flavored Markdown table. "
                            "Ensure the headers are correctly identified if possible.\n\nTable Data:\n{table_content}\n\nMarkdown Table:")
        else: # Default to JSON or just pass through
             prompt_text = None # No LLM needed if just passing JSON

        if prompt_text:
            table_prompt = ChatPromptTemplate.from_template(prompt_text)
            table_chain = table_prompt | llm_table | StrOutputParser()
        else:
            table_chain = None

        print("Initialized Ollama for table analysis.")

    except ImportError:
        print("Required libraries for LLM table analysis not found.")
        USE_LLM_FOR_TABLES = False
        table_chain = None
    except Exception as e:
        print(f"Failed to initialize Ollama for tables: {e}")
        USE_LLM_FOR_TABLES = False
        table_chain = None


def format_table_to_md(table_data: List[List[str]]) -> str:
    """Converts a list of lists into a Markdown table."""
    if not table_data:
        return ""
    try:
        # Assume first row is header
        header = table_data[0]
        rows = table_data[1:]

        # Create header line and separator
        md = "| " + " | ".join(map(str, header)) + " |\n"
        md += "|-" + "-|".join(['-' * len(str(h)) for h in header]) + "-|\n"

        # Create rows
        for row in rows:
            # Ensure row has same number of columns as header, pad if necessary
            padded_row = row + ["" for _ in range(len(header) - len(row))]
            md += "| " + " | ".join(map(str, padded_row[:len(header)])) + " |\n" # Use only expected number of cells

        return md.strip()
    except Exception as e:
        print(f"Error formatting table to Markdown: {e}")
        # Fallback: simple representation
        return "\n".join(["\t".join(map(str, row)) for row in table_data])


def analyze_tables(state: GraphState) -> Dict[str, Any]:
    """
    Agent 5: Analyzes and standardizes tables found by the parser.

    Args:
        state: The current graph state.

    Returns:
        A dictionary with the updated 'table_data'.
    """
    print("Analyzing tables...")
    raw_elements = state.get("raw_elements", [])
    processed_tables = []

    # Find table elements (can be 'table' from PyMuPDF, 'table_html' from unstructured, etc.)
    table_elements = [el for el in raw_elements if el.get("type") in ["table", "table_html"]]

    if not table_elements:
        print("No table elements found to analyze.")
        return {}

    print(f"Found {len(table_elements)} table elements.")

    for i, table_el in enumerate(table_elements):
        print(f"  Processing table {i+1}/{len(table_elements)}...")
        content = table_el.get("content")
        metadata = table_el.get("metadata", {})
        table_type = table_el.get("type")
        output_content = content # Default: pass through original content
        format_used = "original"

        try:
            table_str_for_llm = ""
            if table_type == "table" and isinstance(content, list): # PyMuPDF list of lists
                table_str_for_llm = format_table_to_md(content) # Convert to MD for LLM
                 # Or use pandas for intermediate structure:
                 # df = pd.DataFrame(content[1:], columns=content[0])
                 # table_str_for_llm = df.to_markdown(index=False)
            elif table_type == "table_html" and isinstance(content, str): # Unstructured HTML
                table_str_for_llm = content # Pass HTML directly if LLM can handle it
                # Alternative: Use pandas to parse HTML table
                # try:
                #     dfs = pd.read_html(io.StringIO(content))
                #     if dfs:
                #         df = dfs[0] # Assume first table
                #         table_str_for_llm = df.to_markdown(index=False)
                # except Exception as pd_e:
                #     print(f"    Pandas failed to parse HTML table: {pd_e}")
                #     # Keep original HTML for LLM
            else:
                 table_str_for_llm = str(content) # Fallback to string representation


            if USE_LLM_FOR_TABLES and table_chain and table_str_for_llm:
                print(f"    Processing table with LLM (Output: {TABLE_OUTPUT_FORMAT})...")
                llm_result = table_chain.invoke({"table_content": table_str_for_llm})
                output_content = llm_result.strip()
                format_used = f"llm_{TABLE_OUTPUT_FORMAT}"
            elif TABLE_OUTPUT_FORMAT == 'markdown' and table_type == "table":
                 print("    Formatting table to Markdown (basic)...")
                 output_content = format_table_to_md(content)
                 format_used = 'markdown_basic'
            elif TABLE_OUTPUT_FORMAT == 'json' and table_type == "table":
                 print("    Formatting table to JSON...")
                 # Convert list of lists to list of dicts (assuming first row is header)
                 if content and len(content) > 0:
                     header = content[0]
                     data = [dict(zip(header, row)) for row in content[1:]]
                     output_content = json.dumps(data, indent=2)
                     format_used = 'json'
                 else:
                     output_content = json.dumps([]) # Empty list for empty table
                     format_used = 'json'
            # Add more format conversions if needed (e.g., HTML to JSON)

            processed_tables.append({
                "table_ref": f"table_{metadata.get('page_number', 'N')}_{metadata.get('table_index', i)}",
                "data": output_content,
                "format": format_used,
                "metadata": metadata
            })

        except Exception as e:
            print(f"    Error processing table {i+1}: {e}")
            # Store error or fallback content
            processed_tables.append({
                "table_ref": f"table_{metadata.get('page_number', 'N')}_{metadata.get('table_index', i)}",
                "data": f"Error processing table: {e}",
                "format": "error",
                "metadata": metadata
            })


    print(f"Finished table analysis. Processed {len(processed_tables)} tables.")
    return {"table_data": processed_tables}
