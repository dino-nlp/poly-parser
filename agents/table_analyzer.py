from typing import Dict, Any, List
from graph_definition import GraphState
import json
import pandas as pd # Optional: For structured processing if needed
import io # For using StringIO with pandas read_html

# --- Configuration ---
USE_LLM_FOR_TABLES = True
TABLE_OUTPUT_FORMAT = 'markdown' # 'json', 'markdown', 'summary'
DEFAULT_LANGUAGE = "English" # Fallback language

# Initialize Ollama (if used for tables)
llm_table = None
table_chain = None
if USE_LLM_FOR_TABLES:
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        import os

        llm_table = ChatOllama(model=os.getenv("TABLE_ANALYZER_MODEL"), temperature=0)
        print(f"LLM model for table analyzer: {os.getenv("TABLE_ANALYZER_MODEL")}")

        # Define prompts based on output format, including language
        prompt_text = None
        if TABLE_OUTPUT_FORMAT == 'summary':
            prompt_text = ("You are a data analysis assistant. Analyze the following table data "
                        "(provided as Markdown, list of lists, or HTML) and provide a concise summary "
                        "of its key information or purpose in {language}.\n\nTable Data:\n{table_content}\n\nSummary (in {language}):"
                        "NO FURTHER EXPLANATION, JUST PROVIDE THE RESULT.")
        elif TABLE_OUTPUT_FORMAT == 'markdown':
             prompt_text = ("You are a data formatting assistant. Convert the following table data "
                            "(provided as list of lists or HTML) into a clean GitHub-flavored Markdown table. "
                            "Ensure the headers are correctly identified if possible. Respond ONLY with the Markdown table.\n\nTable Data:\n{table_content}\n\nMarkdown Table:"
                            "NO FURTHER EXPLANATION, JUST PROVIDE THE RESULT.")
             # Note: Markdown itself is language-agnostic, but the LLM processing it understands the context language.
        # Add other formats if needed

        if prompt_text:
            table_prompt = ChatPromptTemplate.from_template(prompt_text)
            table_chain = table_prompt | llm_table | StrOutputParser()
            print("Initialized Ollama for table analysis.")
        else:
            # No LLM needed if just converting to JSON or passing through original
            USE_LLM_FOR_TABLES = False
            print("LLM usage for tables disabled as no suitable prompt for format/language.")

    except ImportError:
        print("Required libraries for LLM table analysis not found.")
        USE_LLM_FOR_TABLES = False
    except Exception as e:
        print(f"Failed to initialize Ollama for tables: {e}")
        USE_LLM_FOR_TABLES = False


def format_table_to_md(table_data: List[List[str]]) -> str:
    """Converts a list of lists into a Markdown table."""
    # ... (function remains the same)
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
    Agent 5: Analyzes and standardizes tables, considering language for summaries.
    """
    print("Analyzing tables...")
    raw_elements = state.get("raw_elements", [])
    # Get detected language from state, fallback to default
    language = state.get("language", DEFAULT_LANGUAGE)
    print(f"  Using language: {language}")

    processed_tables = []

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
        output_content = content
        format_used = "original"
        input_for_llm = "" # Prepare input string for LLM

        try:
            # --- Prepare input for LLM or direct conversion ---
            if table_type == "table" and isinstance(content, list):
                # Convert list of lists to Markdown for LLM or direct use
                input_for_llm = format_table_to_md(content)
                if TABLE_OUTPUT_FORMAT == 'markdown' and not USE_LLM_FOR_TABLES:
                    output_content = input_for_llm
                    format_used = 'markdown_basic'
                elif TABLE_OUTPUT_FORMAT == 'json' and not USE_LLM_FOR_TABLES:
                     if content and len(content) > 0:
                         header = content[0]
                         data = [dict(zip(header, row)) for row in content[1:]]
                         output_content = json.dumps(data, indent=2)
                         format_used = 'json'
                     else:
                         output_content = json.dumps([])
                         format_used = 'json'
                # Else, input_for_llm will be used by LLM below

            elif table_type == "table_html" and isinstance(content, str):
                input_for_llm = content # Pass HTML to LLM
                # Optionally use pandas to convert HTML to MD first
                # try:
                #     dfs = pd.read_html(io.StringIO(content))
                #     if dfs: input_for_llm = dfs[0].to_markdown(index=False)
                # except Exception: pass # Keep original HTML if parse fails

            else:
                 input_for_llm = str(content) # Fallback

            # --- Use LLM if configured ---
            if USE_LLM_FOR_TABLES and table_chain and input_for_llm:
                print(f"    Processing table with LLM (Output: {TABLE_OUTPUT_FORMAT}, Lang: {language})...")
                llm_result = table_chain.invoke({
                    "table_content": input_for_llm,
                    "language": language # Pass language to the prompt context
                })
                output_content = llm_result.strip()
                format_used = f"llm_{TABLE_OUTPUT_FORMAT}"
            elif not format_used.startswith('llm') and format_used == 'original':
                # Handle cases where no direct conversion happened and LLM wasn't used
                if TABLE_OUTPUT_FORMAT == 'markdown':
                     output_content = input_for_llm # Use the prepared MD/HTML/str
                     format_used = 'markdown_fallback' if table_type != "table" else 'markdown_basic'
                elif TABLE_OUTPUT_FORMAT == 'json':
                     # Attempt JSON conversion if possible, otherwise keep original string
                     try:
                         # This might fail if input_for_llm isn't valid JSON structure
                         output_content = json.dumps(input_for_llm) # Less likely to be useful
                         format_used = 'json_fallback'
                     except TypeError:
                         output_content = input_for_llm # Keep original
                         format_used = 'original_string'
                else: # e.g., summary requested but LLM disabled
                    output_content = f"Table content (Format: {table_type}):\n" + input_for_llm
                    format_used = 'original_string'


            processed_tables.append({
                "table_ref": f"table_{metadata.get('page_number', 'N')}_{metadata.get('table_index', i)}",
                "data": output_content,
                "format": format_used,
                "analysis_language": language if format_used.startswith('llm_summary') else None, # Track lang only if summary generated
                "metadata": metadata
            })

        except Exception as e:
            print(f"    Error processing table {i+1}: {e}")
            processed_tables.append({
                "table_ref": f"table_{metadata.get('page_number', 'N')}_{metadata.get('table_index', i)}",
                "data": f"Error processing table: {e}",
                "format": "error",
                "metadata": metadata
            })

    print(f"Finished table analysis. Processed {len(processed_tables)} tables.")
    return {"table_data": processed_tables}
