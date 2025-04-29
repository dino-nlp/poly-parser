import fitz # PyMuPDF
from typing import Dict, Any, List
from graph_definition import GraphState # Import state definition for type hinting

# Placeholder for more advanced parsing like unstructured.io
# from unstructured.partition.pdf import partition_pdf

def parse_document(state: GraphState) -> Dict[str, Any]:
    """
    Agent 1: Parses the PDF document to extract raw elements.

    Args:
        state: The current graph state containing the pdf_path.

    Returns:
        A dictionary with the updated 'raw_elements' and 'metadata'.
    """
    pdf_path = state["pdf_path"]
    raw_elements = []
    doc_metadata = {"source": pdf_path}

    try:
        print(f"Parsing document: {pdf_path}")
        doc = fitz.open(pdf_path)
        doc_metadata["page_count"] = doc.page_count
        # Add more metadata extraction if needed (title, author, etc.)
        # doc_metadata.update(doc.metadata) # Be careful, metadata can be messy

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_metadata = {"page_number": page_num + 1}

            # 1. Extract Text Blocks
            text_blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
            for block in text_blocks:
                if block['type'] == 0: # Text block
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"] + " "
                        block_text += "\n" # Add newline after each line
                    if block_text.strip():
                         raw_elements.append({
                             "type": "text",
                             "content": block_text.strip(),
                             "metadata": {**page_metadata, "bbox": block["bbox"]} # Add bounding box
                         })

            # 2. Extract Images (References)
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                # In a real scenario, you might save the image temporarily or pass bytes
                # For simplicity here, we just note its existence and location.
                # Agent 3 (Image Analyzer) would need access to the actual image data later.
                # This might require saving images temporarily or passing bytes in the state (can be large).
                # Let's store a reference for now.
                raw_elements.append({
                    "type": "image_ref",
                    "content": f"Image_{page_num + 1}_{img_index}.{image_ext}", # Placeholder name
                    "metadata": {
                        **page_metadata,
                        "xref": xref,
                        # "bbox": page.get_image_bbox(img_info).irect # Get bbox if needed
                        # Storing image_bytes directly in state is usually not recommended
                        # Consider saving to a temp dir and passing the path, or using a shared store.
                        "temp_image_path": None # Placeholder for path if saved
                    }
                })

            # 3. Extract Tables (Basic Heuristics or use libraries like camelot-py or unstructured)
            # PyMuPDF has basic table detection, but it's often not robust.
            # find_tables() returns TableFinder object
            tables = page.find_tables()
            for i, tab in enumerate(tables):
                 # tab.extract() gives the table content as list of lists
                 table_content = tab.extract()
                 # You might want to convert this to Markdown or JSON
                 raw_elements.append({
                     "type": "table",
                     "content": table_content, # Store as list of lists for now
                     "metadata": {**page_metadata, "bbox": tab.bbox, "table_index": i}
                 })

            # 4. Placeholder for Charts (requires more advanced analysis)
            # Chart detection is complex. Often treated as images initially.

        print(f"Parsed {len(raw_elements)} raw elements from {doc.page_count} pages.")
        doc.close()

        return {"raw_elements": raw_elements, "metadata": doc_metadata}

    except Exception as e:
        print(f"Error parsing PDF {pdf_path}: {e}")
        # Raise the exception to be caught by the node wrapper
        raise e

# --- Example using unstructured (more powerful but complex setup) ---
# def parse_document_unstructured(state: GraphState) -> Dict[str, Any]:
#     pdf_path = state["pdf_path"]
#     print(f"Parsing document with unstructured: {pdf_path}")
#     try:
#         # Configure unstructured as needed (e.g., strategy='hi_res' for layout detection)
#         # This might require installing specific dependencies like tesseract, poppler, etc.
#         elements = partition_pdf(
#             filename=pdf_path,
#             strategy="auto", # or "hi_res" if using models
#             # hi_res_model_name="yolox" # Example if using hi_res
#             infer_table_structure=True,
#             extract_images_in_pdf=True, # Requires appropriate setup
#             # chunking_strategy="by_title" # Example chunking within unstructured
#         )
#
#         raw_elements = []
#         for element in elements:
#             el_data = {
#                 "type": type(element).__name__, # e.g., Title, NarrativeText, Table, Image
#                 "content": element.text if hasattr(element, 'text') else str(element), # Basic text
#                 "metadata": element.metadata.to_dict()
#             }
#             # Handle specific types like tables or images if needed
#             if el_data['type'] == 'Table' and hasattr(element, 'metadata') and hasattr(element.metadata, 'text_as_html'):
#                  el_data['content'] = element.metadata.text_as_html # Get HTML representation
#                  el_data['type'] = 'table_html' # Mark as HTML table
#
#             # Handle images (unstructured might save them)
#             # if el_data['type'] == 'Image': ...
#
#             raw_elements.append(el_data)
#
#         print(f"Parsed {len(raw_elements)} elements using unstructured.")
#         # Extract document-level metadata if available from unstructured
#         doc_metadata = {"source": pdf_path, "parser": "unstructured"}
#
#         return {"raw_elements": raw_elements, "metadata": doc_metadata}
#
#     except Exception as e:
#         print(f"Error parsing PDF {pdf_path} with unstructured: {e}")
#         raise e

