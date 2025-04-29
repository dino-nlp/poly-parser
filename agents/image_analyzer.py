from typing import Dict, Any, List
from graph_definition import GraphState
import os
import fitz # To extract image bytes if needed

# --- Configuration ---
# Set to True to use a local multi-modal LLM (like Llava via Ollama)
# Requires Ollama server running with a multi-modal model: ollama run llava
USE_MULTIMODAL_LLM = True
# Set to True to attempt OCR if LLM fails or isn't used
USE_OCR_FALLBACK = False # Requires pytesseract and Tesseract install

# Initialize Ollama for image analysis (if used)
llm_image = None
if USE_MULTIMODAL_LLM:
    try:
        from langchain_community.llms import Ollama
        # Ensure the model name here matches your multi-modal model in Ollama
        llm_image = Ollama(model="llava", base_url=os.getenv("OLLAMA_BASE_URL"))
        print("Initialized Ollama for image analysis (llava).")
    except ImportError:
        print("langchain_community.llms.Ollama not found. Cannot use LLM for image analysis.")
        USE_MULTIMODAL_LLM = False
    except Exception as e:
        print(f"Failed to initialize Ollama for images: {e}")
        USE_MULTIMODAL_LLM = False

# OCR setup (if used)
# import pytesseract
# from PIL import Image
# import io
# if USE_OCR_FALLBACK:
#     try:
#         # You might need to set the TESSERACT_CMD environment variable or path
#         # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' # Example path
#         pass # Basic import check
#     except ImportError:
#         print("Pytesseract or Pillow not found. Cannot use OCR fallback.")
#         USE_OCR_FALLBACK = False


def analyze_images(state: GraphState) -> Dict[str, Any]:
    """
    Agent 3: Analyzes images identified by the parser.
    Generates descriptions (captioning) and performs OCR if applicable.

    Args:
        state: The current graph state.

    Returns:
        A dictionary with the updated 'image_descriptions'.
    """
    print("Analyzing images...")
    raw_elements = state.get("raw_elements", [])
    pdf_path = state["pdf_path"]
    image_descriptions = []
    doc = None # Keep doc open only if needed

    image_refs = [el for el in raw_elements if el.get("type") == "image_ref"]
    if not image_refs:
        print("No image references found to analyze.")
        return {}

    print(f"Found {len(image_refs)} image references.")

    try:
        # Open the PDF only once if we need to extract image bytes
        if USE_MULTIMODAL_LLM or USE_OCR_FALLBACK:
            doc = fitz.open(pdf_path)

        for i, img_ref in enumerate(image_refs):
            img_metadata = img_ref.get("metadata", {})
            img_name = img_ref.get("content", f"image_{i}")
            xref = img_metadata.get("xref")
            print(f"  Analyzing image {i+1}/{len(image_refs)}: {img_name} (xref: {xref})")

            description = f"Image: {img_name}" # Default description
            ocr_text = None
            image_bytes = None

            # --- Get Image Data ---
            if (USE_MULTIMODAL_LLM or USE_OCR_FALLBACK) and xref and doc:
                try:
                    base_image = doc.extract_image(xref)
                    if base_image:
                        image_bytes = base_image["image"]
                    else:
                        print(f"    Could not extract image for xref {xref}.")
                except Exception as e:
                    print(f"    Error extracting image bytes for xref {xref}: {e}")

            # --- Use Multi-modal LLM for Description/OCR ---
            if USE_MULTIMODAL_LLM and llm_image and image_bytes:
                print("    Attempting analysis with multi-modal LLM...")
                try:
                    # Langchain's Ollama integration might need specific input format
                    # for multi-modal. Check langchain-ollama documentation.
                    # This is a conceptual example - syntax might differ:
                    # llm_result = llm_image.invoke(
                    #     prompt="Describe this image in detail. If it contains text, extract it.",
                    #     images=[image_bytes] # How images are passed might vary
                    # )
                    # For now, let's assume a simplified direct call if possible,
                    # or you might need langchain multi-modal message types.

                    # Placeholder: Assume direct call with base64 encoded image if supported
                    import base64
                    img_base64 = base64.b64encode(image_bytes).decode('utf-8')

                    # Construct prompt for Llava (or similar model)
                    prompt = "Describe this image in detail. What does it show? Is there any text visible? If yes, what does the text say?"

                    # Use the multimodal invoke method (check latest Langchain docs)
                    # This might involve HumanMessage with image content
                    from langchain_core.messages import HumanMessage
                    msg = HumanMessage(
                        content=[
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                         ]
                    )
                    llm_result = llm_image.invoke(msg)


                    print(f"    LLM Result (raw): {llm_result[:100]}...") # Print start of result
                    # Parse the result (this depends heavily on the LLM's output format)
                    # Simple approach: Assume description is the main part, look for OCR cues
                    description = llm_result.strip()
                    # Try to extract OCR'd text if the LLM indicated it found some
                    # This requires specific prompting and parsing logic
                    if "text says:" in llm_result.lower():
                        # Attempt to extract text after a marker
                        parts = re.split(r'text says:', llm_result, flags=re.IGNORECASE)
                        if len(parts) > 1:
                            ocr_text = parts[1].strip()
                            # Maybe remove the OCR part from the main description
                            description = parts[0].strip()

                except Exception as e:
                    print(f"    Multi-modal LLM analysis failed: {e}")
                    # Fallback?

            # --- Fallback to OCR (if LLM failed or wasn't used) ---
            # if not ocr_text and USE_OCR_FALLBACK and image_bytes:
            #     print("    Attempting OCR fallback...")
            #     try:
            #         img_pil = Image.open(io.BytesIO(image_bytes))
            #         ocr_text = pytesseract.image_to_string(img_pil).strip()
            #         if ocr_text:
            #             print(f"    OCR found text: {ocr_text[:100]}...")
            #             # Add OCR text to description if no LLM description exists
            #             if description == f"Image: {img_name}":
            #                 description = f"Image: {img_name} containing text."
            #         else:
            #              print("    OCR did not find significant text.")
            #     except Exception as e:
            #         print(f"    OCR fallback failed: {e}")

            # --- Store Result ---
            image_descriptions.append({
                "image_ref": img_name,
                "description": description,
                "ocr_text": ocr_text if ocr_text else None,
                "analysis_method": "llm" if USE_MULTIMODAL_LLM and image_bytes else ("ocr" if ocr_text else "none"),
                "metadata": img_metadata
            })

    except Exception as e:
        print(f"Error during image analysis setup or loop: {e}")
        # Raise exception to be caught by the node wrapper
        raise e
    finally:
        if doc:
            doc.close()

    print(f"Finished image analysis. Generated {len(image_descriptions)} descriptions.")
    return {"image_descriptions": image_descriptions}

