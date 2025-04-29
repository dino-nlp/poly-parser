from typing import Dict, Any, List
from graph_definition import GraphState
import os
import fitz # To extract image bytes if needed
import base64
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# --- Configuration ---
USE_MULTIMODAL_LLM = True
USE_OCR_FALLBACK = False # Requires pytesseract and Tesseract install
DEFAULT_LANGUAGE = "English" # Fallback language


# Initialize Ollama for image analysis (if used)
llm_image = None
if USE_MULTIMODAL_LLM:
    try:
        from langchain_ollama import ChatOllama
        llm_image = ChatOllama(model=os.getenv("IMAGE_ANALYZER_MODEL"), temperature=0)
        print("Initialized Ollama for image analysis (llava).")
    except ImportError:
        print("langchain_community.llms.Ollama not found. Cannot use LLM for image analysis.")
        USE_MULTIMODAL_LLM = False
    except Exception as e:
        print(f"Failed to initialize Ollama for images: {e}")
        USE_MULTIMODAL_LLM = False

# OCR setup (if used)
# ...

def analyze_images(state: GraphState) -> Dict[str, Any]:
    """
    Agent 3: Analyzes images, generating descriptions/OCR in the detected language.
    """
    print("Analyzing images...")
    raw_elements = state.get("raw_elements", [])
    pdf_path = state["pdf_path"]
    # Get detected language from state, fallback to default
    language = state.get("language", DEFAULT_LANGUAGE)
    print(f"  Using language: {language}")

    image_descriptions = []
    doc = None

    image_refs = [el for el in raw_elements if el.get("type") == "image_ref"]
    if not image_refs:
        print("No image references found to analyze.")
        return {}

    print(f"Found {len(image_refs)} image references.")

    try:
        if (USE_MULTIMODAL_LLM or USE_OCR_FALLBACK) and image_refs:
             doc = fitz.open(pdf_path)

        for i, img_ref in enumerate(image_refs):
            img_metadata = img_ref.get("metadata", {})
            img_name = img_ref.get("content", f"image_{i}")
            xref = img_metadata.get("xref")
            print(f"  Analyzing image {i+1}/{len(image_refs)}: {img_name} (xref: {xref})")

            description = f"Image: {img_name}"
            ocr_text = None
            image_bytes = None

            # --- Get Image Data ---
            if xref and doc:
                try:
                    base_image = doc.extract_image(xref)
                    if base_image: image_bytes = base_image["image"]
                    else: print(f"    Could not extract image for xref {xref}.")
                except Exception as e: print(f"    Error extracting image bytes for xref {xref}: {e}")

            # --- Use Multi-modal LLM ---
            if USE_MULTIMODAL_LLM and llm_image and image_bytes:
                print("    Attempting analysis with multi-modal LLM...")
                try:
                    img_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    # Include language in the prompt
                    prompt = f"Describe this image in detail in {language}. What does it show? Is there any text visible? If yes, extract the text exactly as it appears."
                    # system_prompt = "You are an assistant tasked with describing table or image"
                    # system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
                    # msg = HumanMessage(
                    #     content=[
                    #         {"type": "text", "text": prompt},
                    #         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    #      ]
                    # )
                    human_prompt = [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + "{image_base64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ]
                    llm_result = llm_image.invoke(msg)
                    print(f"    LLM Result (raw): {llm_result[:100]}...")
                    description = llm_result.strip()
                    # Simple OCR extraction attempt (adjust based on LLM output format)
                    # Look for common phrases LLM might use before extracted text
                    ocr_markers = ["extracted text:", "text says:", "text visible:", "text found:"]
                    ocr_found = False
                    for marker in ocr_markers:
                         if marker in llm_result.lower():
                             parts = re.split(f'{marker}', llm_result, flags=re.IGNORECASE, maxsplit=1)
                             if len(parts) > 1:
                                 ocr_text = parts[1].strip()
                                 # Optionally remove OCR part from main description if desired
                                 # description = parts[0].strip()
                                 ocr_found = True
                                 break
                    if ocr_found:
                         print(f"    Extracted potential OCR text: {ocr_text[:100]}...")


                except Exception as e:
                    print(f"    Multi-modal LLM analysis failed: {e}")

            # --- Fallback to OCR ---
            # if not ocr_text and USE_OCR_FALLBACK and image_bytes:
            #    ... (OCR logic remains the same)

            # --- Store Result ---
            image_descriptions.append({
                "image_ref": img_name,
                "description": description,
                "ocr_text": ocr_text if ocr_text else None,
                "analysis_method": "llm" if USE_MULTIMODAL_LLM and image_bytes else ("ocr" if ocr_text else "none"),
                "analysis_language": language, # Store language used
                "metadata": img_metadata
            })

    except Exception as e:
        print(f"Error during image analysis setup or loop: {e}")
        raise e
    finally:
        if doc: doc.close()

    print(f"Finished image analysis. Generated {len(image_descriptions)} descriptions.")
    return {"image_descriptions": image_descriptions}
