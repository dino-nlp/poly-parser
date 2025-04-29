from typing import Dict, Any
from graph_definition import GraphState
# Use a lightweight library like langdetect or gcld3
# pip install langdetect
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Ensure reproducibility for langdetect
DetectorFactory.seed = 0

def detect_language(state: GraphState) -> Dict[str, Any]:
    """
    Agent 1.5: Detects the language of the extracted text elements (optional).
    This version detects the dominant language of the first few text blocks.
    A more robust version might detect per-block or per-page language.

    Args:
        state: The current graph state.

    Returns:
        A dictionary with the updated 'language' field in the state.
    """
    print("Detecting language...")
    raw_elements = state.get("raw_elements", [])
    if not raw_elements:
        print("No raw elements found to detect language.")
        return {} # No changes if no elements

    # Combine text from the first few text blocks for detection
    text_sample = ""
    text_count = 0
    for element in raw_elements:
        if element.get("type") == "text" and element.get("content"):
            text_sample += element["content"] + "\n"
            text_count += 1
            if text_count >= 5 or len(text_sample) > 1000: # Limit sample size
                break

    if not text_sample.strip():
        print("No text content found to detect language.")
        return {}

    detected_language = None
    try:
        detected_language = detect(text_sample)
        print(f"Detected language: {detected_language}")
    except LangDetectException:
        print("Could not detect language with certainty.")
        detected_language = "unknown"
    except Exception as e:
        print(f"Error during language detection: {e}")
        # Don't stop the pipeline, just mark as unknown
        detected_language = "error"

    # Update the main state with the detected language
    return {"language": detected_language}

# --- Alternative: Per-element language detection ---
# def detect_language_per_element(state: GraphState) -> Dict[str, Any]:
#     """Detects language for each text element."""
#     print("Detecting language per text element...")
#     raw_elements = state.get("raw_elements", [])
#     updated_elements = []
#     detected_languages = set()
#
#     for element in raw_elements:
#         if element.get("type") == "text" and element.get("content"):
#             try:
#                 lang = detect(element["content"][:500]) # Detect on first 500 chars
#                 element["metadata"]["language"] = lang
#                 detected_languages.add(lang)
#             except LangDetectException:
#                 element["metadata"]["language"] = "unknown"
#             except Exception as e:
#                  print(f"Error detecting language for element: {e}")
#                  element["metadata"]["language"] = "error"
#         updated_elements.append(element)
#
#     # Determine overall document language (e.g., most frequent)
#     # This logic can be more sophisticated
#     main_language = "multi" if len(detected_languages) > 1 else (list(detected_languages)[0] if detected_languages else "unknown")
#     print(f"Detected languages: {detected_languages}. Main language set to: {main_language}")
#
#     return {"raw_elements": updated_elements, "language": main_language}
