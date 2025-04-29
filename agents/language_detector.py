from typing import Dict, Any, Optional
from graph_definition import GraphState
# Use a lightweight library like langdetect or gcld3
# pip install langdetect pycountry
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import pycountry # Import pycountry

# Ensure reproducibility for langdetect
DetectorFactory.seed = 0

def get_language_name(lang_code: str) -> Optional[str]:
    """Converts a 2-letter language code (ISO 639-1) to its English name."""
    if not lang_code or len(lang_code) != 2:
        return None
    try:
        # Get language object from pycountry using the alpha_2 code
        language = pycountry.languages.get(alpha_2=lang_code)
        return language.name if language else None
    except Exception as e:
        print(f"Could not find language name for code '{lang_code}': {e}")
        return None

def detect_language(state: GraphState) -> Dict[str, Any]:
    """
    Agent 1.5: Detects the language of the extracted text elements
    and stores the full language name (e.g., 'Vietnamese').

    Args:
        state: The current graph state.

    Returns:
        A dictionary with the updated 'language' field (full name) in the state.
    """
    print("Detecting language...")
    raw_elements = state.get("raw_elements", [])
    detected_language_name = "English" # Default to English

    if not raw_elements:
        print("No raw elements found to detect language. Defaulting to English.")
        return {"language": detected_language_name}

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
        print("No text content found to detect language. Defaulting to English.")
        return {"language": detected_language_name}

    detected_language_code = None
    try:
        detected_language_code = detect(text_sample)
        print(f"Detected language code: {detected_language_code}")
        # Convert code to full name
        lang_name = get_language_name(detected_language_code)
        if lang_name:
            detected_language_name = lang_name
            print(f"Converted to language name: {detected_language_name}")
        else:
            print(f"Could not convert code '{detected_language_code}' to name. Using code as name.")
            detected_language_name = detected_language_code # Use code if name not found
    except LangDetectException:
        print("Could not detect language with certainty. Defaulting to English.")
        # Keep default English
    except Exception as e:
        print(f"Error during language detection: {e}. Defaulting to English.")
        # Keep default English

    # Update the main state with the detected language name
    return {"language": detected_language_name}

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
