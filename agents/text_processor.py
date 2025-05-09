import re
import os
from typing import Dict, Any, List
from graph_definition import GraphState
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Optional: NLTK for sentence splitting, SpaCy for NER
# import nltk
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt')
# from nltk.tokenize import sent_tokenize

# import spacy
# # Download a model: python -m spacy download en_core_web_sm
# try:
#     nlp_ner = spacy.load("en_core_web_sm") # Or other language model
# except OSError:
#     print("Spacy NER model not found. Download it: python -m spacy download en_core_web_sm")
#     nlp_ner = None

# --- Configuration ---
USE_LLM_FOR_CLEANING = True
PERFORM_NER = False
USE_LLM_FOR_NER_ACRONYMS = True and USE_LLM_FOR_CLEANING
DEFAULT_LANGUAGE = "English" # Fallback language

# Initialize Ollama LLM
llm = ChatOllama(model=os.getenv("TEXT_PROCESSOR_MODEL"), temperature=0)
print(f"LLM model for text processing: {os.getenv("TABLE_ANALYZER_MODEL")}")

# --- Basic Cleaning Functions ---
def basic_text_cleaning(text: str) -> str:
    # ... (cleaning logic remains the same)
    # Remove excessive newlines and whitespace
    text = re.sub(r'\n{3,}', '\n\n', text) # Replace 3+ newlines with 2
    text = re.sub(r' +', ' ', text) # Replace multiple spaces with single space
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE) # Remove leading/trailing whitespace per line

    # Simple header/footer removal (example - needs customization)
    lines = text.split('\n')
    # Remove lines that look like page numbers (e.g., just digits, or "Page X of Y")
    lines = [line for line in lines if not re.fullmatch(r'\s*\d+\s*', line)]
    lines = [line for line in lines if not re.match(r'\s*Page \d+( of \d+)?\s*$', line, re.IGNORECASE)]
    # Add more rules based on common header/footer patterns in your documents

    # Re-join lines, trying to reconnect hyphenated words at line breaks
    cleaned_text = ""
    for i, line in enumerate(lines):
        if line.endswith('-') and i + 1 < len(lines):
            # Check if next line starts with lowercase (likely continuation)
            if re.match(r'^[a-z]', lines[i+1]):
                 cleaned_text += line[:-1] # Add line without hyphen
            else:
                 cleaned_text += line + "\n" # Keep hyphen, add newline
        else:
            cleaned_text += line + "\n"

    # Final whitespace cleanup
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()

    return cleaned_text


# --- LLM Chains (Now language aware) ---

# Cleaning Prompt
cleaning_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert text processing assistant. Your task is to clean and reformat the provided text extracted from a PDF. Focus on creating well-structured paragraphs and sentences in {language}. Remove redundant whitespace, correct broken sentences, and eliminate artifacts like page numbers or simple headers/footers if they appear within the main text flow. Do NOT remove meaningful content. Preserve the original meaning and structure. If the text contains lists or code blocks, try to format them appropriately using markdown. Respond ONLY with the cleaned text."),
    ("user", "Please clean and reformat the following text:\n\n---\n{text_chunk}\n---")
])
cleaning_chain = cleaning_prompt_template | llm | StrOutputParser()

# NER/Acronym Prompt
ner_acronym_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert linguistic analyst. Analyze the provided text chunk ({language}). Identify key Named Entities (like Person, Organization, Location, Date, Product) and any Acronyms used. For acronyms, provide their likely full form if discernible from the context or common knowledge. Present the results clearly in {language}. If no entities or acronyms are found, state that clearly in {language}.\n\nRespond ONLY in the following format:\nNamed Entities:\n[List entities here, e.g., PERSON: John Doe, ORG: Acme Corp]\n\nAcronyms:\n[List acronyms here, e.g., NLP: Natural Language Processing]"),
    ("user", "Analyze the following text for Named Entities and Acronyms:\n\n---\n{cleaned_text}\n---")
])
ner_acronym_chain = ner_acronym_prompt_template | llm | StrOutputParser()


# --- Main Agent Function ---
def process_text(state: GraphState) -> Dict[str, Any]:
    """
    Agent 2: Cleans, enhances, and potentially extracts entities from text elements,
             considering the detected language.
    """
    print("Processing text elements...")
    raw_elements = state.get("raw_elements", [])
    # Get detected language from state, fallback to default
    language = state.get("language", DEFAULT_LANGUAGE)
    print(f"  Using language: {language}")

    processed_chunks = []
    text_to_process = []
    # --- Consolidate text blocks (same as before) ---
    current_text_block = ""
    current_metadata = {}
    for element in raw_elements:
        if element.get("type") == "text" and element.get("content"):
            if not current_text_block:
                current_metadata = element.get("metadata", {})
            current_text_block += element["content"] + "\n\n"
        else:
            if current_text_block:
                text_to_process.append({"content": current_text_block.strip(), "metadata": current_metadata})
                current_text_block = ""
                current_metadata = {}
    if current_text_block:
        text_to_process.append({"content": current_text_block.strip(), "metadata": current_metadata})

    print(f"Consolidated into {len(text_to_process)} text blocks for processing.")

    for i, text_block in enumerate(text_to_process):
        print(f"  Processing block {i+1}/{len(text_to_process)}...")
        original_text = text_block["content"]
        metadata = text_block["metadata"]
        cleaned_text = ""
        entities = {}
        acronyms = {}

        if USE_LLM_FOR_CLEANING:
            print("    Cleaning with LLM...")
            try:
                # Pass language to the chain
                cleaned_text = cleaning_chain.invoke({
                    "text_chunk": original_text,
                    "language": language
                })
            except Exception as e:
                print(f"    LLM cleaning failed: {e}. Falling back to basic cleaning.")
                cleaned_text = basic_text_cleaning(original_text)
        else:
            print("    Cleaning with basic rules...")
            cleaned_text = basic_text_cleaning(original_text)

        # --- Optional: NER and Acronym Handling ---
        if PERFORM_NER and nlp_ner:
            # ... (SpaCy NER logic - might need language-specific model)
            print("    Performing NER with spaCy...")
            # Consider loading a language-specific spaCy model based on 'language' if available
            # doc = nlp_ner(cleaned_text)
            # entities = ...
            # acronyms = ...
            pass # Placeholder

        elif USE_LLM_FOR_NER_ACRONYMS:
            print("    Performing NER/Acronym detection with LLM...")
            try:
                # Pass language and cleaned text to the chain
                analysis_result = ner_acronym_chain.invoke({
                    "cleaned_text": cleaned_text,
                    "language": language
                })
                # --- Parse LLM output (same logic as before) ---
                entities_str = re.search(r"Named Entities:\n(.*?)\n\nAcronyms:", analysis_result, re.DOTALL)
                acronyms_str = re.search(r"Acronyms:\n(.*)", analysis_result, re.DOTALL)
                if entities_str:
                    entities_list = entities_str.group(1).strip().split('\n')
                    entities = {}
                    for item in entities_list:
                        if ':' in item:
                            etype, evalue = item.split(':', 1)
                            if etype.strip() not in entities:
                                entities[etype.strip()] = []
                            entities[etype.strip()].append(evalue.strip())
                if acronyms_str:
                     acronyms_list = acronyms_str.group(1).strip().split('\n')
                     acronyms = {}
                     for item in acronyms_list:
                         if ':' in item:
                             acr, full = item.split(':', 1)
                             acronyms[acr.strip()] = full.strip()
                         else:
                             if "detected" not in acronyms: acronyms["detected"] = []
                             acronyms["detected"].append(item.strip())
                print(f"    LLM analysis found: {len(entities)} entity types, {len(acronyms)} acronyms.")
            except Exception as e:
                print(f"    LLM NER/Acronym analysis failed: {e}")

        # --- Store processed chunk ---
        processed_chunks.append({
            "text": cleaned_text,
            "metadata": {
                **metadata,
                "cleaned_with": "llm" if USE_LLM_FOR_CLEANING else "basic",
                "entities": entities if entities else None,
                "acronyms": acronyms if acronyms else None,
                "processed_language": language # Add language used for processing
            }
        })

    print(f"Finished processing text. Generated {len(processed_chunks)} processed chunks.")
    return {"processed_text_chunks": processed_chunks}
