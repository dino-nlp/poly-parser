from typing import Dict, Any, List
from graph_definition import GraphState
import os
import fitz # To potentially extract chart images
import base64
import re # Import re for potential parsing if needed
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
# --- Configuration ---
USE_MULTIMODAL_LLM_FOR_CHARTS = True
DEFAULT_LANGUAGE = "English" # Fallback language

# Initialize Ollama for chart analysis (reuse image LLM if suitable)
llm_chart = None
if USE_MULTIMODAL_LLM_FOR_CHARTS:
    try:
        from langchain_ollama import ChatOllama
        llm_chart = ChatOllama(model=os.getenv("IMAGE_ANALYZER_MODEL"), temperature=0)
        print(f"Initialized Ollama for chart analysis ({os.getenv("IMAGE_ANALYZER_MODEL")}).")
    except ImportError:
        print("langchain_ollama not found. Cannot use LLM for chart analysis.")
        USE_MULTIMODAL_LLM_FOR_CHARTS = False
    except Exception as e:
        print(f"Failed to initialize Ollama for charts: {e}")
        USE_MULTIMODAL_LLM_FOR_CHARTS = False

def analyze_charts(state: GraphState) -> Dict[str, Any]:
    """
    Agent 4: Analyzes charts (often treated as images), providing summaries
             in the detected language.
    """
    print("Analyzing charts (as images)...")
    raw_elements = state.get("raw_elements", [])
    pdf_path = state["pdf_path"]
    # Get detected language from state, fallback to default
    language = state.get("language", DEFAULT_LANGUAGE)
    print(f"  Using language: {language}")

    chart_summaries = []
    doc = None

    # Identify potential charts (reusing image_refs for simplicity)
    image_refs = [el for el in raw_elements if el.get("type") == "image_ref"]
    if not image_refs:
        print("No image references found to analyze as potential charts.")
        return {}

    print(f"Found {len(image_refs)} potential charts (analyzing as images).")

    if not USE_MULTIMODAL_LLM_FOR_CHARTS or not llm_chart:
        print("Multi-modal LLM for charts is not configured. Skipping chart analysis.")
        return {}

    try:
        if image_refs: # Only open doc if there are images to process
            doc = fitz.open(pdf_path)

        for i, img_ref in enumerate(image_refs):
            img_metadata = img_ref.get("metadata", {})
            img_name = img_ref.get("content", f"image_{i}")
            xref = img_metadata.get("xref")
            print(f"  Analyzing potential chart {i+1}/{len(image_refs)}: {img_name} (xref: {xref})")

            summary = f"Chart/Image: {img_name}"
            image_bytes = None

            # --- Get Image Data ---
            if xref and doc:
                try:
                    base_image = doc.extract_image(xref)
                    if base_image: image_bytes = base_image["image"]
                    else: print(f"    Could not extract image for xref {xref}."); continue
                except Exception as e: print(f"    Error extracting image bytes for xref {xref}: {e}"); continue

            if not image_bytes: continue

            # --- Use Multi-modal LLM for Analysis ---
            print("    Attempting analysis with multi-modal LLM...")
            try:
                img_base64 = base64.b64encode(image_bytes).decode('utf-8')
                # Include language in the prompt
                prompt = (f"Analyze this image in {language}. Is it a chart or graph? "
                        f"If yes, identify the chart type (e.g., bar, line, pie). "
                        f"Describe the main data presented, key trends, or insights shown in the chart in {language}. "
                        f"Extract axis labels and the title if visible. "
                        f"If it's not a chart, briefly describe what it is in {language}."
                        "NO FURTHER EXPLANATION, JUST PROVIDE THE RESULT.")
                
                system_prompt = "You are an assistant tasked with describing table or image or chart"
                system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
                human_prompt = [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + "{img_base64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ]
                human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
                final_prompt = ChatPromptTemplate.from_messages(
                    [
                        system_message_template,
                        human_message_template
                    ]
                )
                parser_chain = final_prompt | llm_chart | StrOutputParser()
                llm_result = parser_chain.invoke(img_base64)
                summary = llm_result.strip()
                print(f"    LLM Chart Analysis Result: {summary[:150]}...")

            except Exception as e:
                print(f"    Multi-modal LLM chart analysis failed: {e}")
                summary = f"Chart/Image: {img_name} - Analysis failed."

            # --- Store Result ---
            chart_summaries.append({
                "chart_ref": img_name,
                "summary": summary,
                "analysis_method": "llm",
                "analysis_language": language, # Store language used
                "metadata": img_metadata
            })

    except Exception as e:
        print(f"Error during chart analysis setup or loop: {e}")
        raise e
    finally:
        if doc: doc.close()

    print(f"Finished chart analysis. Generated {len(chart_summaries)} summaries.")
    return {"chart_summaries": chart_summaries}

