from typing import Dict, Any, List
from graph_definition import GraphState
import os
import fitz # To potentially extract chart images

# --- Configuration ---
# Chart analysis is complex. Using a multi-modal LLM is the most promising approach.
USE_MULTIMODAL_LLM_FOR_CHARTS = True

# Initialize Ollama for chart analysis (reuse image LLM if suitable)
llm_chart = None
if USE_MULTIMODAL_LLM_FOR_CHARTS:
    try:
        from langchain_community.llms import Ollama
        # Use the same multi-modal model or a different one if specialized
        llm_chart = Ollama(model="llava", base_url=os.getenv("OLLAMA_BASE_URL"))
        print("Initialized Ollama for chart analysis (llava).")
    except ImportError:
        print("langchain_community.llms.Ollama not found. Cannot use LLM for chart analysis.")
        USE_MULTIMODAL_LLM_FOR_CHARTS = False
    except Exception as e:
        print(f"Failed to initialize Ollama for charts: {e}")
        USE_MULTIMODAL_LLM_FOR_CHARTS = False

def analyze_charts(state: GraphState) -> Dict[str, Any]:
    """
    Agent 4: Analyzes charts (often treated as images).
    Attempts to interpret the chart type, data, and trends using an LLM.

    Args:
        state: The current graph state.

    Returns:
        A dictionary with the updated 'chart_summaries'.
    """
    print("Analyzing charts (as images)...")
    raw_elements = state.get("raw_elements", [])
    pdf_path = state["pdf_path"]
    chart_summaries = []
    doc = None

    # Identify potential charts. This is heuristic. We might assume any 'image'
    # could be a chart, or look for specific keywords in nearby text (complex).
    # For simplicity, let's re-use the image_refs identified earlier, assuming
    # charts are saved/represented as images.
    # A better approach might involve layout analysis first.
    image_refs = [el for el in raw_elements if el.get("type") == "image_ref"]
    if not image_refs:
        print("No image references found to analyze as potential charts.")
        return {}

    print(f"Found {len(image_refs)} potential charts (analyzing as images).")

    if not USE_MULTIMODAL_LLM_FOR_CHARTS or not llm_chart:
        print("Multi-modal LLM for charts is not configured. Skipping chart analysis.")
        return {}

    try:
        doc = fitz.open(pdf_path) # Open PDF to extract image bytes

        for i, img_ref in enumerate(image_refs):
            img_metadata = img_ref.get("metadata", {})
            img_name = img_ref.get("content", f"image_{i}")
            xref = img_metadata.get("xref")
            print(f"  Analyzing potential chart {i+1}/{len(image_refs)}: {img_name} (xref: {xref})")

            summary = f"Chart/Image: {img_name}" # Default
            image_bytes = None

            # --- Get Image Data ---
            if xref and doc:
                try:
                    base_image = doc.extract_image(xref)
                    if base_image:
                        image_bytes = base_image["image"]
                    else:
                        print(f"    Could not extract image for xref {xref}.")
                        continue # Skip if no image data
                except Exception as e:
                    print(f"    Error extracting image bytes for xref {xref}: {e}")
                    continue # Skip if error

            if not image_bytes:
                continue # Skip if no image data

            # --- Use Multi-modal LLM for Analysis ---
            print("    Attempting analysis with multi-modal LLM...")
            try:
                import base64
                img_base64 = base64.b64encode(image_bytes).decode('utf-8')

                # Specific prompt for chart analysis
                prompt = ("Analyze this image. Is it a chart or graph? "
                          "If yes, identify the chart type (e.g., bar, line, pie). "
                          "Describe the main data presented, key trends, or insights shown in the chart. "
                          "Extract axis labels and the title if visible. "
                          "If it's not a chart, briefly describe what it is.")

                from langchain_core.messages import HumanMessage
                msg = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                     ]
                )
                llm_result = llm_chart.invoke(msg)
                summary = llm_result.strip()
                print(f"    LLM Chart Analysis Result: {summary[:150]}...")

            except Exception as e:
                print(f"    Multi-modal LLM chart analysis failed: {e}")
                summary = f"Chart/Image: {img_name} - Analysis failed."


            # --- Store Result ---
            chart_summaries.append({
                "chart_ref": img_name, # Reference back to the image element
                "summary": summary,
                "analysis_method": "llm",
                "metadata": img_metadata
            })

    except Exception as e:
        print(f"Error during chart analysis setup or loop: {e}")
        raise e # Propagate error
    finally:
        if doc:
            doc.close()

    print(f"Finished chart analysis. Generated {len(chart_summaries)} summaries.")
    return {"chart_summaries": chart_summaries}
