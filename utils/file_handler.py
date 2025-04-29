import json
import os

def save_json_output(data: Any, output_path: str):
    """
    Saves the given data structure as a JSON file.

    Args:
        data: The data to save (should be JSON serializable).
        output_path: The path where the JSON file will be saved.
    """
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        with open(output_path, 'w', encoding='utf-8') as f:
            # Use ensure_ascii=False for proper UTF-8 encoding of various characters
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved output to {output_path}")

    except TypeError as e:
        print(f"Error: Data is not JSON serializable: {e}")
        # Try saving with a fallback string conversion (less ideal)
        try:
            print("Attempting to save with string conversion for non-serializable parts...")
            with open(output_path.replace(".json", "_partial_str.json"), 'w', encoding='utf-8') as f:
                json.dump(_force_serializable(data), f, indent=2, ensure_ascii=False)
            print(f"Partially saved output with string conversions to {output_path.replace('.json', '_partial_str.json')}")
        except Exception as final_e:
            print(f"Error: Could not save JSON output even with conversion: {final_e}")

    except IOError as e:
        print(f"Error: Could not write to file {output_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")


def _force_serializable(obj: Any) -> Any:
    """Recursively converts non-serializable items to strings."""
    if isinstance(obj, dict):
        return {k: _force_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_force_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # Convert anything else to its string representation
        return str(obj)

# Example function to load data (if needed for testing)
def load_json_data(file_path: str) -> Any:
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {file_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        return None
