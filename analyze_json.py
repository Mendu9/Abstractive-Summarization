import os
import json

def analyze_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print("Analyzing file:", file_path)
    print("Top-level keys:", list(data.keys()))
    
    for key, value in data.items():
        print(f"\nKey: '{key}' -> Type: {type(value).__name__}")
        if isinstance(value, dict):
            print("  Nested keys:", list(value.keys()))
        elif isinstance(value, list):
            if len(value) > 0:
                print("  Type of first element in list:", type(value[0]).__name__)
                if isinstance(value[0], dict):
                    print("  Keys in first list element:", list(value[0].keys()))
                else:
                    # If it's not a dict, show a sample value.
                    print("  Sample value:", value[0])
            else:
                print("  (Empty list)")
    print("\n" + "="*60 + "\n")

def main():
    # Define the offline dataset directory.
    offline_dir = os.path.join("data", "gov-report")
    # Folders for CRS and GAO files.
    crs_dir = os.path.join(offline_dir, "crs")
    gao_dir = os.path.join(offline_dir, "gao")
    
    # Get one JSON file from CRS.
    crs_files = [os.path.join(crs_dir, f) for f in os.listdir(crs_dir) if f.endswith(".json")]
    # Get one JSON file from GAO.
    gao_files = [os.path.join(gao_dir, f) for f in os.listdir(gao_dir) if f.endswith(".json")]
    
    if crs_files:
        print("CRS sample:")
        analyze_file(crs_files[0])
    else:
        print("No JSON files found in the CRS directory.")
    
    if gao_files:
        print("GAO sample:")
        analyze_file(gao_files[0])
    else:
        print("No JSON files found in the GAO directory.")

if __name__ == "__main__":
    main()
