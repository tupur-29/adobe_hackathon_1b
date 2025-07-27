import os
import json
import sys
from src.analyzer_1b import PersonaBasedPDFAnalyzer


INPUT_DIR = '/app/input'
OUTPUT_DIR = '/app/output'
MODEL_DIR = '/app/models'

def run_analysis():
    """
    Main execution function. Finds input, runs the analyzer, and saves the output.
    """
    print("🚀 Starting Persona-Driven Document Intelligence Process...")

    
    input_json_path = os.path.join(INPUT_DIR, 'challenge1b_input.json')
    if not os.path.exists(input_json_path):
        print(f"❌ ERROR: Input file 'challenge1b_input.json' not found in {INPUT_DIR}. Exiting.")
        sys.exit(1)

    
    try:
        model_file = next(f for f in os.listdir(MODEL_DIR) if f.endswith('.txt'))
        r1a_model_path = os.path.join(MODEL_DIR, model_file)
        print(f"✅ Found 1A Model: {r1a_model_path}")
    except StopIteration:
        print(f"❌ ERROR: No .txt model file found in {MODEL_DIR}. Exiting.")
        sys.exit(1)

    
    try:
        
        analyzer = PersonaBasedPDFAnalyzer(r1a_model_path=r1a_model_path)
        
        
        final_output = analyzer.analyze_documents(input_json_path=input_json_path)

    except Exception as e:
        print(f"🔥 An unexpected error occurred during analysis: {e}")
        
        final_output = {
            "error": "An exception occurred during processing.",
            "details": str(e)
        }
        
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'challenge1b_final_output.json')
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*50)
        print("🏁 Analysis complete!")
        print(f"💾 Final output saved to: {output_path}")
        print("="*50)

    except Exception as e:
        print(f"❌ ERROR: Could not write output file to {output_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_analysis()
    