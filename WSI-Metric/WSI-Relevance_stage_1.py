"""
WSI-Relevance Stage 1: Claims Extraction from Model Outputs
This script extracts atomic claims from model-generated pathological diagnosis texts.
"""

import json
import os
import pandas as pd
from openai import OpenAI


PROMPT_TEMPLATE = """
You are an AI assistant specialized in processing pathological diagnosis texts. I will provide you with a pathology diagnosis text.

Your task is to:

Claims Extraction:
    • Divide the text into several distinct and granular claims.
    • Keep closely related information together in the same claim to preserve context and meaning. Do not split sentences or ideas that are logically connected.
    • Break down complex sentences into smaller, individual claims only if it does not disrupt the logical flow or separate connected ideas.
    • Ensure there is no omission or repetition among the claims.

Guidelines:
    • Only output the claims without including any additional text or explanations.
    • Each claim should be concise and represent a single fact or point.
    • Maintain the integrity of statements that are contextually connected.

Output Format: Present the extracted claims as a list in the following format:
["claim1", "claim2", "claim3", ...]
"""


def initialize_client(api_key, base_url):
    """Initialize OpenAI client with custom configuration."""
    return OpenAI(api_key=api_key, base_url=base_url)


def extract_claims(client, prompt, content, model="gpt-4"):
    """Extract claims from content using the specified model."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': prompt},
                {"role": "user", "content": content}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing request: {e}")
        return None


def process_model_outputs(input_csv, output_file, client, model="gpt-4", 
                          metadata_filter=None):
    """
    Process model output CSV file and extract claims.
    
    Args:
        input_csv: Path to input CSV file containing model outputs
        output_file: Path to output JSON file for extracted claims
        client: OpenAI client instance
        model: Model name to use for extraction
        metadata_filter: Metadata value to filter entries (e.g., "Report")
    """
    # Load input data from CSV
    df = pd.read_csv(input_csv)
    
    if df.empty:
        print(f"No data loaded from {input_csv}")
        return

    # Load or initialize processed claims
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            processed_claims = json.load(f)
        print(f"Loaded existing claims from {output_file}")
    else:
        processed_claims = {}
        print(f"Initialized empty claims dictionary")

    # Process each row
    for index, row in df.iterrows():
        question_id = row.get("question_id")
        model_output = row.get("model_output")
        metadata = row.get("metadata")

        if pd.isna(question_id) or pd.isna(model_output):
            continue

        # Apply metadata filter if specified
        if metadata_filter and metadata != metadata_filter:
            continue

        # Skip if already processed with valid claims
        if question_id in processed_claims and processed_claims[question_id]["claims"] is not None:
            print(f"{question_id}: Already processed, skipping")
            continue

        # Process entry
        print(f"Processing {question_id}")

        # Extract claims using model
        claims_response = extract_claims(client, PROMPT_TEMPLATE, model_output, model)

        if not claims_response:
            processed_claims[question_id] = {
                "metadata": metadata,
                "text": model_output,
                "claims": None
            }
            continue

        # Parse and validate claims
        try:
            claims_list = json.loads(claims_response)
            if not isinstance(claims_list, list):
                raise ValueError("Extracted claims are not in list format")
            processed_claims[question_id] = {
                "metadata": metadata,
                "text": model_output,
                "claims": claims_list
            }
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error processing {question_id}: {e}")
            processed_claims[question_id] = {
                "metadata": metadata,
                "text": model_output,
                "claims": None
            }

        # Save after each entry
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_claims, f, ensure_ascii=False, indent=4)
        print(f"Completed processing {question_id}")

    print("All data processed successfully.")


def main():
    """Main execution function."""
    # Configuration
    API_KEY = "your-api-key-here"
    BASE_URL = "https://api.openai.com/v1"
    MODEL = "gpt-5-mini"
    
    # File paths
    INPUT_CSV = "model_outputs.csv"
    OUTPUT_FILE = "extracted_model_claims.json"
    
    # Optional metadata filter
    METADATA_FILTER = "Report"  # Set to None to process all entries
    
    # Initialize client
    client = initialize_client(API_KEY, BASE_URL)
    
    # Process file
    process_model_outputs(INPUT_CSV, OUTPUT_FILE, client, MODEL, METADATA_FILTER)


if __name__ == "__main__":
    main()