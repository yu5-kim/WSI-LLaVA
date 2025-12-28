"""
WSI-Precision Stage 2: Claim-based Evaluation
This script evaluates model responses against ground truth claims.
"""

import json
import os
import pandas as pd
from openai import OpenAI


EVALUATION_PROMPT = """
Please act as an impartial judge and evaluate the correctness of the AI assistant's pathology dialogue for each claim based on the following scoring criteria. Provide an explanation for each evaluation and assign a score.

**Scoring Criteria:**
- **1**: The information in the pathology dialogue is completely correct regarding the claim.
- **0.7**: The information is mostly correct and closely aligns with the claim.
- **0.3**: The claim is mentioned but contains errors in the core content (e.g., mistakes in differentiation degree or malignancy).
- **0**: The information in the pathology dialogue is completely incorrect regarding the claim.

Output Requirements:

Please output your evaluations as a list of dictionaries in plain text format (not JSON). The format should be as follows:

[
    {
        "claim": "Original claim1",
        "explanation": "Explanation for the score",
        "score": 1 or 0.7 or 0.3 or 0
    },
    {
        "claim": "Original claim2",
        "explanation": "Explanation for the score",
        "score": 1 or 0.7 or 0.3 or 0
    },
    ...
]
"""


def initialize_client(api_key, base_url):
    """Initialize OpenAI client with custom configuration."""
    return OpenAI(api_key=api_key, base_url=base_url)


def evaluate_claims(client, prompt, content, model="gpt-4"):
    """Evaluate claims using the specified model."""
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


def load_ground_truth(gt_file_path):
    """Load ground truth claims from JSON file."""
    if not os.path.exists(gt_file_path):
        print(f"Ground truth file not found: {gt_file_path}")
        return None
    
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_or_initialize_eval_data(eval_file_path):
    """Load existing evaluation data or initialize empty dictionary."""
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        print(f"Loaded existing evaluation data from {eval_file_path}")
        return eval_data
    else:
        print(f"Initialized empty evaluation data")
        return {}


def process_evaluation(model_responses_csv, ground_truth_json, output_eval_json, 
                       client, model="gpt-4"):
    """
    Evaluate model responses against ground truth claims.
    
    Args:
        model_responses_csv: CSV file containing model responses
        ground_truth_json: JSON file containing ground truth claims
        output_eval_json: Output JSON file for evaluation results
        client: OpenAI client instance
        model: Model name to use for evaluation
    """
    # Load ground truth
    gt_data = load_ground_truth(ground_truth_json)
    if gt_data is None:
        return

    # Load or initialize evaluation data
    eval_data = load_or_initialize_eval_data(output_eval_json)

    # Load model responses
    test_df = pd.read_csv(model_responses_csv)

    # Process each test entry
    for index, row in test_df.iterrows():
        question_id = row.get("question_id")
        metadata = row.get("metadata", "")

        if pd.isna(question_id):
            continue

        # Skip if already processed with valid results
        if question_id in eval_data and eval_data[question_id].get("claim_results", None) not in [None, []]:
            continue

        print(f"Processing {question_id}")

        # Extract model response and metadata
        model_response = row.get("model_output", "No text available")
        response_type = metadata if pd.notna(metadata) else "default_type"

        eval_data[question_id] = {
            "model_answer": model_response,
            "type": response_type
        }

        # Get ground truth claims
        if question_id in gt_data and "claims" in gt_data[question_id]:
            claims = gt_data[question_id]["claims"]

            # Construct evaluation context
            context = f"claims:{claims} \n pathology dialogue responses:{model_response}"

            # Evaluate claims
            claim_results_str = evaluate_claims(client, EVALUATION_PROMPT, context, model)
            
            try:
                claim_results = json.loads(claim_results_str)
            except json.JSONDecodeError:
                claim_results = []

            # Create output entry
            output_entry = {
                "type": response_type,
                "id": question_id,
                "model_response": model_response,
                "claim_results": claim_results
            }

            eval_data[question_id] = output_entry

            # Save after each entry
            with open(output_eval_json, 'w', encoding='utf-8') as f:
                json.dump(eval_data, f, ensure_ascii=False, indent=4)

            print(f"Completed processing {question_id}")
        else:
            print(f"Warning: No ground truth found for {question_id}")

    print("Evaluation completed successfully.")


def main():
    """Main execution function."""
    # Configuration
    API_KEY = "your-api-key-here"
    BASE_URL = "https://api.openai.com/v1"
    MODEL = "gpt-5-mini"
    
    # File paths
    MODEL_RESPONSES_CSV = "model_responses.csv"
    GROUND_TRUTH_JSON = "extracted_claims.json"
    OUTPUT_EVAL_JSON = "evaluation_results.json"
    
    # Initialize client
    client = initialize_client(API_KEY, BASE_URL)
    
    # Process evaluation
    process_evaluation(MODEL_RESPONSES_CSV, GROUND_TRUTH_JSON, 
                      OUTPUT_EVAL_JSON, client, MODEL)


if __name__ == "__main__":
    main()