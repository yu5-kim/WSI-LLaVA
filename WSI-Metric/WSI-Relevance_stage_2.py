"""
WSI-Relevance Stage 2: Relevance Evaluation
This script evaluates the relevance of ground truth answers to model-generated claims.
"""

import json
import os
from openai import OpenAI


EVALUATION_PROMPT = """
Please act as an impartial judge and evaluate the relevance of the original ground truth answer to each claim derived from the model's answer. Provide an explanation for each evaluation and assign a score based on the following criteria.

Scoring Criteria:
- **1**: The content in the ground truth answer is completely relevant to the claim.
- **0.7**: The content is mostly relevant but has minor omissions or deviations.
- **0.3**: The content is partially relevant with significant omissions or irrelevant information.
- **0**: The content in the ground truth answer is not relevant to the claim.

Evaluation Guidelines:
    • Focus on Relevance: Assess how well the content in the ground truth answer pertains to the claim derived from the model's answer.
    • Use Partial Credit: Utilize intermediate scores (0.7 and 0.3) to reflect varying degrees of relevance.
    • Ignore Unmentioned Claims: Do not evaluate claims that are not addressed in the ground truth answer.

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


def evaluate_relevance(client, prompt, content, model="gpt-4"):
    """Evaluate relevance using the specified model."""
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


def load_json_file(file_path):
    """Load data from JSON file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
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


def process_relevance_evaluation(ground_truth_json, model_claims_json,
                                  output_eval_json, client, model="gpt-4", 
                                  metadata_filter=None):
    """
    Evaluate relevance of ground truth to model-generated claims.
    
    Args:
        ground_truth_json: JSON file containing ground truth text
        model_claims_json: JSON file containing extracted claims from model responses
        output_eval_json: Output JSON file for evaluation results
        client: OpenAI client instance
        model: Model name to use for evaluation
        metadata_filter: Metadata value to filter entries (e.g., "Report")
    """
    # Load model claims
    model_claims_data = load_json_file(model_claims_json)
    if model_claims_data is None:
        return

    # Load ground truth data
    ground_truth_data = load_json_file(ground_truth_json)
    if ground_truth_data is None:
        return

    # Load or initialize evaluation data
    eval_data = load_or_initialize_eval_data(output_eval_json)

    # Process each entry
    for question_id, gt_entry in ground_truth_data.items():
        metadata = gt_entry.get("metadata")
        ground_truth_text = gt_entry.get("text", "No text available")

        # Apply metadata filter if specified
        if metadata_filter and metadata != metadata_filter:
            continue

        # Skip if already processed
        if question_id in eval_data:
            continue

        print(f"Processing {question_id}")

        # Create basic entry
        entry_data = {
            "type": metadata,
            "id": question_id,
            "ground_truth_answer": ground_truth_text,
        }

        # Check if model claims exist for this question
        if question_id in model_claims_data and "claims" in model_claims_data[question_id]:
            claims = model_claims_data[question_id]["claims"]

            # Construct evaluation context
            context = f"claims:{claims} \n ground truth answer:{ground_truth_text}"

            # Evaluate relevance
            try:
                claim_results_str = evaluate_relevance(client, EVALUATION_PROMPT, context, model)
                claim_results = json.loads(claim_results_str)
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error evaluating {question_id}: {e}")
                claim_results = []

            entry_data["claim_results"] = claim_results
        else:
            entry_data["claim_results"] = []

        # Add to results
        eval_data[question_id] = entry_data

        # Save after each entry
        with open(output_eval_json, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=4)
        
        print(f"Completed processing {question_id}")

    print("All data processed successfully.")


def main():
    """Main execution function."""
    # Configuration
    API_KEY = "your-api-key-here"
    BASE_URL = "https://api.openai.com/v1"
    MODEL = "gpt-5-mini"
    
    # File paths
    GROUND_TRUTH_JSON = "ground_truth_data.json"
    MODEL_CLAIMS_JSON = "extracted_model_claims.json"
    OUTPUT_EVAL_JSON = "relevance_evaluation_results.json"
    
    # Optional metadata filter
    METADATA_FILTER = "Report"  # Set to None to process all entries
    
    # Initialize client
    client = initialize_client(API_KEY, BASE_URL)
    
    # Process evaluation
    process_relevance_evaluation(
        GROUND_TRUTH_JSON,
        MODEL_CLAIMS_JSON,
        OUTPUT_EVAL_JSON,
        client,
        MODEL,
        METADATA_FILTER
    )


if __name__ == "__main__":
    main()