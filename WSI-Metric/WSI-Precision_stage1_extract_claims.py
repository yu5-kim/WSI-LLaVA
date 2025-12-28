"""
WSI-Precision Stage 1: Claims Extraction
This script extracts atomic claims from pathological diagnosis Q&A pairs.
"""

import json
import os
from openai import OpenAI


PROMPT_TEMPLATE = """
You are an AI assistant specialized in processing pathological diagnosis Q&A pairs. I will provide you with a pathology diagnosis question and its corresponding answer. 

Your task is to:
Claims Extraction:
    • Carefully analyze the answer and remove any unnecessary information that is not directly relevant to the question.
    • Only extract claims that directly address the question. Discard any information that does not directly answer or pertain to the question.
    • Divide the refined answer into several distinct and granular claims.
    • Keep closely related information together in the same claim to preserve context and meaning. Do not split sentences or ideas that are logically connected.
    • Break down complex sentences into smaller, individual claims only if it does not disrupt the logical flow or separate connected ideas.
    • Ensure there is no omission or repetition among the claims.

Guidelines:
    • Only output the claims without including any additional text or explanations.
    • Each claim should be concise and represent a single fact or point directly related to the question.
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


def process_json_file(input_file, output_file, client, model="gpt-4"):
    """
    Process a single JSON file containing pathology Q&A pairs.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file for extracted claims
        client: OpenAI client instance
        model: Model name to use for extraction
    """
    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"Warning: {input_file} is not a list format")
                return
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError in {input_file}: {e}")
            return

    # Load or initialize processed claims
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            processed_claims = json.load(f)
        print(f"Loaded existing claims from {output_file}")
    else:
        processed_claims = {}
        print(f"Initialized empty claims dictionary")

    # Process each entry
    for entry in data:
        question_id = entry.get("id")
        metadata = entry.get("metadata")

        if metadata is None:
            continue

        # Extract GPT response from conversations
        conversations = entry.get("conversations", [])
        text = None
        for conv in conversations:
            if conv.get("from") == "gpt":
                text = conv.get("value")
                break

        if text is None:
            print(f"{question_id}: No GPT response found, skipping")
            continue

        # Skip if already processed with valid claims
        if (question_id in processed_claims and
                processed_claims[question_id]["claims"] is not None and
                len(processed_claims[question_id]["claims"]) > 0):
            print(f"{question_id}: Already processed with valid claims, skipping")
            continue

        # Process entry
        print(f"Processing {question_id}")
        print(f"Text preview: {text[:100]}...")

        # Extract claims using model
        claims_response = extract_claims(client, PROMPT_TEMPLATE, text, model)

        if not claims_response:
            processed_claims[question_id] = {
                "metadata": metadata,
                "text": text,
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
                "text": text,
                "claims": claims_list
            }
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error processing {question_id}: {e}")
            processed_claims[question_id] = {
                "metadata": metadata,
                "text": text,
                "claims": None
            }

        # Save after each entry
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_claims, file=f, ensure_ascii=False, indent=4)
        print(f"Completed processing {question_id}")

    print("All data processed successfully.")


def main():
    """Main execution function."""
    # Configuration
    API_KEY = "your-api-key-here"
    BASE_URL = "https://api.openai.com/v1"
    MODEL = "gpt-5-mini"
    
    # File paths
    INPUT_FILE = "input_qa_pairs.json"
    OUTPUT_FILE = "extracted_claims.json"
    
    # Initialize client
    client = initialize_client(API_KEY, BASE_URL)
    
    # Process file
    process_json_file(INPUT_FILE, OUTPUT_FILE, client, MODEL)


if __name__ == "__main__":
    main()