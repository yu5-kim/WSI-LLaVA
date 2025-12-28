"""
WSI-Relevance Stage 3: Score Aggregation
This script calculates average relevance scores from evaluation results.
"""

import os
import json
from tqdm import tqdm


def load_eval_data(file_path):
    """Load evaluation data from JSON file."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        print(f"Loaded evaluation data from {file_path}")
        return eval_data
    else:
        print(f"Evaluation file not found: {file_path}")
        return {}


def calculate_per_entry_scores(eval_data):
    """
    Calculate average score for each entry.
    
    Args:
        eval_data: Dictionary containing evaluation results
        
    Returns:
        per_entry_scores: {question_id: {type: average_score}}
    """
    per_entry_scores = {}

    for key, value in tqdm(eval_data.items(), desc="Calculating per-entry scores"):
        entry_type = value.get("type", "default_type")
        claim_results = value.get("claim_results", [])

        if not claim_results:
            per_entry_scores[key] = {entry_type: None}
            continue

        # Extract valid scores
        scores = [
            claim.get("score") 
            for claim in claim_results 
            if isinstance(claim.get("score"), (int, float))
        ]

        if not scores:
            per_entry_scores[key] = {entry_type: None}
            continue

        # Calculate average score
        average_score = sum(scores) / len(scores)
        per_entry_scores[key] = {entry_type: round(average_score, 3)}

    return per_entry_scores


def compute_average(scores):
    """Calculate average score with 3 decimal places."""
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 3)


def build_type_average_scores(per_entry_scores):
    """
    Calculate average scores for each type.
    
    Args:
        per_entry_scores: Dictionary of per-entry scores
        
    Returns:
        type_average_scores: {type: average_score}
    """
    type_scores = {}
    
    for entry in per_entry_scores.values():
        for type_, score in entry.items():
            if score is not None:
                if type_ not in type_scores:
                    type_scores[type_] = []
                type_scores[type_].append(score)

    type_average_scores = {
        type_: compute_average(scores) 
        for type_, scores in type_scores.items()
    }
    
    return type_average_scores


def build_overall_average(per_entry_scores):
    """
    Calculate overall average score across all entries.
    
    Args:
        per_entry_scores: Dictionary of per-entry scores
        
    Returns:
        overall_average: float
    """
    scores = [
        score 
        for entry in per_entry_scores.values() 
        for score in entry.values() 
        if score is not None
    ]
    
    return compute_average(scores)


def process_evaluation_file(eval_file_path):
    """
    Process a single evaluation file and calculate all scores.
    
    Args:
        eval_file_path: Path to evaluation JSON file
    """
    # Load evaluation data
    eval_data = load_eval_data(eval_file_path)
    
    if not eval_data:
        print(f"No data to process in {eval_file_path}")
        return

    # Calculate per-entry scores
    per_entry_scores = calculate_per_entry_scores(eval_data)

    # Calculate type-specific average scores
    type_average_scores = build_type_average_scores(per_entry_scores)
    print("\nAverage scores by type:")
    for type_, avg in type_average_scores.items():
        print(f"  {type_}: {avg}")

    # Calculate overall average score
    overall_average = build_overall_average(per_entry_scores)
    print(f"\nOverall average score: {overall_average}")

    # Combine all results
    combined_score_data = {
        "per_entry_scores": per_entry_scores,
        "overall_average": overall_average,
        "type_average_scores": type_average_scores
    }

    # Save combined results
    output_file = eval_file_path.replace("relevance_evaluation_results", "relevance_aggregated_scores")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_score_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nAll scores saved to {output_file}")
    print(f"Score calculation completed for {eval_file_path}")


def process_folder(folder_path, pattern="relevance_evaluation_results"):
    """
    Process all evaluation files in a folder.
    
    Args:
        folder_path: Path to folder containing evaluation files
        pattern: Filename pattern to match (default: "relevance_evaluation_results")
    """
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    for file_name in os.listdir(folder_path):
        if pattern in file_name and file_name.endswith('.json'):
            eval_file_path = os.path.join(folder_path, file_name)
            print(f"\n{'='*60}")
            print(f"Processing: {file_name}")
            print(f"{'='*60}")
            process_evaluation_file(eval_file_path)


def main():
    """Main execution function."""
    # Configuration
    FOLDER_PATH = "evaluation_outputs"
    PATTERN = "relevance_evaluation_results"
    
    # Process all evaluation files in folder
    process_folder(FOLDER_PATH, PATTERN)
    
    # Or process a single file
    # EVAL_FILE = "relevance_evaluation_results.json"
    # process_evaluation_file(EVAL_FILE)


if __name__ == "__main__":
    main()