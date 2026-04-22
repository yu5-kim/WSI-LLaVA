"""
NLP Metrics Evaluator

This script evaluates NLP model predictions using multiple metrics:
- BLEU (1-4)
- ROUGE-L
- METEOR


"""

import json
import argparse
from typing import Dict, Tuple
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import single_meteor_score


class MetricsEvaluator:
    """Evaluate model predictions against ground truth using NLP metrics."""
    
    def __init__(self):
        """Initialize the evaluator and download required NLTK resources."""
        self.rouge_metric = Rouge()
        self.smoothie = SmoothingFunction().method4
        self._download_nltk_resources()
    
    @staticmethod
    def _download_nltk_resources():
        """Download required NLTK resources if not already present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading NLTK wordnet for METEOR...")
            nltk.download('wordnet', quiet=True)
    
    @staticmethod
    def load_data(file_path: str, gt_key: str = 'T-answer',
                  pred_key: str = 'Output', id_key: str = 'question_id') -> Tuple[Dict, Dict]:
        """
        Load ground truth and prediction data from JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            gt_key: Key for ground truth answers
            pred_key: Key for predicted answers
            id_key: Key for question IDs
            
        Returns:
            Tuple of (ground_truth_dict, predictions_dict)
        """
        gt_answers = {}
        pred_answers = {}
        skipped_empty = 0
        empty_predictions = 0
        used_pred_fallback = 0
        used_gt_fallback = 0
        used_id_fallback = 0

        # Common key aliases across evaluation outputs.
        gt_aliases = ('T-answer', 't_answer', 'ground_truth', 'gt', 'answer')
        pred_aliases = ('Output', 'output', 'prediction', 'pred', 'answer', 'response')
        id_aliases = ('question_id', 'qid', 'id')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    qid = data.get(id_key, '')
                    if not qid:
                        for key in id_aliases:
                            value = data.get(key, '')
                            if value:
                                qid = value
                                used_id_fallback += 1
                                break

                    gt_text = data.get(gt_key, '')
                    if not gt_text:
                        for key in gt_aliases:
                            value = data.get(key, '')
                            if value:
                                gt_text = value
                                if key != gt_key:
                                    used_gt_fallback += 1
                                break

                    pred_text = data.get(pred_key, '')
                    if not pred_text:
                        for key in pred_aliases:
                            value = data.get(key, '')
                            if value and value != gt_text:
                                pred_text = value
                                if key != pred_key:
                                    used_pred_fallback += 1
                                break
                        # If only one answer-like field exists, allow it for prediction.
                        if not pred_text and isinstance(data.get('answer', ''), str):
                            pred_text = data.get('answer', '')
                            if pred_key != 'answer':
                                used_pred_fallback += 1

                    qid = str(qid).strip()
                    gt_text = str(gt_text).strip()
                    pred_text = str(pred_text).strip()

                    # Require id + ground truth; empty predictions are kept so the
                    # sample count matches inference (failed generations penalize the average).
                    if qid and gt_text:
                        gt_answers[qid] = gt_text
                        pred_answers[qid] = pred_text
                        if not pred_text:
                            empty_predictions += 1
                    else:
                        skipped_empty += 1
                    
                except json.JSONDecodeError:
                    print(f"⚠️  Warning: Failed to parse line {line_num}, skipping...")
                    continue
        
        print(f"📌 Loaded pairs: {len(gt_answers)} (skipped: {skipped_empty})")
        if empty_predictions:
            print(f"ℹ️  Empty predictions included in metrics: {empty_predictions} (counted as ~0 score)")
        if used_id_fallback:
            print(f"ℹ️  Used fallback ID keys for {used_id_fallback} records")
        if used_gt_fallback:
            print(f"ℹ️  Used fallback GT keys for {used_gt_fallback} records")
        if used_pred_fallback:
            print(f"ℹ️  Used fallback prediction keys for {used_pred_fallback} records")

        return gt_answers, pred_answers
    
    def calculate_bleu(self, reference_tokens: list, candidate_tokens: list) -> list:
        """
        Calculate BLEU-1 to BLEU-4 scores.
        
        Args:
            reference_tokens: Tokenized reference text
            candidate_tokens: Tokenized candidate text
            
        Returns:
            List of BLEU scores [BLEU-1, BLEU-2, BLEU-3, BLEU-4]
        """
        bleu_scores = []
        for n in range(1, 5):
            weights = tuple((1.0 / n) for _ in range(n)) + tuple(0.0 for _ in range(4 - n))
            try:
                score = sentence_bleu(
                    [reference_tokens], 
                    candidate_tokens, 
                    weights=weights, 
                    smoothing_function=self.smoothie
                )
            except Exception:
                score = 0.0
            bleu_scores.append(score)
        
        return bleu_scores
    
    def calculate_rouge_l(self, reference_tokens: list, candidate_tokens: list) -> float:
        """
        Calculate ROUGE-L F1 score.
        
        Args:
            reference_tokens: Tokenized reference text
            candidate_tokens: Tokenized candidate text
            
        Returns:
            ROUGE-L F1 score
        """
        try:
            rouge_scores = self.rouge_metric.get_scores(
                ' '.join(candidate_tokens), 
                ' '.join(reference_tokens)
            )[0]
            return rouge_scores['rouge-l']['f']
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def calculate_meteor(self, reference_tokens: list, candidate_tokens: list) -> float:
        """
        Calculate METEOR score.
        
        Args:
            reference_tokens: Tokenized reference text
            candidate_tokens: Tokenized candidate text
            
        Returns:
            METEOR score
        """
        try:
            return single_meteor_score(reference_tokens, candidate_tokens)
        except Exception:
            return 0.0
    
    def evaluate(self, gt_answers: Dict[str, str], pred_answers: Dict[str, str]) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.
        
        Args:
            gt_answers: Dictionary of ground truth answers
            pred_answers: Dictionary of predicted answers
            
        Returns:
            Dictionary of average metric scores
        """
        # Find matching question IDs
        matched_ids = set(gt_answers.keys()) & set(pred_answers.keys())
        
        if not matched_ids:
            raise ValueError("No matching question_ids found between ground truth and predictions")
        
        print(f"✅ Found {len(matched_ids)} matching question_ids")
        
        # Initialize accumulators
        total_bleu = [0.0] * 4
        total_rouge_l = 0.0
        total_meteor = 0.0
        total_count = len(matched_ids)
        
        # Calculate metrics for each pair
        for qid in matched_ids:
            reference = gt_answers[qid]
            candidate = pred_answers[qid]
            
            # Tokenize
            reference_tokens = nltk.word_tokenize(reference)
            candidate_tokens = nltk.word_tokenize(candidate)
            
            # Calculate BLEU scores
            bleu_scores = self.calculate_bleu(reference_tokens, candidate_tokens)
            for i, score in enumerate(bleu_scores):
                total_bleu[i] += score
            
            # Calculate ROUGE-L
            total_rouge_l += self.calculate_rouge_l(reference_tokens, candidate_tokens)
            
            # Calculate METEOR
            total_meteor += self.calculate_meteor(reference_tokens, candidate_tokens)
        
        # Calculate averages
        results = {
            'BLEU-1': total_bleu[0] / total_count,
            'BLEU-2': total_bleu[1] / total_count,
            'BLEU-3': total_bleu[2] / total_count,
            'BLEU-4': total_bleu[3] / total_count,
            'ROUGE-L': total_rouge_l / total_count,
            'METEOR': total_meteor / total_count,
            'num_samples': total_count
        }
        
        return results
    
    @staticmethod
    def print_results(results: Dict[str, float]):
        """
        Print evaluation results in a formatted table.
        
        Args:
            results: Dictionary of metric scores
        """
        print("\n" + "="*50)
        print("📊 Evaluation Results")
        print("="*50)
        print(f"Number of samples: {results['num_samples']}")
        print("-"*50)
        print(f"BLEU-1:   {results['BLEU-1']:.4f}")
        print(f"BLEU-2:   {results['BLEU-2']:.4f}")
        print(f"BLEU-3:   {results['BLEU-3']:.4f}")
        print(f"BLEU-4:   {results['BLEU-4']:.4f}")
        print(f"ROUGE-L:  {results['ROUGE-L']:.4f}")
        print(f"METEOR:   {results['METEOR']:.4f}")
        print("="*50 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Evaluate NLP model predictions using BLEU, ROUGE-L, and METEOR metrics'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input JSONL file containing ground truth and predictions'
    )
    parser.add_argument(
        '--gt-key',
        type=str,
        default='T-answer',
        help='JSON key for ground truth answers (default: T-answer)'
    )
    parser.add_argument(
        '--pred-key',
        type=str,
        default='Output',
        help='JSON key for predicted answers (default: Output)'
    )
    parser.add_argument(
        '--id-key',
        type=str,
        default='question_id',
        help='JSON key for question IDs (default: question_id)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Optional: Path to save results (.json or .tsv; .tsv writes header + one value row)'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MetricsEvaluator()
    
    # Load data
    print(f"📁 Loading data from: {args.input}")
    gt_answers, pred_answers = evaluator.load_data(
        args.input, 
        gt_key=args.gt_key,
        pred_key=args.pred_key,
        id_key=args.id_key
    )
    
    # Evaluate
    print("🔄 Calculating metrics...")
    results = evaluator.evaluate(gt_answers, pred_answers)
    
    # Print results
    evaluator.print_results(results)
    
    # Save results if output path specified
    if args.output:
        if args.output.lower().endswith('.tsv'):
            metric_order = [
                'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4',
                'ROUGE-L', 'METEOR', 'num_samples',
            ]
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write('\t'.join(metric_order) + '\n')
                f.write(
                    '\t'.join(
                        f"{results[k]:.6f}" if k != 'num_samples' else str(int(results[k]))
                        for k in metric_order
                    )
                    + '\n'
                )
        else:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()