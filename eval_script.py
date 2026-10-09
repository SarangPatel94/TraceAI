import os
import json
import time
from query_engine import CodebaseQAEngine

if __name__ == "__main__":
    engine = CodebaseQAEngine()
    start_time = time.perf_counter()
    if os.path.exists("eval_set.json"):
        with open("eval_set.json", "r", encoding="utf-8") as f:
            summary = engine.run_evaluation(json.load(f))
            print(
                f"Evaluation complete: {summary['passed']}/{summary['total']} passed "
                f"({summary['pass_rate'] * 100:.1f}%). "
                f"HIGH/LOW judge confidence: {summary['judge_high']}/{summary['judge_low']}."
            )
            precision = "N/A" if summary["precision_at_5"] is None else f"{summary['precision_at_5'] * 100:.1f}%"
            recall = "N/A" if summary["recall_at_5"] is None else f"{summary['recall_at_5'] * 100:.1f}%"
            print(f"Precision@5: {precision}; Recall@5: {recall}")
            print(f"Results: {summary['output_file']}")
    else:
        print("No file is present called eval_set.json Please add the same and run the eval engine")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time:.4f} seconds")
