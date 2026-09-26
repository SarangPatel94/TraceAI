import os
import json
from query_engine_groq import CodebaseQAEngine

if __name__ == "__main__":
    engine = CodebaseQAEngine()
    if os.path.exists("eval_set.json"):
        with open("eval_set.json", "r", encoding="utf-8") as f:
            engine.run_evaluation(json.load(f))
    else:
        print("No file is present called eval_set.json Please add the same and run the eval engine")