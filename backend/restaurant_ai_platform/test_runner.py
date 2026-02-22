from pprint import pprint

from .orchestrator import run_pipeline


if __name__ == "__main__":
    print("🚀 Running full pipeline...\n")
    result = run_pipeline()
    print("\n✅ Pipeline finished. Result:\n")
    pprint(result)