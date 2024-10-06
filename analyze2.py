import json

if __name__ == "__main__":
    print("lunar")
    with open("grid_search_results_val.json") as f:
        results = json.load(f)
        for idx, result in enumerate(results):
            print(f"Exp{idx}")
            print("params", result["params"])
            print("score", result["score"])
            print("recall%", result["recall"][-2] * 100)
            print("fp", result["fp"][-2])
    print("Mars")
    with open("mars_grid_search_results_val.json") as f:
        results = json.load(f)
        for idx, result in enumerate(results):
            print(f"Exp{idx}")
            print("params", result["params"])
            print("score", result["score"])
            print("recall%", result["recall"][-2] * 100)
            print("fp", result["fp"][-2])
