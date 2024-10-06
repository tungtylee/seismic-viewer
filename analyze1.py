import json

if __name__ == "__main__":
    with open('grid_search_results.json') as f:
        results = json.load(f)

    # best scores
    idx, best_score_result = max(enumerate(results), key=lambda x: x[1]['score'])
    # print(idx, f"best_score_result: {best_score_result}")
    print("Exp1: best score")
    print("params", best_score_result["params"])
    print("score", best_score_result["score"])

    # best recall time shift less than 320 seconds
    idx, best_recalln2_result = max(enumerate(results), key=lambda x: x[1]['recall'][-2])
    # print(idx, f"best_recalln2_result: {best_recalln2_result}")
    print("Exp2: best recall at timeshift<320")

    print("params", best_recalln2_result["params"])
    print("recall%", best_recalln2_result['recall'][-2]*100)
    print("fp", best_recalln2_result["fp"][-2])

    # reference algorithm
    # [120, 600, 4.0, 1.5]
    with open('grid_search_results_ref.json') as f:
        specific_params_data = json.load(f)[0]
        print("Exp3: reference algo params [120, 600, 4.0, 1.5]")
        print("params", specific_params_data["params"])
        print("score", specific_params_data["score"])
        print("recall%", specific_params_data['recall'][-2]*100)
        print("fp", specific_params_data["fp"][-2])


    # best recall at timeshift<320 if fp < 1.2*699
    best_fp_result = max(((idx, x) for idx, x in enumerate(results) if x['fp'][-2] < 1.2*699), key=lambda x: x[1]['recall'][-2], default=None)
    if best_fp_result:
        best_fp_index, best_fp_data = best_fp_result
        # print(idx, f"best recall and fp < 1.2*699: {best_fp_data}")
        print("Exp4: best recall if fp < 1.2*699")
        print("params", best_fp_data["params"])
        print("recall%", best_fp_data['recall'][-2] * 100)
        print("fp", best_fp_data["fp"][-2])
    else:
        print("Not found")

    # minimized fp if recall > 97*0.8
    best_fp_result = min(((idx, x) for idx, x in enumerate(results) if x['recall'][-2]*100 > 97*0.8), key=lambda x: x[1]['fp'][-2], default=None)
    if best_fp_result:
        best_fp_index, best_fp_data = best_fp_result
        # print(idx, f"best recall and fp < 97*0.8: {best_fp_data}")
        print("Exp5: minimized fp if recall > 97*0.8")
        print("params", best_fp_data["params"])
        print("recall%", best_fp_data['recall'][-2] * 100)
        print("fp", best_fp_data["fp"][-2])
    else:
        print("Not found")
    

    ## mars
    with open('mars_grid_search_results.json') as f:
        results = json.load(f)

    # best scores
    idx, best_score_result = max(enumerate(results), key=lambda x: x[1]['score'])
    # print(idx, f"best_score_result: {best_score_result}")
    print("Exp1: best score")
    print("params", best_score_result["params"])
    print("score", best_score_result["score"])

    # best recall time shift less than 320 seconds
    idx, best_recalln2_result = max(enumerate(results), key=lambda x: x[1]['recall'][-2])
    # print(idx, f"best_recalln2_result: {best_recalln2_result}")
    print("Exp2: best recall at timeshift<320")

    print("params", best_recalln2_result["params"])
    print("recall%", best_recalln2_result['recall'][-2]*100)
    print("fp", best_recalln2_result["fp"][-2])

    # reference algorithm
    # [120, 600, 4.0, 1.5]
    with open('mars_grid_search_results_ref.json') as f:
        specific_params_data = json.load(f)[0]
        print("Exp3: reference algo params [120, 600, 4.0, 1.5]")
        print("params", specific_params_data["params"])
        print("score", specific_params_data["score"])
        print("recall%", specific_params_data['recall'][-2]*100)
        print("fp", specific_params_data["fp"][-2])


    # best recall at timeshift<320 if fp < 1
    best_fp_result = max(((idx, x) for idx, x in enumerate(results) if x['fp'][-2] < 1), key=lambda x: x[1]['recall'][-2], default=None)
    if best_fp_result:
        best_fp_index, best_fp_data = best_fp_result
        # print(idx, f"best recall and fp < 1.2*699: {best_fp_data}")
        print("Exp4: best recall if fp < 1")
        print("params", best_fp_data["params"])
        print("recall%", best_fp_data['recall'][-2] * 100)
        print("fp", best_fp_data["fp"][-2])
    else:
        print("Not found")


    # Exp5 [60, 560, 3.25, 0.75]
    specific_params_result = next(((idx, x) for idx, x in enumerate(results) if x['params'] == [60, 560, 3.25, 0.75]), None)
    if specific_params_result:
        specific_params_index, specific_params_data = specific_params_result
        print("Exp5: candidate algo params [60, 560, 3.25, 0.75]")
        print("params", specific_params_data["params"])
        print("score", specific_params_data["score"])
        print("recall%", specific_params_data['recall'][-2]*100)
        print("fp", specific_params_data["fp"][-2])
    else:
        print("Not found")
    
    # Exp6 [120, 720, 3.25, 0.75]
    specific_params_result = next(((idx, x) for idx, x in enumerate(results) if x['params'] == [120, 720, 3.25, 0.75]), None)
    if specific_params_result:
        specific_params_index, specific_params_data = specific_params_result
        print("Exp6: candidate algo params [60, 560, 3.25, 0.75]")
        print("params", specific_params_data["params"])
        print("score", specific_params_data["score"])
        print("recall%", specific_params_data['recall'][-2]*100)
        print("fp", specific_params_data["fp"][-2])
    else:
        print("Not found")
