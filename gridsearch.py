import itertools
import json
import os

import numpy as np

import baseline
import eval


def read_list_to_fnlist(fn):
    fnlist = []
    with open(fn) as f:
        for line in f:
            row = line.rstrip()
            if len(row) > 0:
                fnlist.append(row)
    return fnlist


if __name__ == "__main__":
    sta_len_v = list(range(100, 201, 25))
    lta_len_v = list(range(400, 751, 50))
    thr_on_v = [3.5, 3.75, 4.0, 4.25, 4.5, 4.75]
    thr_off_v = [1.0, 1.25, 1.5, 1.75, 2.0]
    # sta_len_v = [120]
    # lta_len_v = [600]
    # thr_on_v = [4.0]
    # thr_off_v = [1.5]
    basedir = "space_apps_2024_seismic_detection"
    data_directory = basedir + '/data/lunar/training/data/S12_GradeA/'
    results = []
    comb = 0
    for sta_len, lta_len, thr_on, thr_off in itertools.product(sta_len_v, lta_len_v, thr_on_v, thr_off_v):
        print(comb)
        fnlist = read_list_to_fnlist(os.path.join("algdev", "lunar_gtlist.txt"))
        csvgt = os.path.join("algdev", "lunar_traingt.txt")
        csvpred = os.path.join("algdev", f"lunar_traingt_stalta_{comb}.txt")
        if os.path.exists(csvpred) is False:
            fnamelist, time_abs_str_list, time_rel_list = eval.run_stalta_from_fnlist(data_directory, fnlist, sta_len, lta_len, thr_on, thr_off)
            eval.generate_predcsv_from_list(csvpred, fnamelist, time_abs_str_list, time_rel_list)
        tol, recall, fp = eval.eval_curves(csvgt, csvpred)
        score = float(np.mean(np.array(recall))*100)
        nfp = int(np.sum(np.array(fp)))
        results.append({'params': (sta_len, lta_len, thr_on, thr_off), 'score': score, 'nfp': nfp, 'tol': tol, 'recall': recall, 'fp': fp})
        comb += 1
    with open('grid_search_results.json', 'w') as f:
        json.dump(results, f, indent=4)

