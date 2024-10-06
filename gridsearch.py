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


def exp_for_train():
    sta_len_v = list(range(60, 201, 20))
    lta_len_v = list(range(400, 761, 40))
    thr_on_v = [3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75]
    thr_off_v = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    # sta_len_v = [120]
    # lta_len_v = [600]
    # thr_on_v = [4.0]
    # thr_off_v = [1.5]
    basedir = "space_apps_2024_seismic_detection"
    data_directory = basedir + "/data/lunar/training/data/S12_GradeA/"
    results = []
    comb = 0
    for sta_len, lta_len, thr_on, thr_off in itertools.product(
        sta_len_v, lta_len_v, thr_on_v, thr_off_v
    ):
        print(comb)
        fnlist = read_list_to_fnlist(os.path.join("algdev", "lunar_trainlist.txt"))
        csvgt = os.path.join("algdev", "lunar_traingt.txt")
        csvpred = os.path.join("algdev", f"lunar_traingt_stalta_{comb}.txt")
        if os.path.exists(csvpred) is False:
            fnamelist, time_abs_str_list, time_rel_list = eval.run_stalta_from_fnlist(
                data_directory, fnlist, sta_len, lta_len, thr_on, thr_off
            )
            eval.generate_predcsv_from_list(
                csvpred, fnamelist, time_abs_str_list, time_rel_list
            )
        tol, recall, fp = eval.eval_curves(csvgt, csvpred)
        score = float(np.mean(np.array(recall)) * 100)
        nfp = int(np.sum(np.array(fp)))
        results.append(
            {
                "params": (sta_len, lta_len, thr_on, thr_off),
                "score": score,
                "nfp": nfp,
                "tol": tol,
                "recall": recall,
                "fp": fp,
            }
        )
        comb += 1
    with open("grid_search_results.json", "w") as f:
        json.dump(results, f, indent=4)

    sta_len_v = [120]
    lta_len_v = [600]
    thr_on_v = [4.0]
    thr_off_v = [1.5]
    # sta_len_v = [120]
    # lta_len_v = [600]
    # thr_on_v = [4.0]
    # thr_off_v = [1.5]
    basedir = "space_apps_2024_seismic_detection"
    data_directory = basedir + "/data/lunar/training/data/S12_GradeA/"
    results = []
    comb = 0
    for sta_len, lta_len, thr_on, thr_off in itertools.product(
        sta_len_v, lta_len_v, thr_on_v, thr_off_v
    ):
        print(comb)
        fnlist = read_list_to_fnlist(os.path.join("algdev", "lunar_trainlist.txt"))
        csvgt = os.path.join("algdev", "lunar_traingt.txt")
        csvpred = os.path.join("algdev", f"lunar_traingt_refstalta_{comb}.txt")
        if os.path.exists(csvpred) is False:
            fnamelist, time_abs_str_list, time_rel_list = eval.run_stalta_from_fnlist(
                data_directory, fnlist, sta_len, lta_len, thr_on, thr_off
            )
            eval.generate_predcsv_from_list(
                csvpred, fnamelist, time_abs_str_list, time_rel_list
            )
        tol, recall, fp = eval.eval_curves(csvgt, csvpred)
        score = float(np.mean(np.array(recall)) * 100)
        nfp = int(np.sum(np.array(fp)))
        results.append(
            {
                "params": (sta_len, lta_len, thr_on, thr_off),
                "score": score,
                "nfp": nfp,
                "tol": tol,
                "recall": recall,
                "fp": fp,
            }
        )
        comb += 1
    with open("grid_search_results_ref.json", "w") as f:
        json.dump(results, f, indent=4)

    ## mars
    sta_len_v = list(range(60, 201, 20))
    lta_len_v = list(range(400, 761, 40))
    thr_on_v = [3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75]
    thr_off_v = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    # sta_len_v = [120]
    # lta_len_v = [600]
    # thr_on_v = [4.0]
    # thr_off_v = [1.5]
    basedir = "space_apps_2024_seismic_detection"
    data_directory = basedir + "/data/mars/training/data/"
    results = []
    comb = 0
    for sta_len, lta_len, thr_on, thr_off in itertools.product(
        sta_len_v, lta_len_v, thr_on_v, thr_off_v
    ):
        print(comb)
        fnlist = read_list_to_fnlist(os.path.join("algdev", "mars_trainlist.txt"))
        csvgt = os.path.join("algdev", "mars_traingt.txt")
        csvpred = os.path.join("algdev", f"mars_traingt_stalta_{comb}.txt")
        if os.path.exists(csvpred) is False:
            fnamelist, time_abs_str_list, time_rel_list = eval.run_stalta_from_fnlist(
                data_directory, fnlist, sta_len, lta_len, thr_on, thr_off
            )
            eval.generate_predcsv_from_list(
                csvpred, fnamelist, time_abs_str_list, time_rel_list
            )
        tol, recall, fp = eval.eval_curves(csvgt, csvpred)
        score = float(np.mean(np.array(recall)) * 100)
        nfp = int(np.sum(np.array(fp)))
        results.append(
            {
                "params": (sta_len, lta_len, thr_on, thr_off),
                "score": score,
                "nfp": nfp,
                "tol": tol,
                "recall": recall,
                "fp": fp,
            }
        )
        comb += 1
    with open("mars_grid_search_results.json", "w") as f:
        json.dump(results, f, indent=4)

    sta_len_v = [120]
    lta_len_v = [600]
    thr_on_v = [4.0]
    thr_off_v = [1.5]
    # sta_len_v = [120]
    # lta_len_v = [600]
    # thr_on_v = [4.0]
    # thr_off_v = [1.5]
    basedir = "space_apps_2024_seismic_detection"
    data_directory = basedir + "/data/mars/training/data/"
    results = []
    comb = 0
    for sta_len, lta_len, thr_on, thr_off in itertools.product(
        sta_len_v, lta_len_v, thr_on_v, thr_off_v
    ):
        print(comb)
        fnlist = read_list_to_fnlist(os.path.join("algdev", "lunar_gtlist.txt"))
        csvgt = os.path.join("algdev", "mars_traingt.txt")
        csvpred = os.path.join("algdev", f"mars_traingt_refstalta_{comb}.txt")
        if os.path.exists(csvpred) is False:
            fnamelist, time_abs_str_list, time_rel_list = eval.run_stalta_from_fnlist(
                data_directory, fnlist, sta_len, lta_len, thr_on, thr_off
            )
            eval.generate_predcsv_from_list(
                csvpred, fnamelist, time_abs_str_list, time_rel_list
            )
        tol, recall, fp = eval.eval_curves(csvgt, csvpred)
        score = float(np.mean(np.array(recall)) * 100)
        nfp = int(np.sum(np.array(fp)))
        results.append(
            {
                "params": (sta_len, lta_len, thr_on, thr_off),
                "score": score,
                "nfp": nfp,
                "tol": tol,
                "recall": recall,
                "fp": fp,
            }
        )
        comb += 1
    with open("mars_grid_search_results_ref.json", "w") as f:
        json.dump(results, f, indent=4)


def exp_for_val():
    sta_len_v = [120, 60, 160, 100]
    lta_len_v = [600, 720, 680, 760]
    thr_on_v = [4.0, 3.5, 3.25, 3.75]
    thr_off_v = [1.5, 0.75, 0.75, 1.0]
    basedir = "space_apps_2024_seismic_detection"
    data_directory = basedir + "/data/lunar/training/data/S12_GradeA/"
    results = []

    for m in range(4):
        comb = m
        sta_len, lta_len, thr_on, thr_off = (
            sta_len_v[m],
            lta_len_v[m],
            thr_on_v[m],
            thr_off_v[m],
        )
        fnlist = read_list_to_fnlist(os.path.join("algdev", "lunar_vallist.txt"))
        csvgt = os.path.join("algdev", "lunar_valgt.txt")
        csvpred = os.path.join("algdev", f"lunar_valgt_stalta_{comb}.txt")
        if os.path.exists(csvpred) is False:
            fnamelist, time_abs_str_list, time_rel_list = eval.run_stalta_from_fnlist(
                data_directory, fnlist, sta_len, lta_len, thr_on, thr_off
            )
            eval.generate_predcsv_from_list(
                csvpred, fnamelist, time_abs_str_list, time_rel_list
            )
        tol, recall, fp = eval.eval_curves(csvgt, csvpred)
        score = float(np.mean(np.array(recall)) * 100)
        nfp = int(np.sum(np.array(fp)))
        results.append(
            {
                "params": (sta_len, lta_len, thr_on, thr_off),
                "score": score,
                "nfp": nfp,
                "tol": tol,
                "recall": recall,
                "fp": fp,
            }
        )
        comb += 1
    with open("grid_search_results_val.json", "w") as f:
        json.dump(results, f, indent=4)

    ## mars
    sta_len_v = [120, 60, 60, 60, 160, 100]
    lta_len_v = [600, 400, 400, 720, 680, 760]
    thr_on_v = [4.0, 3.25, 4.75, 3.5, 3.25, 3.75]
    thr_off_v = [1.5, 0.75, 0.75, 0.75, 0.75, 1.0]
    basedir = "space_apps_2024_seismic_detection"
    data_directory = basedir + "/data/mars/training/data/"
    results = []
    for m in range(6):
        comb = m
        sta_len, lta_len, thr_on, thr_off = (
            sta_len_v[m],
            lta_len_v[m],
            thr_on_v[m],
            thr_off_v[m],
        )
        fnlist = read_list_to_fnlist(os.path.join("algdev", "mars_vallist.txt"))
        csvgt = os.path.join("algdev", "mars_valgt.txt")
        csvpred = os.path.join("algdev", f"mars_valgt_stalta_{comb}.txt")
        if os.path.exists(csvpred) is False:
            fnamelist, time_abs_str_list, time_rel_list = eval.run_stalta_from_fnlist(
                data_directory, fnlist, sta_len, lta_len, thr_on, thr_off
            )
            eval.generate_predcsv_from_list(
                csvpred, fnamelist, time_abs_str_list, time_rel_list
            )
        tol, recall, fp = eval.eval_curves(csvgt, csvpred)
        score = float(np.mean(np.array(recall)) * 100)
        nfp = int(np.sum(np.array(fp)))
        results.append(
            {
                "params": (sta_len, lta_len, thr_on, thr_off),
                "score": score,
                "nfp": nfp,
                "tol": tol,
                "recall": recall,
                "fp": fp,
            }
        )
        comb += 1
    with open("mars_grid_search_results_val.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    # run gridsearch
    # exp_for_train()
    exp_for_val()
