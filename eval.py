import csv
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import read

import baseline
import gaussian


def generate_predcsv_from_list(fn, fnlist, time_abs_str_list, time_rel_list):
    # Compile dataframe of detections
    detect_df = pd.DataFrame(
        data={
            "filename": fnlist,
            "time_abs(%Y-%m-%dT%H:%M:%S.%f)": time_abs_str_list,
            "time_rel(sec)": time_rel_list,
        }
    )
    detect_df.head()
    detect_df.to_csv(fn, index=False)


def run_stalta_from_fnlist(
    data_directory, fnlist, sta_len=120, lta_len=600, thr_on=4, thr_off=1.5, ext=False
):
    # basedir = "space_apps_2024_seismic_detection"
    # data_directory = basedir + '/data/lunar/training/data/S12_GradeA/'
    fnamelist = []
    time_abs_str_list = []
    time_rel_list = []
    alltr = []
    alltrdata = []
    for fn in fnlist:
        # handle ext
        if fn[-4:] == ".csv":
            fn = fn[:-4]
        if fn[-6:] == ".mseed":
            fn = fn[:-6]
        mseed_file = f"{data_directory}{fn}.mseed"
        if os.path.exists(mseed_file) is False:
            print(f"Skip {fn}")
            continue
        st = read(mseed_file)
        # This is how you get the data and the time, which is in seconds
        tr = st.traces[0].copy()
        tr_times = tr.times()
        tr_data = tr.data
        if ext:
            alltr.append(tr)
            alltrdata.append(tr_data)
        on_off = baseline.stalta(tr, tr_data, sta_len, lta_len, thr_on, thr_off)
        starttime = tr.stats.starttime.datetime
        for i in np.arange(0, len(on_off)):
            triggers = on_off[i]
            on_time = starttime + timedelta(seconds=tr_times[triggers[0]])
            on_time_str = datetime.strftime(on_time, "%Y-%m-%dT%H:%M:%S.%f")
            time_rel_list.append(tr_times[triggers[0]])
            time_abs_str_list.append(on_time_str)
            fnamelist.append(fn)
    if ext:
        return fnamelist, time_abs_str_list, time_rel_list, alltr, alltrdata
    else:
        return fnamelist, time_abs_str_list, time_rel_list


def generate_sample_pred(fn):
    basedir = "space_apps_2024_seismic_detection"
    data_directory = basedir + "/data/lunar/training/data/S12_GradeA/"
    fnlist = ["xa.s12.00.mhz.1970-06-26HR00_evid00009"]

    fnamelist, time_abs_str_list, time_rel_list = run_stalta_from_fnlist(
        data_directory, fnlist, sta_len=120, lta_len=600, thr_on=4, thr_off=1.5
    )
    generate_predcsv_from_list(fn, fnamelist, time_abs_str_list, time_rel_list)


def eval(csvgt, csvpred, tols=5, verbose=False):
    gt_df = pd.read_csv(csvgt)
    pred_df = pd.read_csv(csvpred)

    # use dateime object
    gt_df["time_abs"] = pd.to_datetime(
        gt_df["time_abs(%Y-%m-%dT%H:%M:%S.%f)"], format="%Y-%m-%dT%H:%M:%S.%f"
    )
    pred_df["time_abs"] = pd.to_datetime(
        pred_df["time_abs(%Y-%m-%dT%H:%M:%S.%f)"], format="%Y-%m-%dT%H:%M:%S.%f"
    )

    hits = 0
    false_positive = 0
    gt_total = len(gt_df)

    pred_used = set()

    for index, gt_row in gt_df.iterrows():
        gt_filename = gt_row["filename"]
        gt_time_abs = gt_row["time_abs"]
        matching_preds = pred_df[pred_df["filename"] == gt_filename]
        if not matching_preds.empty:
            matching_preds = matching_preds.copy()
            matching_preds["time_diff"] = (
                (matching_preds["time_abs"] - gt_time_abs).abs().dt.total_seconds()
            )

            closest_pred_idx = matching_preds["time_diff"].idxmin()
            closest_pred = matching_preds.loc[closest_pred_idx]

            if closest_pred["time_diff"] < tols:
                hits += 1
                pred_used.add(closest_pred_idx)  # 記錄已經使用過的預測

    false_positive = len(pred_df) - len(pred_used)
    recall_rate = hits / gt_total if gt_total > 0 else 0

    if verbose:
        print(
            f"Tols: {tols} / Recall rate: {recall_rate:.2f} / False positives: {false_positive}"
        )
    return tols, recall_rate, false_positive


def eval_curves(csvgt, csvpred, verbose=False):
    tol = []
    recall = []
    fp = []
    for expo in range(8):
        tols = (2**expo) * 5
        tols, recall_rate, false_positive = eval(
            csvgt, csvpred, tols=tols, verbose=verbose
        )
        tol.append(tols)
        recall.append(recall_rate)
        fp.append(false_positive)
    return tol, recall, fp


if __name__ == "__main__":
    csvgt = os.path.join("algdev", "lunar_traingt.txt")
    csvpred = os.path.join("algdev", "sample_lunar_traingt.txt")
    generate_sample_pred(csvpred)
    tol, recall, fp = eval_curves(csvgt, csvpred, verbose=True)
