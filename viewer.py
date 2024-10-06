import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import read

import eval
import gridsearch


def viewer_all(
    tr, tr_data, figtitle, arrivallist_in_s, figfn=None, arrivallist_in_blue=[]
):
    tr_times = tr.times()
    # Initialize figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))

    # Plot trace
    ax.plot(tr_times, tr_data)

    for idx, arrival in enumerate(arrivallist_in_s):
        # Mark detection
        ax.axvline(x=arrival, color="red", label="Rel. Arrival")
        # ax.legend(loc='upper left')

    for idx, arrival in enumerate(arrivallist_in_blue):
        # Mark detection
        ax.axvline(x=arrival, color="blue", linestyle="--", label="Rel. Arrival")
        # ax.legend(loc='upper left')

    # Make the plot pretty
    ax.set_xlim([min(tr_times), max(tr_times)])
    ax.set_ylabel("Velocity (m/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{figtitle}", fontweight="bold")
    if figfn is None:
        plt.show()
        plt.close("all")
    else:
        fig.savefig(figfn, dpi=300, bbox_inches="tight")
        plt.close("all")


if __name__ == "__main__":
    # reference setting
    # [120, 600, 4.0, 1.5]

    # candidate param for lunar
    # [60, 560, 3.25, 0.75]
    # [120, 720, 3.25, 0.75]

    # candidate param for mars
    # [60, 400, 4.75, 0.75]

    os.makedirs("algdev_fig", exist_ok=True)

    sta_len_v = [120, 60, 120, 60]
    lta_len_v = [600, 560, 720, 400]
    thr_on_v = [4.0, 3.25, 3.25, 4.75]
    thr_off_v = [1.5, 0.75, 0.75, 0.75]

    ## lunar
    basedir = "space_apps_2024_seismic_detection"
    data_directory = basedir + "/data/lunar/training/data/S12_GradeA/"
    fnlist = gridsearch.read_list_to_fnlist(os.path.join("algdev", "lunar_gtlist.txt"))
    figfnprefix = os.path.join("algdev_fig", "lunar_")

    for idx, fn in enumerate(fnlist):
        subfnlist = [fn]
        for m in range(4):
            sta_len, lta_len, thr_on, thr_off = (
                sta_len_v[m],
                lta_len_v[m],
                thr_on_v[m],
                thr_off_v[m],
            )
            fnamelist, time_abs_str_list, time_rel_list, alltr, alltrdata = (
                eval.run_stalta_from_fnlist(
                    data_directory,
                    subfnlist,
                    sta_len,
                    lta_len,
                    thr_on,
                    thr_off,
                    ext=True,
                )
            )

            if len(alltr) > 0:
                tr = alltr[0]
                tr_data = alltrdata[0]
                params = f"params: {sta_len}, {lta_len}, {thr_on}, {thr_off}"
                figtitle = fn + "\n" + params
                arrivallist_in_s = time_rel_list
                figfn = figfnprefix + str(idx) + f"_m{m}.png"
                viewer_all(tr, tr_data, figtitle, arrivallist_in_s, figfn=figfn)
            else:
                print("File Missing:", fn)

    ## mars
    basedir = "space_apps_2024_seismic_detection"
    data_directory = basedir + "/data/mars/training/data/"
    fnlist = gridsearch.read_list_to_fnlist(os.path.join("algdev", "mars_gtlist.txt"))
    figfnprefix = os.path.join("algdev_fig", "mars_")

    for idx, fn in enumerate(fnlist):
        subfnlist = [fn]
        for m in range(4):
            sta_len, lta_len, thr_on, thr_off = (
                sta_len_v[m],
                lta_len_v[m],
                thr_on_v[m],
                thr_off_v[m],
            )
            fnamelist, time_abs_str_list, time_rel_list, alltr, alltrdata = (
                eval.run_stalta_from_fnlist(
                    data_directory,
                    subfnlist,
                    sta_len,
                    lta_len,
                    thr_on,
                    thr_off,
                    ext=True,
                )
            )

            if len(alltr) > 0:
                tr = alltr[0]
                tr_data = alltrdata[0]
                params = f"params: {sta_len}, {lta_len}, {thr_on}, {thr_off}"
                figtitle = fn + "\n" + params
                arrivallist_in_s = time_rel_list
                figfn = figfnprefix + str(idx) + f"_m{m}.png"
                viewer_all(tr, tr_data, figtitle, arrivallist_in_s, figfn=figfn)
            else:
                print("File Missing:", fn)
