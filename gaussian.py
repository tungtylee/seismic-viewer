import os

import matplotlib.pyplot as plt
import numpy as np
from obspy import read

import eval
import gridsearch
import viewer


def merge_intervals(on_off):
    merged = []
    on_off.sort(key=lambda x: x[0])
    current = on_off[0]

    for next_interval in on_off[1:]:
        if current[1] >= next_interval[0]:
            current[1] = max(current[1], next_interval[1])
        else:
            merged.append(current)
            current = next_interval

    merged.append(current)
    return merged


def daywise_stat(tr_data):
    data = tr_data
    return np.mean(data), np.std(data)


def fixedgaussian(tr, tr_data, sigma, step_size=30, span=320):
    data = tr_data
    tr_times = tr.times()  # Get relative time using tr.times()
    sampling_rate = tr.stats.sampling_rate

    # Convert step_size from seconds to samples
    step_samples = int(step_size * sampling_rate)

    # Define the fixed threshold: 6 * sigma
    threshold = 6 * sigma

    # Detect events where abs(signal) exceeds the threshold
    events = []
    for start in range(0, len(data), step_samples):
        # Get the data for the current step
        window = data[start : start + step_samples]

        # Find indices where the absolute value of data exceeds the threshold
        event_indices = np.where(np.abs(window) > threshold)[0] + start

        # Append detected events
        events.extend(event_indices)

    # Remove duplicates and sort events
    events = np.unique(events)

    # Convert detected events to on/off format with the specified span
    on_off = []
    for event in events:
        rel_time = tr_times[event]  # Get relative event time from tr_times
        on_off.append([rel_time, rel_time + span])

    return on_off


def gaussian(tr, tr_data, window_size=3600, step_size=30, span=320):
    data = tr_data
    tr_times = tr.times()  # Use tr.times() for relative time
    sampling_rate = tr.stats.sampling_rate

    # Sliding window parameters
    window_samples = int(window_size * sampling_rate)
    step_samples = int(step_size * sampling_rate)

    # Initialize lists for mean (mu), standard deviation (sigma), and time values
    mu_list, sigma_list, times = [], [], []

    # Perform sliding window calculation for mu and sigma
    for start in range(0, len(data) - window_samples, step_samples):
        window = data[start : start + window_samples]
        mu_list.append(np.mean(window))
        sigma_list.append(np.std(window))
        times.append(tr_times[start])  # Directly use tr_times for relative time

    # Convert lists to arrays
    mu_array = np.array(mu_list)
    sigma_array = np.array(sigma_list)
    times = np.array(times)

    # Calculate dynamic threshold (mu + 6 * sigma)
    threshold = mu_array + 6 * sigma_array

    # Detect events where the signal exceeds the threshold
    events = []
    for i, t in enumerate(times):
        start = int(t * sampling_rate)
        end = start + window_samples
        event_indices = np.where(data[start:end] > threshold[i])[0] + start
        events.extend(event_indices)

    # Remove duplicates from detected events
    events = np.unique(events)

    # Convert detected events to on/off format with specified span
    on_off = []
    for event in events:
        rel_time = tr_times[event]  # Use tr_times to get relative event time
        on_off.append([rel_time, rel_time + span])

    return on_off


def daywise_plot():
    ## lunar
    basedir = "space_apps_2024_seismic_detection"
    data_directory = basedir + "/data/lunar/training/data/S12_GradeA/"
    fnlist = gridsearch.read_list_to_fnlist(os.path.join("algdev", "lunar_gtlist.txt"))
    figfnprefix = os.path.join("algdev_fig", "lunar_")
    meanlist = []
    stdlist = []
    sta_len, lta_len, thr_on, thr_off = [60, 560, 3.25, 0.75]
    figfn = "daywise_lunar.png"
    for idx, fn in enumerate(fnlist):
        subfnlist = [fn]
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
            tr_data = alltrdata[0]
            m, s = daywise_stat(tr_data)
            meanlist.append(m)
            stdlist.append(s)
        else:
            print("File Missing:", fn)

    # Initialize figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    idx = range(len(meanlist))
    # Plot meanlist and stdlist
    ax.plot(idx, meanlist, label="Mean", color="blue")
    ax.plot(idx, stdlist, label="Standard Deviation", color="red")
    ax.set_title(f"daywise std and mean", fontweight="bold")
    fig.savefig(figfn, dpi=300, bbox_inches="tight")
    return np.mean(stdlist)


def dygauss_exp():
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
                figfn = figfnprefix + str(idx) + f"_m{m}_dygauss.png"
                on_off = gaussian(tr, tr_data, window_size=3600, step_size=30, span=320)
                # merged_on_off = merge_intervals(on_off)
                arrivallist_in_blue = [x[0] for x in on_off]
                viewer.viewer_all(
                    tr, tr_data, figtitle, arrivallist_in_s, figfn, arrivallist_in_blue
                )
            else:
                print("File Missing:", fn)


def staticgauss_exp(goldsigma):
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
                figfn = figfnprefix + str(idx) + f"_m{m}_stgauss.png"
                on_off = fixedgaussian(
                    tr, tr_data, sigma=goldsigma, step_size=30, span=320
                )
                # merged_on_off = merge_intervals(on_off)
                arrivallist_in_blue = [x[0] for x in on_off]
                viewer.viewer_all(
                    tr, tr_data, figtitle, arrivallist_in_s, figfn, arrivallist_in_blue
                )
            else:
                print("File Missing:", fn)


if __name__ == "__main__":
    print("Perform statistics")
    goldsigma = daywise_plot()
    print("Perform dygauss_exp")
    dygauss_exp()
    print("Perform staticgauss_exp")
    staticgauss_exp(goldsigma)
