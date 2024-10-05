import numpy as np
from obspy.signal.filter import highpass
from obspy.signal.invsim import cosine_taper
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset


def stalta(tr, tr_data, sta_len=120, lta_len=600, thr_on=4, thr_off=1.5):
    # Sampling frequency of our trace
    df = tr.stats.sampling_rate
    # print(df)

    # How long should the short-term and long-term window be, in seconds?
    # sta_len = 120
    # lta_len = 600

    # Run Obspy's STA/LTA to obtain a characteristic function
    # This function basically calculates the ratio of amplitude between the short-term 
    # and long-term windows, moving consecutively in time across the data
    cft = classic_sta_lta(tr_data, int(sta_len * df), int(lta_len * df))

    # Plot characteristic function
    # fig,ax = plt.subplots(1,1,figsize=(12,3))
    # ax.plot(tr_times,cft)
    # ax.set_xlim([min(tr_times),max(tr_times)])
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Characteristic function')

    # Play around with the on and off triggers, based on values in the characteristic function
    # thr_on = 4
    # thr_off = 1.5
    # trigger is utility function provided by obspy
    on_off = np.array(trigger_onset(cft, thr_on, thr_off))
    # The first column contains the indices where the trigger is turned "on". 
    # The second column contains the indices where the trigger is turned "off".

    # Plot on and off triggers
    # fig,ax = plt.subplots(1,1,figsize=(12,3))
    # for i in np.arange(0,len(on_off)):
    #     triggers = on_off[i]
    #     ax.axvline(x = tr_times[triggers[0]], color='red', label='Trig. On')
    #     ax.axvline(x = tr_times[triggers[1]], color='purple', label='Trig. Off')

    # Plot seismogram
    # ax.plot(tr_times,tr_data)
    # ax.set_xlim([min(tr_times),max(tr_times)])
    # ax.legend()

    return on_off