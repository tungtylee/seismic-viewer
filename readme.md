# Space App 2024

## Data Processing and Basic Utility

* Unzip `space_apps_2024_seismic_detection.zip`
* Split training data to train and val
* Baseline algorithm: sta/lta (sta_len=120, lta_len=600, thr_on=4, thr_off=1.5)
* Eval csvgt and csvpred, eval at different tolerant seconds

## Exploration and Study

* Grid search over the train (val is not included)
* Analyze results from the process of grid search
* View all four candidate stalta settings: `python viewer.py`
* Run dynamic gaussian and static gaussian: `python gaussian.py`

## Some problems

* lunar training data
    * Exists: xa.s12.00.mhz.1971-04-13HR02_evid00029.mseed
    * Not found: xa.s12.00.mhz.1971-04-13HR00_evid00029.mseed

## References:

[1] Space Apps 2024 Seismic Detection Data Packet.

[2] `space_apps_2024_seismic_detection/demo_notebook.ipynb`

[3] https://www.geopsy.org/wiki/index.php/Geopsy:_STA/LTA