# A Simple Vision-Inspired Seismic Detection and Package

(Team Whitespace for SpaceApp 2024)
Welcome to our project! We are amateurs passionate about seismic detection and we’re open to suggestions. Our goal is to collaborate and learn with others who share the same interest in seismic detection.

## Slides 

[Link to Slides](./SV-WhiteSpace.pdf)

## Q and A (also in slides)

* Why Use a Background Model?
  * Object Detection Requires Specific Knowledge: We’d need to define quake characteristics, which we don’t have.
  * General Approach: Background models allow us to detect anomalies without specific quake properties.
  * Amateurs in Seismology: We’re still learning about quakes, so we prefer a simpler, flexible method.

* Why Not Use Deep Learning?
  * Limited Seismology Expertise: We use DL in vision, but applying it here without enough knowledge could lead to bad results.
  * Noise Modeling: DL might struggle without proper noise understanding.

## Output

* Please look at the folder `output/`
* Generate test sequences: `python gaussian_test.py`

## Please Note before Start (not included in slides)

* We found that the training data contains very few quakes, e.g., lunar == 76 and Mars == 2.
* Based on machine learning principles, we cannot directly use the training data for validation. However, we still split the training data into training and validation sets to correctly develop our method.
* How to select a reference STA/LTA algorithm?
    * A demo script space_apps_2024_seismic_detection/demo_notebook.ipynb uses [120, 600, 4.0, 1.5]. We think this is a good starting point for further study.
* Detection can be measured using recall and precision. Here, we focus on recall and the false positive rate.
    * In object detection for computer vision, you might use TPR (True Positive Rate) and FPR (False Positive Rate) to generate an ROC curve and calculate the AUC, which considers both recall and false positives.
    * Alternatively, you might want to set an acceptable FPR, e.g., 0.1% or 1%. However, we are unsure what the acceptable FPR is for seismic detection.
* How do we define a "hit"?
    * In the sample found in space_apps_2024_seismic_detection/demo_notebook.ipynb, we observed differences of more than three minutes between predictions and the ground truth. Therefore, we use a tolerance of 320 seconds between predictions and ground truth. However, in our code, we calculate recall at 8 different tolerance levels. Please refer to the eval_curves function in eval.py for more details.

    ```
    def eval_curves(csvgt, csvpred, verbose=False):
        tol = []
        recall = []
        fp = []
        for expo in range(8):
            tols = (2**expo) * 5
    ```
* After tuning the STA/LTA algorithm, we found that some sharp changes in the signal are hard to remove. We cannot set the LTA too large. Therefore, we expect Gaussian background modeling to provide better long-term statistics.

* We also put some fig in `algdev_fig/` for reference. Maybe remove in the future.

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
* filename field in mars training data could include extension, e.g. csv.

## References:

[1] Space Apps 2024 Seismic Detection Data Packet.

[2] `space_apps_2024_seismic_detection/demo_notebook.ipynb`

[3] https://www.geopsy.org/wiki/index.php/Geopsy:_STA/LTA
