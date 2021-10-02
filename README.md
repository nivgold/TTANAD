# Test-Time Augmentation for Network Anomaly Detection
The official code of the paper "Test-Time Augmentation for Network Anomaly Detection".


![proposed framework](https://raw.githubusercontent.com/nivgold/TTANAD/main/fig.png)

## Abstract

> Machine learning-based Network Intrusion Detection Systems (NIDS) are designed to protect the network by identifying anomalous behaviors or improper uses. In recent years, advanced attacks have been adapted to mimic as legitimate traffic to avoid alerting such systems. Previous works mainly focused on modeling, i.e., improving the anomaly detector; in this paper, we introduce a novel method, Test-Time Augmentation for Network Anomaly Detection (TTANAD), which utilizes test-time augmentation to improve anomaly detection from the data side. TTANAD takes advantage of the temporal characteristics of the traffic data and produces temporal test-time augmentations on the monitored traffic data. This method aims to create additional points of view when examining the network traffic on inference. Our experimental results demonstrate better performance when using TTANAD.

## Datasets

As described in the paper, the experiments were conducted using the two well-known datasets CIC-IDS2017, CSE-CIC-IDS2018 from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/): 

* **CIC-IDS2017** - includes eight different files containing five days' normal and malicious traffic data. Combining these files results in roughly 3 million instances and 83 features with 15 labels - 1 normal and 14 attack labels.

* **CSE-CIC-IDS2018** - contains about 16 million instances collected over ten days, with roughly 17% of the instances compromised of malicious traffic.


## Dependencies

The required dependencies are specified in `environment.yml`.

For setting up the environment, use [Anaconda](https://www.anaconda.com/):
```bash
$ conda env create -f environment.yml
$ conda activate adtta
```

## Running the Code

**NOTES:**

- A valid dataset name can be only one from the described earlier (either *ids17* or *ids18*).
- It is important to run the scripts from being inside `src/` (i.e. ```$ cd src/```)

---

* ### **Preprocessing**
        
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A preprocessing phase on a desired dataset. This should be done before training and testing.

```bash
$ cd src/
$ python preprocess.py --dataset ids18 --windowsize 5
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The output from this run is a .npz file that is going to be generated in the desired dataset's `data/` folder:

&emsp;&emsp;&emsp;&emsp;- `ids18_preprocessed_5_window_size.npz`


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;With this file, the data_loader can now operate, which is essential for training and testing.

---

* ### **Train & Test**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To train the anomaly detector:

```bash
$ cd src/
$ python main.py --dataset id18 --windowsize 5
```
