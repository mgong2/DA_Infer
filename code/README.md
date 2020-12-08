This experimental code provides simulation results in Table 1.

## Requirements
Python 3.6
Pytorch >= 1.0.0

## Training

To train the model(s) in the paper, run these commands:
1. train LV-CGAN networks according to the graph
```
bash run_jobs_train.sh
```
2. Infer target domain labels
```infer
bash run_jobs_infer_bayesian.sh
```

## Evaluation
Run the following script to collect the results.
```summary
python summarize_results.py
```

## Results
See table 2 in the main paper.
