# causal-text
This is a package for computing the causal effects of text. Concretely this means algorithms for quantifying the degree of influence some user-defiend property (E.g. sentiment, respect) has on an outcome (email reply time), while controlling for potential confounds (topic, etc). 

## Quickstart

Follow these commands to run the full TextCause algorithm on a simulated dataset.

```
pip install -r requirements.txt
python main.py --run_cb
```

## Usage

* **Prepare your data.** This system expects a TSV file, with columns
  * `text`: string, the text you're studying
  * `Y`: int, binary outcome of interest
  * `C`: int, categorical confounder
  * `T_proxy`: int, your binary treatment indicator, e.g. the output of a classifier or lexicon
  * `T_true` (optional): int, binary indicator for the "true" (i.e. non-predicted) treatment
* **Run the system.** 
  * For `python main.py --data /path/to/your/data.tsv --no_simulate`
  * And if you want to run BERT for text adjustment (i.e. the full TextCause algorithm): 
     `python main.py --data /path/to/your/data.tsv --no_simulate --run_cb`
  * If you want to run the simulation: `python main.py --run_cb`
* **Look at your results.** When finished, the system will print out all of its hyperparameters and a bunch of different ATE estimates. The estimators are:
    * `unadj_T`: the unadjusted effect of T
    * `ate_T`: backdoor-adjusted effect of T
    * `unadj_T_proxy`: the unadjusted effect of T proxy
    * `ate_T_proxy`: backdoor-adjusted effect of T proxy
    * `ate_matrix`: matrix-adjusted effect of T_proxy using the measurement model P(T_hat | T)
    * `ate_T_plus_reg`: backdoor-adjusted effect of a boosted T
    * `ate_T_plus_pu`: backdoor-adjusted effect of a boosted T, but where the label improvement comes from a one-class classifier instead of a logistic regression
    * `ate_cb_T_proxy`: text-adjusted ATE estimate
    * `ate_cb_T_plus`: the full TextCause algorithm; test adjustment + T boosting.
    
