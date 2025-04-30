### Baseline experiments

**The source code for part baseline experiments is located in the `adbench/normal_experiments folder`.The new experiments are located in the `adbench/normal_experiments_add` folder.**

We split each dataset into 70% training, 20% validation, and 10% testing. The training set was used to learn models and OMR rules; the validation set guided hyperparameter tuning and cleaning strategy selection via Bayesian optimization; and the test set was reserved only for final evaluation of accuracy and robustness.

The command to run is:

```
python adbench/normal_experiments/**.py
```

### Installation
```python
pip install adbench
pip install --upgrade adbench
```

_Prerequisite: Downloading datasets in ADBench from the github repo_
```python
from adbench.myutils import Utils
utils = Utils() # utility function
# download datasets from the remote github repo
# we recommend jihulab for China mainland user and github otherwise
utils.download_datasets(repo='jihulab')
```


**_Run Entire Experiments of ADBench_**

```python
from adbench.run import RunPipeline

'''
Params:
suffix: file name suffix;

parallel: running either 'unsupervise', 'semi-supervise', or 'supervise' (AD) algorithms,
corresponding to the Angle I: Availability of Ground Truth Labels (Supervision);

realistic_synthetic_mode: testing on 'local', 'global', 'dependency', and 'cluster' anomalies, 
corresponding to the Angle II: Types of Anomalies;

noise type: evaluating algorithms on 'duplicated_anomalies', 'irrelevant_features' and 'label_contamination',
corresponding to the Angle III: Model Robustness with Noisy and Corrupted Data.
'''

# return the results including [params, model_name, metrics, time_fit, time_inference]
# besides, results will be automatically saved in the dataframe and ouputted as csv file in adbench/result folder
pipeline = RunPipeline(suffix='ADBench', parallel='semi-supervise', realistic_synthetic_mode=None, noise_type=None)
results = pipeline.run()

pipeline = RunPipeline(suffix='ADBench', parallel='unsupervise', realistic_synthetic_mode='cluster', noise_type=None)
results = pipeline.run()

pipeline = RunPipeline(suffix='ADBench', parallel='supervise', realistic_synthetic_mode=None, noise_type='irrelevant_features')
results = pipeline.run()
```

**_Run Your Customized Algorithms on either ADBench Datasets or Your Customized Dataset_**
```python
# customized model on ADBench's datasets
from adbench.run import RunPipeline
from adbench.baseline.Customized.run import Customized

# notice that you should specify the corresponding category of your customized AD algorithm
# for example, here we use Logistic Regression as customized clf, which belongs to the supervised algorithm
# for your own algorithm, you can realize the same usage as other baselines by modifying the fit.py, model.py, and run.py files in the adbench/baseline/Customized
pipeline = RunPipeline(suffix='ADBench', parallel='supervise', realistic_synthetic_mode=None, noise_type=None)
results = pipeline.run(clf=Customized)

# customized model on customized dataset
import numpy as np
dataset = {}
dataset['X'] = np.random.randn(1000, 20)
dataset['y'] = np.random.choice([0, 1], 1000)
results = pipeline.run(dataset=dataset, clf=Customized)
```

We have unified all the datasets in .npz format, and you can directly access a dataset by the following script

```python
import numpy as np
data = np.load('adbench/datasets/Classical/6_cardio.npz', allow_pickle=True)
X, y = data['X'], data['y']
```

