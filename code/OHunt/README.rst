OHunt
~~~~~~~~~~~~~~
OHunt's ML-based outlier detector is implemented based on DeepOD.
The folders under the OHunt root directory correspond to the implementation of various components involved in OHunt,
including but not limited to various OMR predicates.
The OHunt/normal_experiment directory contains the implementation of experiments, rule discovery, and the fix process.
We plan to modularize OHunt and integrate it into a Python open-source library in the future.
The new experiments are located in the Ohunt/normal_exp_val folder.
We split each dataset into 70% training, 20% validation, and 10% testing. The training set was used to learn models and OMR
rules; the validation set guided hyperparameter tuning and cleaning strategy selection via Bayesian optimization; 
and the test set was reserved only for final evaluation of accuracy and robustness.

**The Python environment installation command is**:

.. code-block:: bash


    pip install -r OHunt/requirements.txt

**The command to run is**:

.. code-block:: bash


    python OHunt/normal_experiments/**.py

Installation
~~~~~~~~~~~~~~
The DeepOD framework can be installed via:


.. code-block:: bash


    pip install deepod


install a developing version (strongly recommend)


.. code-block:: bash


    git clone https://github.com/xuhongzuo/DeepOD.git
    cd DeepOD
    pip install .


Usages
~~~~~~~~~~~~~~~~~


Directly use detection models:
::::::::::::::::::::::::::::::::::::::::::

Outlier detectors can be used in a few lines of code.


**for tabular outlier detection:**

.. code-block:: python


    # unsupervised methods
    from deepod.models.tabular import DeepSVDD
    clf = DeepSVDD()
    clf.fit(X_train, y=None)
    scores = clf.decision_function(X_test)

    # weakly-supervised methods
    from deepod.models.tabular import DevNet
    clf = DevNet()
    clf.fit(X_train, y=semi_y) # semi_y uses 1 for known anomalies, and 0 for unlabeled data
    scores = clf.decision_function(X_test)

    # evaluation of tabular anomaly detection
    from deepod.metrics import tabular_metrics
    auc, ap, f1 = tabular_metrics(y_test, scores)



Testbed usage:
::::::::::::::::::::::::::::::::::::::::::


Testbed contains the whole process of testing an outlier detection model, including data loading, preprocessing, anomaly detection, and evaluation.

Please refer to ``testbed/``

* ``testbed/testbed_unsupervised_ad.py`` is for testing unsupervised tabular anomaly detection models.
 
* ``testbed/testbed_unsupervised_tsad.py`` is for testing unsupervised time-series anomaly detection models.


Key arguments:

* ``--input_dir``: name of the folder that contains datasets (.csv, .npy)

* ``--dataset``: "FULL" represents testing all the files within the folder, or a list of dataset names using commas to split them (e.g., "10_cover*,20_letter*")

* ``--model``: anomaly detection model name

* ``--runs``: how many times running the detection model, finally report an average performance with standard deviation values


Example: 

1. Download outlier detection datasets.
2. modify the ``dataset_root`` variable as the directory of the dataset.
3. ``input_dir`` is the sub-folder name of the ``dataset_root``, e.g., ``Classical`` or ``NLP_by_BERT``.  
4. use the following command in the bash


.. code-block:: bash

    
    cd DeepOD
    pip install .
    cd testbed
    python testbed_unsupervised_ad.py --model DeepIsolationForest --runs 5 --input_dir ADBench
   


