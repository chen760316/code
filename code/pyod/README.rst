----

Baseline experiments
^^^^^^^^^^^^

**The source code for all benchmark experiments is located in the pyod/normal_experiments folder. The new experiments are located in the pyod/normal_experiments_add folder. Our baseline tests rely on the open-source pyod library.**
We split each dataset into 70% training, 20% validation, and 10% testing. The training set was used to learn models and OMR rules; the validation set guided hyperparameter tuning and cleaning strategy selection via Bayesian optimization; 
and the test set was reserved only for final evaluation of accuracy and robustness.

**The command to run is**:

.. code-block:: bash

   python pyod/normal_experiments/**.py



**Outlier Detection with 5 Lines of Code**:

.. code-block:: python

    # Example: Training an ECOD detector
    from pyod.models.ecod import ECOD
    clf = ECOD()
    clf.fit(X_train)
    y_train_scores = clf.decision_scores_  # Outlier scores for training data
    y_test_scores = clf.decision_function(X_test)  # Outlier scores for test data


**Selecting the Right Algorithm:** Unsure where to start? Consider these robust and interpretable options:

- `ECOD <https://github.com/yzhao062/pyod/blob/master/examples/ecod_example.py>`_: Example of using ECOD for outlier detection
- `Isolation Forest <https://github.com/yzhao062/pyod/blob/master/examples/iforest_example.py>`_: Example of using Isolation Forest for outlier detection

Alternatively, explore `MetaOD <https://github.com/yzhao062/MetaOD>`_ for a data-driven approach.

----

Installation
^^^^^^^^^^^^

PyOD is designed for easy installation using either **pip** or **conda**. We recommend using the latest version of PyOD due to frequent updates and enhancements:

.. code-block:: bash

   pip install pyod            # normal install
   pip install --upgrade pyod  # or update if needed

.. code-block:: bash

   conda install -c conda-forge pyod

Alternatively, you can clone and run the setup.py file:

.. code-block:: bash

   git clone https://github.com/yzhao062/pyod.git
   cd pyod
   pip install .

**Required Dependencies**:

* Python 3.8 or higher
* joblib
* matplotlib
* numpy>=1.19
* numba>=0.51
* scipy>=1.5.1
* scikit_learn>=0.22.0

**Optional Dependencies (see details below)**:

* combo (optional, required for models/combination.py and FeatureBagging)
* pytorch (optional, required for AutoEncoder, and other deep learning models)
* suod (optional, required for running SUOD model)
* xgboost (optional, required for XGBOD)
* pythresh (optional, required for thresholding)

----


API Cheatsheet & Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^

The full API Reference is available at `PyOD Documentation <https://pyod.readthedocs.io/en/latest/pyod.html>`_. Below is a quick cheatsheet for all detectors:

* **fit(X)**: Fit the detector. The parameter y is ignored in unsupervised methods.
* **decision_function(X)**: Predict raw anomaly scores for X using the fitted detector.
* **predict(X)**: Determine whether a sample is an outlier or not as binary labels using the fitted detector.
* **predict_proba(X)**: Estimate the probability of a sample being an outlier using the fitted detector.
* **predict_confidence(X)**: Assess the model's confidence on a per-sample basis (applicable in predict and predict_proba) [#Perini2020Quantifying]_.

**Key Attributes of a fitted model**:

* **decision_scores_**: Outlier scores of the training data. Higher scores typically indicate more abnormal behavior. Outliers usually have higher scores.
* **labels_**: Binary labels of the training data, where 0 indicates inliers and 1 indicates outliers/anomalies.

----

Model Save & Load
^^^^^^^^^^^^^^^^^

PyOD takes a similar approach of sklearn regarding model persistence.
See `model persistence <https://scikit-learn.org/stable/modules/model_persistence.html>`_ for clarification.

In short, we recommend to use joblib or pickle for saving and loading PyOD models.
See `"examples/save_load_model_example.py" <https://github.com/yzhao062/pyod/blob/master/examples/save_load_model_example.py>`_ for an example.
In short, it is simple as below:

.. code-block:: python

    from joblib import dump, load

    # save the model
    dump(clf, 'clf.joblib')
    # load the model
    clf = load('clf.joblib')

It is known that there are challenges in saving neural network models.
Check `#328 <https://github.com/yzhao062/pyod/issues/328#issuecomment-917192704>`_
and `#88 <https://github.com/yzhao062/pyod/issues/88#issuecomment-615343139>`_
for temporary workaround.


----


Fast Train with SUOD
^^^^^^^^^^^^^^^^^^^^

**Fast training and prediction**: it is possible to train and predict with
a large number of detection models in PyOD by leveraging SUOD framework [#Zhao2021SUOD]_.
See  `SUOD Paper <https://www.andrew.cmu.edu/user/yuezhao2/papers/21-mlsys-suod.pdf>`_
and  `SUOD example <https://github.com/yzhao062/pyod/blob/master/examples/suod_example.py>`_.


.. code-block:: python

    from pyod.models.suod import SUOD

    # initialized a group of outlier detectors for acceleration
    detector_list = [LOF(n_neighbors=15), LOF(n_neighbors=20),
                     LOF(n_neighbors=25), LOF(n_neighbors=35),
                     COPOD(), IForest(n_estimators=100),
                     IForest(n_estimators=200)]

    # decide the number of parallel process, and the combination method
    # then clf can be used as any outlier detection model
    clf = SUOD(base_estimators=detector_list, n_jobs=2, combination='average',
               verbose=False)

----

Thresholding Outlier Scores
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A more data-based approach can be taken when setting the contamination level. By using a thresholding method, guessing an arbitrary value can be replaced with tested techniques for separating inliers and outliers. Refer to `PyThresh <https://github.com/KulikDM/pythresh>`_ for a more in-depth look at thresholding.

.. code-block:: python

    from pyod.models.knn import KNN
    from pyod.models.thresholds import FILTER

    # Set the outlier detection and thresholding methods
    clf = KNN(contamination=FILTER())


See supported thresholding methods in `thresholding <https://github.com/yzhao062/pyod/blob/master/docs/thresholding.rst>`_.

----



Implemented Algorithms
^^^^^^^^^^^^^^^^^^^^^^

PyOD toolkit consists of four major functional groups:

**(i) Individual Detection Algorithms** :

===================  ==================  ======================================================================================================  =====  ========================================
Type                 Abbr                Algorithm                                                                                               Year   Ref
===================  ==================  ======================================================================================================  =====  ========================================
Probabilistic        ECOD                Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions                        2022   [#Li2021ECOD]_
Probabilistic        ABOD                Angle-Based Outlier Detection                                                                           2008   [#Kriegel2008Angle]_
Probabilistic        FastABOD            Fast Angle-Based Outlier Detection using approximation                                                  2008   [#Kriegel2008Angle]_
Probabilistic        COPOD               COPOD: Copula-Based Outlier Detection                                                                   2020   [#Li2020COPOD]_
Probabilistic        MAD                 Median Absolute Deviation (MAD)                                                                         1993   [#Iglewicz1993How]_
Probabilistic        SOS                 Stochastic Outlier Selection                                                                            2012   [#Janssens2012Stochastic]_
Probabilistic        QMCD                Quasi-Monte Carlo Discrepancy outlier detection                                                         2001   [#Fang2001Wrap]_
Probabilistic        KDE                 Outlier Detection with Kernel Density Functions                                                         2007   [#Latecki2007Outlier]_
Probabilistic        Sampling            Rapid distance-based outlier detection via sampling                                                     2013   [#Sugiyama2013Rapid]_
Probabilistic        GMM                 Probabilistic Mixture Modeling for Outlier Analysis                                                            [#Aggarwal2015Outlier]_ [Ch.2]
Linear Model         PCA                 Principal Component Analysis (the sum of weighted projected distances to the eigenvector hyperplanes)   2003   [#Shyu2003A]_
Linear Model         KPCA                Kernel Principal Component Analysis                                                                     2007   [#Hoffmann2007Kernel]_
Linear Model         MCD                 Minimum Covariance Determinant (use the mahalanobis distances as the outlier scores)                    1999   [#Hardin2004Outlier]_ [#Rousseeuw1999A]_
Linear Model         CD                  Use Cook's distance for outlier detection                                                               1977   [#Cook1977Detection]_
Linear Model         OCSVM               One-Class Support Vector Machines                                                                       2001   [#Scholkopf2001Estimating]_
Linear Model         LMDD                Deviation-based Outlier Detection (LMDD)                                                                1996   [#Arning1996A]_
Proximity-Based      LOF                 Local Outlier Factor                                                                                    2000   [#Breunig2000LOF]_
Proximity-Based      COF                 Connectivity-Based Outlier Factor                                                                       2002   [#Tang2002Enhancing]_
Proximity-Based      (Incremental) COF   Memory Efficient Connectivity-Based Outlier Factor (slower but reduce storage complexity)               2002   [#Tang2002Enhancing]_
Proximity-Based      CBLOF               Clustering-Based Local Outlier Factor                                                                   2003   [#He2003Discovering]_
Proximity-Based      LOCI                LOCI: Fast outlier detection using the local correlation integral                                       2003   [#Papadimitriou2003LOCI]_
Proximity-Based      HBOS                Histogram-based Outlier Score                                                                           2012   [#Goldstein2012Histogram]_
Proximity-Based      kNN                 k Nearest Neighbors (use the distance to the kth nearest neighbor as the outlier score)                 2000   [#Ramaswamy2000Efficient]_
Proximity-Based      AvgKNN              Average kNN (use the average distance to k nearest neighbors as the outlier score)                      2002   [#Angiulli2002Fast]_
Proximity-Based      MedKNN              Median kNN (use the median distance to k nearest neighbors as the outlier score)                        2002   [#Angiulli2002Fast]_
Proximity-Based      SOD                 Subspace Outlier Detection                                                                              2009   [#Kriegel2009Outlier]_
Proximity-Based      ROD                 Rotation-based Outlier Detection                                                                        2020   [#Almardeny2020A]_
Outlier Ensembles    IForest             Isolation Forest                                                                                        2008   [#Liu2008Isolation]_
Outlier Ensembles    INNE                Isolation-based Anomaly Detection Using Nearest-Neighbor Ensembles                                      2018   [#Bandaragoda2018Isolation]_
Outlier Ensembles    DIF                 Deep Isolation Forest for Anomaly Detection                                                             2023   [#Xu2023Deep]_
Outlier Ensembles    FB                  Feature Bagging                                                                                         2005   [#Lazarevic2005Feature]_
Outlier Ensembles    LSCP                LSCP: Locally Selective Combination of Parallel Outlier Ensembles                                       2019   [#Zhao2019LSCP]_
Outlier Ensembles    XGBOD               Extreme Boosting Based Outlier Detection **(Supervised)**                                               2018   [#Zhao2018XGBOD]_
Outlier Ensembles    LODA                Lightweight On-line Detector of Anomalies                                                               2016   [#Pevny2016Loda]_
Outlier Ensembles    SUOD                SUOD: Accelerating Large-scale Unsupervised Heterogeneous Outlier Detection **(Acceleration)**          2021   [#Zhao2021SUOD]_
Neural Networks      AutoEncoder         Fully connected AutoEncoder (use reconstruction error as the outlier score)                                    [#Aggarwal2015Outlier]_ [Ch.3]
Neural Networks      VAE                 Variational AutoEncoder (use reconstruction error as the outlier score)                                 2013   [#Kingma2013Auto]_
Neural Networks      Beta-VAE            Variational AutoEncoder (all customized loss term by varying gamma and capacity)                        2018   [#Burgess2018Understanding]_
Neural Networks      SO_GAAL             Single-Objective Generative Adversarial Active Learning                                                 2019   [#Liu2019Generative]_
Neural Networks      MO_GAAL             Multiple-Objective Generative Adversarial Active Learning                                               2019   [#Liu2019Generative]_
Neural Networks      DeepSVDD            Deep One-Class Classification                                                                           2018   [#Ruff2018Deep]_
Neural Networks      AnoGAN              Anomaly Detection with Generative Adversarial Networks                                                  2017   [#Schlegl2017Unsupervised]_
Neural Networks      ALAD                Adversarially learned anomaly detection                                                                 2018   [#Zenati2018Adversarially]_
Neural Networks      AE1SVM              Autoencoder-based One-class Support Vector Machine                                                      2019   [#Nguyen2019scalable]_
Neural Networks      DevNet              Deep Anomaly Detection with Deviation Networks                                                          2019   [#Pang2019Deep]_
Graph-based          R-Graph             Outlier detection by R-graph                                                                            2017   [#You2017Provable]_
Graph-based          LUNAR               LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks                               2022   [#Goodge2022Lunar]_
===================  ==================  ======================================================================================================  =====  ========================================


**(ii) Outlier Ensembles & Outlier Detector Combination Frameworks**:

===================  ================  =====================================================================================================  =====  ========================================
Type                 Abbr              Algorithm                                                                                              Year   Ref
===================  ================  =====================================================================================================  =====  ========================================
Outlier Ensembles    FB                Feature Bagging                                                                                        2005   [#Lazarevic2005Feature]_
Outlier Ensembles    LSCP              LSCP: Locally Selective Combination of Parallel Outlier Ensembles                                      2019   [#Zhao2019LSCP]_
Outlier Ensembles    XGBOD             Extreme Boosting Based Outlier Detection **(Supervised)**                                              2018   [#Zhao2018XGBOD]_
Outlier Ensembles    LODA              Lightweight On-line Detector of Anomalies                                                              2016   [#Pevny2016Loda]_
Outlier Ensembles    SUOD              SUOD: Accelerating Large-scale Unsupervised Heterogeneous Outlier Detection **(Acceleration)**         2021   [#Zhao2021SUOD]_
Outlier Ensembles    INNE              Isolation-based Anomaly Detection Using Nearest-Neighbor Ensembles                                     2018   [#Bandaragoda2018Isolation]_
Combination          Average           Simple combination by averaging the scores                                                             2015   [#Aggarwal2015Theoretical]_
Combination          Weighted Average  Simple combination by averaging the scores with detector weights                                       2015   [#Aggarwal2015Theoretical]_
Combination          Maximization      Simple combination by taking the maximum scores                                                        2015   [#Aggarwal2015Theoretical]_
Combination          AOM               Average of Maximum                                                                                     2015   [#Aggarwal2015Theoretical]_
Combination          MOA               Maximization of Average                                                                                2015   [#Aggarwal2015Theoretical]_
Combination          Median            Simple combination by taking the median of the scores                                                  2015   [#Aggarwal2015Theoretical]_
Combination          majority Vote     Simple combination by taking the majority vote of the labels (weights can be used)                     2015   [#Aggarwal2015Theoretical]_
===================  ================  =====================================================================================================  =====  ========================================


**(iii) Utility Functions**:

===================  ======================  =====================================================================================================================================================  ======================================================================================================================================
Type                 Name                    Function                                                                                                                                               Documentation
===================  ======================  =====================================================================================================================================================  ======================================================================================================================================
Data                 generate_data           Synthesized data generation; normal data is generated by a multivariate Gaussian and outliers are generated by a uniform distribution                  `generate_data <https://pyod.readthedocs.io/en/latest/pyod.utils.html#module-pyod.utils.data.generate_data>`_
Data                 generate_data_clusters  Synthesized data generation in clusters; more complex data patterns can be created with multiple clusters                                              `generate_data_clusters <https://pyod.readthedocs.io/en/latest/pyod.utils.html#pyod.utils.data.generate_data_clusters>`_
Stat                 wpearsonr               Calculate the weighted Pearson correlation of two samples                                                                                              `wpearsonr <https://pyod.readthedocs.io/en/latest/pyod.utils.html#module-pyod.utils.stat_models.wpearsonr>`_
Utility              get_label_n             Turn raw outlier scores into binary labels by assign 1 to top n outlier scores                                                                         `get_label_n <https://pyod.readthedocs.io/en/latest/pyod.utils.html#module-pyod.utils.utility.get_label_n>`_
Utility              precision_n_scores      calculate precision @ rank n                                                                                                                           `precision_n_scores <https://pyod.readthedocs.io/en/latest/pyod.utils.html#module-pyod.utils.utility.precision_n_scores>`_
===================  ======================  =====================================================================================================================================================  ======================================================================================================================================

----

Quick Start for Outlier Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#. Initialize a kNN detector, fit the model, and make the prediction.

   .. code-block:: python


       from pyod.models.knn import KNN   # kNN detector

       # train kNN detector
       clf_name = 'KNN'
       clf = KNN()
       clf.fit(X_train)

       # get the prediction label and outlier scores of the training data
       y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
       y_train_scores = clf.decision_scores_  # raw outlier scores

       # get the prediction on the test data
       y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
       y_test_scores = clf.decision_function(X_test)  # outlier scores

       # it is possible to get the prediction confidence as well
       y_test_pred, y_test_pred_confidence = clf.predict(X_test, return_confidence=True)  # outlier labels (0 or 1) and confidence in the range of [0,1]

#. Evaluate the prediction by ROC and Precision @ Rank n (p@n).

   .. code-block:: python

       from pyod.utils.data import evaluate_print

       # evaluate and print the results
       print("\nOn Training Data:")
       evaluate_print(clf_name, y_train, y_train_scores)
       print("\nOn Test Data:")
       evaluate_print(clf_name, y_test, y_test_scores)

