# PBT-project
This repository contains all the notebooks, code, and data needed to reproduce the results from our paper:\
\
**"Application of Deep Learning to Predict Persistence, Bioaccumulation and Toxicity of Pharmaceuticals"**  \
*Journal of Chemical Information and Modeling (ACS Publications), 2025*  \
*DOI: [https://doi.org/10.1021/acs.jcim.4c02293]*

The study explores the application of a deep-learning (DL) model to predict the Persistence, Bioaccumulation and Toxicity (PBT) of pharmaceuticals. We aim to discover substructures within molecules that are linked to PBT characteristics. By incorporating these findings into early-stage drug discovery, we can design drugs with reduced environmental impact.


The repository is organized as follow: 
 - Notebooks 
     - ***Dataset_compilation_and_Pre-processing*** reports the section Filtering and Pre-processing procedure  in the paper **"Application of Deep Learning to Predict Persistence, Bioaccumulation and Toxicity of Pharmaceuticals"**. It describes the dataset compilation used to train the DL-based model and the standardized procedure adopted to clean and filter it. 
     - ***Dataset partitioning*** illustates the section Dataset partitioning in the same  paper describing the splitting strategies adopter to train the DL-based model and their statistical validation with Kolmogorov-Smirnov (KS) test. 
     - ***Training_and_predicting_splitting_strategies.ipynb*** contains the entire training and prediction process of the DL-based models for the three different splitting strategies.
     - ***noaddfeatures_training_predicting.ipynb*** contains the training and prediction process without using the rdkit additional features of the DL-based models for the three different splitting strategies.
     - ***Applicability_Domain_Analysis*** describes t-Distributed Stochastic Neighbour Embedding (t-SNE) plots in Figure 2 of the main paper and the Applicability Domain (AD) of the DL-model.
     - ***Interpretability_analysis*** illustates the section Interpretability Analysis Application to Pharmaceuticals in the same paper describing the extraction of PBT-relevant substructures through chemprop built-in interpret function.
     - ***How_to_predict_PBT_with_your_dataset.ipynb*** provides guidance on how to make predictions of Persistence, Bioaccumulation and Toxicity (PBT) using the DL-based model proposed on your own dataset.
 - **Python Files:**
   - `applicability_domain.py`: Contains functions for analyzing the applicability domain using PCA and Mahalanobis distance
   - `chem_utils.py`: Contains utility function for chemical structure processing
   - `GP_QSPR.py`: Contains function for GP-QSPR analysis

 - **Datasets/**: contains datasets used in this study.
 - **Compiled_dataset/**: contains all the datasets collected from different sources in order to build our compiled dataset.
 - **Splitting_strategy_datasets/**: contains training and test sets obtained from our compiled dataset by splitting it according to three different strategies in order to train and test the DL-based models.
 - **kfold_outputs/**: contains all the outputs after training and predicting DL-based cross-validation models for the three splitting strategies.
 - **0fold_outputs/**: contains all the outputs after training and predicting DL-based models on all the data for the three splitting strategies. 
 - **outputs_preds_interpr/**: contains the final prediction output of the compiled dataset together with the interpretation output and PBT-related substructures obtained.
 - **images/**: contains the images that were generated through the notebooks.

The .csv files are the files generated in the notebooks.

The .yml files are the conda environments used to conduct the work.
