# UCL_IRDM_2017_GROUP_37
Our solution to the Home Depot Kaggle Competition - by Chris George, Said Kassim and Joao Reis 

End to end pipeline:

*Acquiring the data - Download directly from the Kaggle website https://www.kaggle.com/c/home-depot-product-search-relevance/data

*Data Exploration - Run the 'data_exploration.R' script to view our data exploration plots.

*Data Cleaning - Run the 'Data Cleaning/HomeDepot_Data_Cleaning.py' script. This will read in the downloaded data from the 'input' folder, 
                 clean the data and save the cleaned data in 'input_clean'.

*Feature Extraction - Run the 'Feature Extraction/Feature_Extraction.py' script. This will read in the cleaned data from the 'input_clean' 
                      folder, extract a range of features such as tf-idf and bm25 variants, and save the features as 'X_train', 'X_test', 
                      and labels 'Y_train' in the 'input_clean' folder.

*Learning to Rank Models - Run any of the .py files in the home directory which contain model names. Each model script will read in the                              features from 'input_clean', run a 10-fold cross validation routine to showcase the performance of each model,                            and save the test predictions in the 'output' folder. Models include OLS, Lasso, Ridge, KNN, SVM, Random                                  Forests, XGBoost, Neural Networks and a weighted average ensemble. We have also included the scripts we wrote                              to optimize each model, when optimization is required.
               
 *Model Evaluation - Several metrics were implemented in order to evaluate the performance of our models using different criteria. The code for these can be found in the 'Metrics.py' file.
 
 ------ Additional Scripts ------
 *CVPlots - This reads in the cross-validation scores from each model (saved in the output folder after running the model scripts), and               plots the cross validation scores.
 
 *FeatureImportance - This reads in the train features, splits the data by relevance (\<2, \>=2) to form two sets, computes the relative                         difference between the means of each feature for both sets. Using this relative difference as a feature                                 importance criterion, the script then plots the 3 most and least important features.
 
 *PlotKaggleScores -  This script plots the kaggle submission scores for each of our models, ordered by performance.
 
 *PostProcessingTest - This script plots the improvement in RMSE, MAE, MSE, R2 and Explained Variance achieved on a baseline model by                            post-processing predictions
 
 *LinearRegressionBaseline - This script plots the effect of our data cleaning and feature extraction on a baseline model.
 
 *closest - this contains a function which can be used for post processing predictions. Given a list of possible predictions (e.g. 30                 distinct values from 1 to 3), and a prediction (in the reals), the function converts the prediction into the list item closest             to it.
