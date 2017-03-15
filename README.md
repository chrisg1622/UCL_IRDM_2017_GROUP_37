# UCL_IRDM_2017_GROUP_37
Our solution to the Home Depot Kaggle Competition - by Chris George, Said Kassim and Joao Reis 

End to end pipeline:

*Acquiring the data - Download directly from the Kaggle website https://www.kaggle.com/c/home-depot-product-search-relevance/data

*Data Cleaning - Run the 'Data Cleaning/HomeDepot_Data_Cleaning.py' script. This will read in the downloaded data from the 'input' folder, 
                 clean the data and save the cleaned data in 'input_clean'.

*Feature Extraction - Run the 'Feature Extraction/Feature_Extraction.py' script. This will read in the cleaned data from the 'input_clean' 
                      folder, extract a range of features such as tf-idf and bm25 variants, and save the features as 'X_train', 'X_test', 
                      and labels 'Y_train' in the 'input_clean' folder.

*Learning to Rank Models - Run any of the .py files in the home directory. Each model script will read in the features from 'input_clean',
                           run a 10-fold cross validation routine to showcase the performance of each model, and save the test predictions
                           in the 'output' folder. Models include OLS, Lasso, Ridge, KNN, SVM, Random Forests, XGBoost and Neural                                    Networks.
                      
