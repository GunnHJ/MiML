# MiML (multiple imputation machine learning) project
Code and data used for "How to Apply Variable Selection Machine Learning Algorithms with Multiply Imputed Data:  A Missing Discussion"

# Description of project

Use an applied example to illustrate how to fit a LASSO when using listwise deletion
or multiple imputation to handle the missing data. We evaluated a separate approach, stacked
approach, and MI-LASSO (Chen & Wang, 2013).

# Files

baselineDataBlimp_0907.csv: raw data set used in Blimp program to impute missing values
Multiple Imputation (Blimp): Blimp syntax file
imps_stacked.dat: stacked set of raw data set and imputed data sets
PSRs_with_Labels: data set that contains the PSR values from the imputation process
LASSOs.R: syntax file to fit the LASSOs for all 4 approaches and calculate descriptive statistics
