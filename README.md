# MiML (multiple imputation machine learning) project
Code and data used for "How to Apply Variable Selection Machine Learning Algorithms with Multiply Imputed Data:  A Missing Discussion"

# Description of project

This tutorial walks through how to fit a LASSO when using listwise deletion or multiple imputation to handle the missing data. We used an applied example from the ATN CARES to illustrate the separate approach, stacked approach, MI-LASSO (Chen & Wang, 2013), and listwise deletion.

# Files

baselineDataBlimp_0907.csv: raw data set used in Blimp program to impute missing values

Multiple Imputation (Blimp): Blimp syntax file

PSRs_with_Labels: data set created by Blimp that contains the PSR values from the imputation process

LASSOs.R: syntax file to fit the LASSOs for all 4 approaches and calculate descriptive statistics (need Blimp to create imps_stacked.dat)
