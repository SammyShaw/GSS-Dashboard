#### GSS trends table
import os 
import pyreadstat
import pandas as pd
import numpy as np
import pingouin as pg
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt

os.chdir("C:/Dat_Sci/Data Projects/GSS/dashboard project")

## Short save ############################################
indexes = pd.read_csv("dashboard_raw.csv")

""" 
For index measures, perform:
a. reliability analysis, 
b. Factor Analysis, and 
c. sensitivity test for minimum number of variables per row. 
"""


##########################################################
# Confidence in Insitutions
##########################################################

confidence_vars = ["coneduc", "confed", "conmedic", "conarmy", "conbus",
                   "conclerg", "confinan", "conjudge", "conlabor", "conlegis",
                   "conpress", "consci", "contv"]

# A. Reliability 
################
# First, Alpha analysis shows inter-item reliability 
confidence_columns = indexes[confidence_vars].copy()

reliability = pg.cronbach_alpha(data = confidence_columns)
print(reliability) # Alpha = .79

years = indexes["YEAR"].unique()
for year in sorted(years):
    inst_conf_year = indexes[indexes["YEAR"] == year][confidence_vars]
    alpha, ci = pg.cronbach_alpha(data = inst_conf_year)
    print(year, alpha, ci)


# B. Factor Analysis 
####################
fa = FactorAnalyzer(n_factors=3, rotation="varimax")  # Adjust n_factors as needed
fa.fit(confidence_columns)

# Get eigenvalues to determine the optimal number of factors
ev, v = fa.get_eigenvalues()
print("Eigenvalues:", ev)
# eigenvalues > 1 indicate a useful factor. In this case there are 3 strong factors

# Print factor loadings
loadings = fa.loadings_
print("Factor Loadings:\n", loadings)
# factor loadings > .4 indicate a useful inclusion

# next step: separate and name three domains 
factor1 = ["conmedic", "conarmy", "conbus", "confinan", "conjudge"]
factor2 = ["confed", "conlegis"]
factor3 = ["conpress", "contv"]

indexes["gen_inst_conf"] = indexes[factor1].mean(axis = 1)
indexes["government_conf"] = indexes[factor2].mean(axis = 1)
indexes["media_conf"] = indexes[factor3].mean(axis = 1)

# reliability analysis on three sub-scales
gen_cols = indexes[factor1].copy()
gov_cols = indexes[factor2].copy()
media_cols = indexes[factor3].copy()

reliability_1 = pg.cronbach_alpha(data = gen_cols)
print(reliability_1) # 0.64 - Use Whole (13 measure scale)
reliability_2 = pg.cronbach_alpha(data = gov_cols)
print(reliability_2) # 0.62
reliability_3 = pg.cronbach_alpha(data = media_cols)
print(reliability_3) # 0.56

conf_vars.extend(["conf_index", "gen_inst_conf", "government_conf", "media_conf"])

dash_institutions = dash.groupby("YEAR")[conf_vars].mean().reset_index()

# C. Sensitivity Test
####################
# Count confidenc variables for each R to limit missing cases. 
indexes["num_conf_vars"] = indexes[confidence_vars].notna().sum(axis=1)

### num_conf_vars sensitivity test
thresholds = [6, 8, 10, 11, 12]
results = {}

for threshold in thresholds:
    # Create a copy of the dataset to avoid modifying the original
    temp_df = indexes.copy()
    
    # Recalculate the confidence index based on the threshold
    temp_df["conf_index"] = temp_df[confidence_vars].mean(axis=1, skipna=True)
    temp_df.loc[temp_df["num_conf_vars"] < threshold, "conf_index"] = np.nan
    
    # Summarize the results
    results[threshold] = temp_df["conf_index"].describe()

# Display the summaries for each threshold
for threshold, summary in results.items():
    print(f"Threshold: {threshold}")
    print(summary)
    print("\n" + "="*50 + "\n")
    
# Use minimum 8 variables
indexes["conf_index"] = indexes[confidence_vars].mean(axis=1, skipna=True)
indexes.loc[indexes["num_conf_vars"] < 8, "conf_index"] = np.nan


########  Religiosity ##############################

relig_vars = ["attend", "pray", "reliten", "god", "bible"]

# A. Reliability 
relig_columns = indexes[relig_vars].copy()

reliability = pg.cronbach_alpha(data = relig_columns)
print(reliability) # Alpha = .78

# B. Factor Analysis
fa = FactorAnalyzer(n_factors=3, rotation="varimax")  # Adjust n_factors as needed
fa.fit(relig_columns)

# Get eigenvalues to determine the optimal number of factors
ev, v = fa.get_eigenvalues()
print("Eigenvalues:", ev)
# eigenvalues > 1 indicate a useful factor. In this case there are 3 strong factors

# Print factor loadings
loadings = fa.loadings_
print("Factor Loadings:\n", loadings)
# factor loadings > .4 indicate a useful inclusion

# C. Sensitivity
indexes["num_relig_vars"] = indexes[relig_vars].notna().sum(axis=1)

### num_conf_vars sensitivity test
thresholds = [2, 3, 4, 5]
results = {}

for threshold in thresholds:
    # Create a copy of the dataset to avoid modifying the original
    temp_df = indexes.copy()
    
    # Recalculate the confidence index based on the threshold
    temp_df["religiosity"] = temp_df[relig_vars].mean(axis=1, skipna=True)
    temp_df.loc[temp_df["num_relig_vars"] < threshold, "religiosity"] = np.nan
    
    # Summarize the results
    results[threshold] = temp_df["religiosity"].describe()

# Display the summaries for each threshold
for threshold, summary in results.items():
    print(f"Threshold: {threshold}")
    print(summary)
    print("\n" + "="*50 + "\n")
    
# Use minimum 2 variables
# indexes["religiosity"] = indexes[relig_vars].mean(axis=1, skipna=True)
# indexes.loc[indexes["num_relig_vars"] < 2, "religiosity"] = np.nan


### HAPPINESS ###################################################

happy_vars = ["happy", "life", "haprelate"]

# A. Reliability 
happy_columns = indexes[happy_vars].copy()

reliability = pg.cronbach_alpha(data = happy_columns)
print(reliability) # Alpha = .60

# B. Factor Analysis
fa = FactorAnalyzer(n_factors=2, rotation="varimax")  # Adjust n_factors as needed
fa.fit(happy_columns)

# Get eigenvalues to determine the optimal number of factors
ev, v = fa.get_eigenvalues()
print("Eigenvalues:", ev)
# eigenvalues > 1 indicate a useful factor. In this case there are 3 strong factors

# Print factor loadings
loadings = fa.loadings_
print("Factor Loadings:\n", loadings)
# factor loadings > .4 indicate a useful inclusion

# C. Sensitivity
indexes["num_happy_vars"] = indexes[happy_vars].notna().sum(axis=1)

### num_conf_vars sensitivity test
thresholds = [2, 3]
results = {}

for threshold in thresholds:
    # Create a copy of the dataset to avoid modifying the original
    temp_df = indexes.copy()
    
    # Recalculate the confidence index based on the threshold
    temp_df["happiness"] = temp_df[happy_vars].mean(axis=1, skipna=True)
    temp_df.loc[temp_df["num_happy_vars"] < threshold, "happiness"] = np.nan
    
    # Summarize the results
    results[threshold] = temp_df["happiness"].describe()

# Display the summaries for each threshold
for threshold, summary in results.items():
    print(f"Threshold: {threshold}")
    print(summary)
    print("\n" + "="*50 + "\n")
    
# Use minimum 2 variables
# indexes["happiness"] = indexes[happy_vars].mean(axis=1, skipna=True)
# indexes.loc[indexes["num_happy_vars"] < 2, "happiness"] = np.nan

#########  Social Attitudes ###############################################

social_attitude_vars = ["trust", "helpful", "fair"]

# A. Reliability 
social_attitude_columns = indexes[social_attitude_vars].copy()

reliability = pg.cronbach_alpha(data = social_attitude_columns)
print(reliability) # Alpha = .67

# B. Factor Analysis
fa = FactorAnalyzer(n_factors=3, rotation="varimax")  # Adjust n_factors as needed
fa.fit(social_attitude_columns)

# Get eigenvalues to determine the optimal number of factors
ev, v = fa.get_eigenvalues()
print("Eigenvalues:", ev)
# eigenvalues > 1 indicate a useful factor. In this case there are 3 strong factors

# Print factor loadings
loadings = fa.loadings_
print("Factor Loadings:\n", loadings)
# factor loadings > .4 indicate a useful inclusion

# C. Sensitivity
indexes["num_soc_att_vars"] = indexes[social_attitude_vars].notna().sum(axis=1)

### num_conf_vars sensitivity test
thresholds = [2, 3]
results = {}

for threshold in thresholds:
    # Create a copy of the dataset to avoid modifying the original
    temp_df = indexes.copy()
    
    # Recalculate the confidence index based on the threshold
    temp_df["social_attitudes"] = temp_df[social_attitude_vars].mean(axis=1, skipna=True)
    temp_df.loc[temp_df["num_soc_att_vars"] < threshold, "social_attitudes"] = np.nan
    
    # Summarize the results
    results[threshold] = temp_df["social_attitudes"].describe()

# Display the summaries for each threshold
for threshold, summary in results.items():
    print(f"Threshold: {threshold}")
    print(summary)
    print("\n" + "="*50 + "\n")
    
# Use minimum 2 variables
# indexes["social_attitudes"] = indexes[social_attitude_vars].mean(axis=1, skipna=True)
# indexes.loc[indexes["num_soc_att_vars"] < 2, "social_attitudes"] = np.nan


#### Social Relationships #################

# First, Alpha analysis shows inter-item reliability 
soc_vars = ["socrel", "socfrend", "socommun", "socbar"]

social_columns = indexes[soc_vars].copy()

reliability = pg.cronbach_alpha(data = social_columns)
print(reliability) # Alpha = .44


##### Factor Analysis #####
fa = FactorAnalyzer(n_factors=2, rotation="varimax")  # Adjust n_factors as needed
fa.fit(social_columns)

# Get eigenvalues to determine the optimal number of factors
ev, v = fa.get_eigenvalues()
print("Eigenvalues:", ev)
# eigenvalues > 1 indicate a useful factor.

# Print factor loadings
loadings = fa.loadings_
print("Factor Loadings:\n", loadings)
# factor loadings > .4 indicate a useful inclusion

# next step: separate and name two domains 
factor1 = ["socbar", "socfrend"] # nightlife (out?)
factor2 = ["socrel", "socfrend"] # friends and family (in)?

indexes["soc_out"] = indexes[factor1].mean(axis = 1)
indexes["soc_in"] = indexes[factor2].mean(axis = 1)

# Alpha analysis on three sub-scales
soc_out_cols = indexes[factor1].copy()
soc_in_cols = indexes[factor2].copy()

reliability_1 = pg.cronbach_alpha(data = soc_out_cols)
print(reliability_1)
reliability_2 = pg.cronbach_alpha(data = soc_in_cols)
print(reliability_2)

#### Sensitivity 
indexes["num_soc_vars"] = indexes[soc_vars].notna().sum(axis=1)

### num_conf_vars sensitivity test
thresholds = [2, 3, 4]
results = {}

for threshold in thresholds:
    # Create a copy of the dataset to avoid modifying the original
    temp_df = indexes.copy()
    
    # Recalculate the confidence index based on the threshold
    temp_df["social"] = temp_df[soc_vars].mean(axis=1, skipna=True)
    temp_df.loc[temp_df["num_soc_vars"] < threshold, "social"] = np.nan
    
    # Summarize the results
    results[threshold] = temp_df["social"].describe()

# Display the summaries for each threshold
for threshold, summary in results.items():
    print(f"Threshold: {threshold}")
    print(summary)
    print("\n" + "="*50 + "\n") ### miniscule differences. 

 




