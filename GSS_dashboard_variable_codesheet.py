# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 08:28:22 2025
GSS reCode Sheet with NORCSIZE, REALINC, & PRESTG10
"""

import os 
import pyreadstat
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

os.chdir("C:/Dat_Sci/Data Projects/GSS/dashboard project")

df, meta = pyreadstat.read_sas7bdat('C:/Dat_Sci/Datasets/GSS_sas/gss7222_r3.sas7bdat')

trends = df[["YEAR", "SIZE", "XNORCSIZ", "AGE", "SEX", "EDUC", "PRESTG10", "REALINC",
           "DEGREE", "RACE", "HAPPY", "TRUST", "HELPFUL", "FAIR", "HEALTH", "LIFE", 
           "HAPMAR", "HAPCOHAB", "RELITEN", "GOD", "BIBLE", "REGION", "ATTEND", 
           "PRAY", "HRS1", "SATJOB", "MOBILE16", "POLVIEWS", "PARTYID", "SOCREL", 
           "SOCOMMUN", "SOCBAR", "SOCFREND", "CONEDUC", "CONFED", "CONMEDIC", "CONARMY",
           "CONBUS", "CONCLERG", "CONFINAN", "CONJUDGE", "CONLABOR", 
           "CONLEGIS", "CONPRESS", "CONSCI", "CONTV"]].copy()


##############################################################################
## Short save raw data table #################################################
trends.to_csv("gss_rawdata.csv", index=False)
trends = pd.read_csv("gss_rawdata.csv")
##############################################################################

"""
1. Raw variable recodes and index construction [line 60]
    a. Recodes
        i. Numeric & index variables
        ii. Categorical variables
    b. Construct Raw indexes [441]
        i. Visualize distributions [line 596]
    c. Save recoded raw data table 

2. Standardize Measures and indices [637]
    a. Standardize
    b. Save as Z-score table [line 713]
    
3. Trend data construction
    a. Use yearly means to impute missing values & Reconstruct Indices and trend measures using imputed data
    B. Standardize Imputed Data
    C. Aggregate for Yearly trend table
    d. Save as yearly trend data [line 936]
"""

##############################################################################
## Part Ia. 
## i. Numeric Variable Recodes 
##############################################################################

###### Confidence in institutions variables #################################
conf_vars = ["CONEDUC", "CONFED", "CONMEDIC", "CONARMY", "CONBUS", 
             "CONCLERG", "CONFINAN", "CONJUDGE", "CONLABOR",
             "CONLEGIS", "CONPRESS", "CONSCI", "CONTV"]

# recode in positive direction
for var in conf_vars: 
    trends[var.lower()] = 4 - trends[var] # 1 = "hardly any", 2 = "only some", 3 = "a great deal"
    
# drop original 
trends.drop(conf_vars, axis = 1, inplace=True)

###### Religiosity ##########################################################
# First recode potential religiosity variables measured differently so that high numbers = high reliosity 

# Frequency r prays
trends["pray"] = 6 - trends["PRAY"] # reverse direction # 0 = "never", 5 = "several times/day"
trends.drop(["PRAY"], axis = 1, inplace=True)

# Frequency r attends religious service
trends["attend"] = trends["ATTEND"] # already coded in the right direction # 0 = "never", 8 = "several times/week"
trends.drop(["ATTEND"], axis = 1, inplace=True)

# RELITEN = R's belief in religion
reliten_recodes = {1: 3, # strong affiliation
                   3: 2, # 'somewhat strong' 
                   2: 1, # 'not very strong'
                   4: 0  # 'no religion'
}

trends["reliten"] = trends["RELITEN"].map(reliten_recodes) 
trends.drop(["RELITEN"], axis = 1, inplace=True)

# BIBLE = R's belief in the bible
trends["bible"] = trends["BIBLE"].replace(4, np.nan) # omiting 4 (bible is "other") does not work inside map()

bible_recodes = {
    1: 3, # "word of god" -> high religiosity
    2: 2, # "inspired words"
    3: 1  # "ancient book" -> low religiosity
}

trends["bible"] = trends["bible"].map(bible_recodes)
trends.drop(["BIBLE"], axis = 1, inplace=True)

# R's confidence in the existence of God
trends["god"] = trends["GOD"] - 1 # 5 = "no doubts"; 0 = "don't believe"
trends.drop(["GOD"], axis = 1, inplace=True)


#### HAPPINESS ##############################################################
# General happiness
trends["happy"] = 4 - trends["HAPPY"] # 3 = "very happy"; 1 = "not too happy"
trends.drop(["HAPPY"], axis = 1, inplace=True)

# Feelings about life
trends["life"] = 4 - trends["LIFE"] # 3 = "Exciting"; 1 = "Dull"
trends.drop(["LIFE"], axis = 1, inplace=True)

# Happily married
trends["hapmar"] = 4 - trends["HAPMAR"] # 3 = "very happy"; 1 = "not too happy"
trends.drop(["HAPMAR"], axis = 1, inplace=True)

# Happily cohabitating (for those partned but not married)
trends["hapcohab"] = 4 - trends["HAPCOHAB"] # 3 = "Exciting"; 1 = "Dull"
trends.drop(["HAPCOHAB"], axis = 1, inplace=True)

# Make raw Haprelate variable
# initialize a column for a conditional operation
trends["haprelate"] = np.nan

# Include hapmar for married respondents
trends.loc[(trends["hapmar"].notna()), "haprelate"] = trends["hapmar"]

# Include hapcohab for respondents with a valid 'hapcohab' value
trends.loc[(trends["hapmar"].isna()) & trends["hapcohab"].notna(), "haprelate"] = trends["hapcohab"]
trends["hapmar"].describe()
trends["hapcohab"].describe()
trends["haprelate"].describe()


##### SOCIAL ATTITUDES #######################################################
# Trust
trust_recodes = {
    1: 3, # "most people can be trusted" -> high trust
    2: 1, # "cant be too careful" -> low trust
    3: 2  # "depends" -> ambivalent
}
trends["trust"] = trends["TRUST"].map(trust_recodes)
trends.drop(["TRUST"], axis = 1, inplace=True)

# Helpful
helpful_recodes = {
    1: 3, # "people try to be helpful" -> high trust
    2: 1, # "people look after themselves" -> low trust
    3: 2  # "depends" -> ambivalent
}
trends["helpful"] = trends["HELPFUL"].map(trust_recodes)
trends.drop(["HELPFUL"], axis = 1, inplace=True)

# Fair recodes are ordered different
fair_recodes = {
    1: 1, # "people would take advantage of you" -> high trust
    2: 3, # "people would try to be fair" -> low trust
    3: 2  # "depends" -> ambivalent
}

trends["fair"] = trends["FAIR"].map(fair_recodes)
trends.drop(["FAIR"], axis = 1, inplace=True)

##### Social Relationships Index #############################################
soc_vars = ["SOCREL", "SOCOMMUN", "SOCBAR", "SOCFREND"]

for var in soc_vars: 
    trends[var.lower()] = 7 - trends[var] # recode each variable to be 0 = "never," 6 = "almost daily" with lower case name
    trends.drop(var, axis = 1, inplace=True)
    
# change variable list to lower case for later
soc_vars = [var.lower() for var in soc_vars]

##### Work/Life Balance #####################################################
trends["hrs1"] = trends["HRS1"]

# recode job satisfaction
trends["satjob"] = 5 - trends["SATJOB"] # reverse order so "very satisfied" is highest. 4 = "very satisfited", 1 = "very dissatisfied"

trends.drop(["HRS1", "SATJOB"], axis=1, inplace=True)

##### HEALTH ################################################################
trends["health"] = 5 - trends["HEALTH"] # reverse code # 1 = "poor", 4 = "excellent" 

trends.drop(["HEALTH"], axis=1, inplace=True)

##### EDUC ##################################################################
trends["educ"] = trends["EDUC"] # numeric years of school
trends.drop(["EDUC"], axis = 1, inplace = True)


##### Socioeconomic Status (numeric) #########################################

ses_vars = ["educ", "PRESTG10", "REALINC"] 
# prestig10 is occupational prestige updated to a 2010 standard; Realinc is total family income in 1986 dollars.

scaler = StandardScaler() # using Scikitlearn standardization function

trends[[var + "_z" for var in ses_vars]] = scaler.fit_transform(trends[ses_vars])
#    def standardize(x):
#       return (x - x.mean()) / x.std()
#    trends[ses_vars] = standardize(trends[ses_vars]) does the same thing
    
trends[[var + "_z" for var in ses_vars]] = scaler.fit_transform(trends[ses_vars])

trends["ses_index"] = trends[[var + "_z" for var in ses_vars]].mean(axis=1, skipna = True)

trends["num_ses_vars"] = trends[ses_vars].notna().sum(axis=1)
trends.loc[trends["num_ses_vars"] < 3, "ses_index"] = np.nan # if one of three is nan, then ses is nan. 
print(trends["ses_index"].describe())

trends.drop(["PRESTG10", "REALINC", "educ_z", "PRESTG10_z", "REALINC_z"], axis = 1, inplace = True)

# Plot 
plt.figure(figsize = (8,5))
plt.hist(trends["ses_index"].dropna(), bins=16, color='salmon', edgecolor='black')
plt.title('Distribution of SES')
plt.grid(axis='y')
plt.show()


#############################################################################
## Part 1a. 
## ii. CATEGORICAL RECODES 
#############################################################################

### SES Cateogries ##########################################################
# construct ses categories for dimension filter
# Define conditions
lower_30 = trends["ses_index"].quantile(0.30)
upper_70 = trends["ses_index"].quantile(0.70)

conditions = [
    trends["ses_index"] < lower_30,
    (trends["ses_index"] >= lower_30) & (trends["ses_index"] <= upper_70),
    trends["ses_index"] > upper_70
]

# Corresponding categories
choices = ["Low SES", "Mid SES", "High SES"]

# Apply conditions
trends["ses_category"] = np.select(conditions, choices, default=np.nan)

# Check category distribution
print(trends["ses_category"].value_counts(normalize=True))


### Gender ##################################################################
gender_map = {1: "Male", 2: "Female"}
trends["gender"] = trends["SEX"].map(gender_map)

trends.drop("SEX", axis = 1, inplace = True)

trends["gender"].value_counts()
trends["gender"].isna().sum()


###### Political Party #####################################################
"""GSS codes: 0: "Strong Democrat", 1: "Loose Democrat", 2: "Left Independent", 3: "Center Independent", 
4: "Right Independent", 5: "Loose Republican", 6: "Strong Republican", 7: "Other Party"
"""
party_map = {0: "Strong Democrat", 
             1: "Loose Democrat", 
             2: "Left Independent", 
             3: "Center Independent", 
             4: "Right Independent", 
             5: "Loose Republican", 
             6: "Strong Republican", 
             7: "Other Party"}
    
trends["party"] = trends["PARTYID"].map(party_map)

trends["party"].value_counts()
trends["party"].isna().sum()
trends["party"].groupby(trends["YEAR"]).value_counts()


######## Political Party Centered (Numeric) ##################################
# This variable does not end up getting used. 
trends["party_centered"] = trends["PARTYID"].replace(7, np.nan) - 3

trends["party_centered"].value_counts()

trends.drop(["PARTYID"], axis = 1, inplace=True)


###### Political Views ######################################################
""" Original: 1 = "Extremely liberal", 2 = "Liberal", 
3 = "slightly liberal", 4 = "moderate", 5 = "slightly conservative", 
6 - 7 = "conservative" and "extremely conservative"
"""
pol_view_map = {1: "Extremely Liberal",
             2: "Liberal", 
             3: "Slightly Liberal", 
             4: "Moderate", 
             5: "Slightly Conservative", 
             6: "Conservative", 
             7: "Extremely Conservative"}
    
trends["polview"] = trends["POLVIEWS"].map(pol_view_map)

trends["polview"].value_counts()
trends["polview"].isna().sum()
trends["polview"].groupby(trends["YEAR"]).value_counts()


###### Political Views Centered (Numeric) ####################################
trends["pol_centered"] = trends["POLVIEWS"] - 4

trends["pol_centered"].value_counts()
trends["pol_centered"].isna().sum()

trends.drop(["POLVIEWS"], axis = 1, inplace=True)


##### Region #################################################################
region_map = {1: "New_England",  
              2: "Mid_Atlantic", 
              3: "East_North_Central", 
              4: "West_North_Central", 
              5: "South_Atlantic", 
              6: "East_South_Atlantic",
              7: "West_South_Central", 
              8: "Mountain", 
              9: "Pacific"}

trends["region"] = trends["REGION"].map(region_map)

trends.drop(["REGION"], axis = 1, inplace=True)

trends["region"].value_counts()
trends["region"].isna().sum()
trends["region"].groupby(trends["YEAR"]).value_counts()


##### CITY SIZE ##############################################################
# USE SIZE & XNORCSIZ # There are a few place/size measures. Unfortunately, when comparing data across variables, the place size and labels do not always add up

""" SIZE = x * 1000; 
    XNORSIZ Codes:
    1: "Large central city (>250k)",
    2: "Med central city (50-250k)",
    3: "Large city suburb",
    4: "Med city suburb",
    5: "Large city, unincorporated",
    6: "Med city, unincorporated",
    7: "small city (non-msa)",
    8: "small town or village",
    9: "1-2.5k",
    10: "Open Country"
""" 
conditions = [
    (trends['XNORCSIZ'] == 1) & (trends['SIZE'] >= 1000),
    (trends['XNORCSIZ'] == 1) & (trends['SIZE'] <1000),
    (trends['XNORCSIZ'] == 2),
    (trends['XNORCSIZ'].isin([3, 4])),
    (trends['XNORCSIZ'].isin([5, 6])),
    (trends['XNORCSIZ'] == 7),
    (trends['XNORCSIZ'].isin([8, 9, 10]))
]

choices = [
    "Metropolis",
    "Large City",
    "Medium City",
    "Suburb",
    "Urban Unincorporated",
    "Small City",
    "Rural"
]

trends['place_category'] = np.select(conditions, choices, default=None)

# Check the distribution
print(trends['place_category'].value_counts())

trends.drop(["XNORCSIZ", "SIZE"], axis=1, inplace=True)


###### Mobility since age 16 #################################################

mobility_map = {1: "Same_place",  
                2: "Same_state_new_city",
                3: "Diff_state"}
    
trends["mobile16"] = trends["MOBILE16"].map(mobility_map)

trends.drop(["MOBILE16"], axis = 1, inplace=True)

trends["mobile16"].value_counts()
trends["mobile16"].isna().sum()
trends["mobile16"].groupby(trends["YEAR"]).value_counts()


###### Race #################################################################
race_map = {1: "White", 
            2: "Black", 
            3: "Other_race"}

trends["race"] = trends["RACE"].map(race_map)

trends.drop(["RACE"], axis = 1, inplace=True)

trends["race"].value_counts()
trends["race"].isna().sum()
trends["race"].groupby(trends["YEAR"]).value_counts()


########## Degree ############################################################

degree_map = {0: "Less_than_HS", 
              1: "High_School",
              2: "Associate_JC", 
              3: "Bachelors", 
              4: "Graduate"}

trends["degree"] = trends["DEGREE"].map(degree_map)

trends["degree"].value_counts()
trends["degree"].isna().sum()
trends["degree"].groupby(trends["YEAR"]).value_counts()

trends.drop(["DEGREE"], axis = 1, inplace=True)


#### AGE Group ###############################################################
# Age categories for demographic parameter selection

age_bins = [18, 29, 39, 49, 65, 130]
age_labels = ["18-29", "30-39", "40-49", "50-64", "65+"]
trends["age_group"] = pd.cut(trends["AGE"], bins = age_bins, labels = age_labels, right = True)

trends.drop(["AGE"], axis = 1, inplace=True)

##############################################################################
### Part 1B. 
### Include Raw Indices 
##############################################################################
""" 
See "index analyses" for alpha, factor analysis, and threshold tests
"""
############################
# Confidence in institutions
############################
conf_vars = [var.lower() for var in conf_vars] # rename variables to lowercase
trends["conf_raw"] = trends[conf_vars].mean(axis = 1, skipna = True)

# c.f., threshold analysis: exclude rows with < 8
trends["num_conf_vars"] = trends[conf_vars].notna().sum(axis=1)
trends.loc[trends["num_conf_vars"] < 8, "conf_raw"] = np.nan

# C.f., Factor Analysis "index analysis" sheet
factor1 = ["conmedic", "conarmy", "conbus", "confinan", "conjudge"]
factor2 = ["confed", "conlegis"]
factor3 = ["conpress", "contv"]

# General Confidence in Institutions 
trends["conf_gen"] = trends[factor1].mean(axis = 1)
trends["num_gen_vars"] = trends[conf_vars].notna().sum(axis=1)
trends.loc[trends["num_gen_vars"] < 4, "conf_gen"] = np.nan

# Confidence in Government
trends["conf_gov"] = trends[factor2].mean(axis = 1)
trends["num_gov_vars"] = trends[conf_vars].notna().sum(axis=1)
trends.loc[trends["num_gov_vars"] < 2, "conf_gov"] = np.nan

# Confidence in the Media
trends["conf_media"] = trends[factor3].mean(axis = 1)
trends["num_med_vars"] = trends[conf_vars].notna().sum(axis=1)
trends.loc[trends["num_med_vars"] < 2, "conf_media"] = np.nan


#############
# Religiosity 
#############
# Standardize components, then average 
relig_vars = ["attend", "pray", "god", "bible", "reliten"]

# standardize 
def standardize(x):
    return (x - x.mean()) / x.std()

for var in relig_vars: 
    trends[var + "_z"] = standardize(trends[var])

relig_vars_z = [var + "_z" for var in relig_vars]
trends[relig_vars_z].describe()

trends["religiosity_raw"] = trends[relig_vars_z].mean(axis=1, skipna=True)

trends["religiosity_raw"].describe()

# omit for R's with < 2 vars? C.f., Threshold test (appendix)
trends["num_relig_vars"] = trends[relig_vars].notna().sum(axis=1)
trends.loc[trends["num_relig_vars"] < 2, "religiosity_raw"] = np.nan
trends["religiosity_raw"].describe() # omitting rows with excess nas truncates the range a bit 
# Ss standardize again when constructing Z-scores data table...


###########
# Happiness
###########
hap_vars = ["happy", "life", "haprelate"]
trends["happiness_raw"] = trends[hap_vars].mean(axis=1, skipna=True)

# omit for R's with < 2 vars. 
trends["num_hap_vars"] = trends[hap_vars].notna().sum(axis=1)
trends.loc[trends["num_hap_vars"] < 2, "happiness_raw"] = np.nan


##################
# Social Attitudes 
##################
# because of the distribution, use only rows with all three measures. Add for the raw variable
soc_att_vars = ["trust", "helpful", "fair"]
trends["social_attitude_raw"] = trends[soc_att_vars].sum(axis=1, skipna=True)

# omit for R's with < 2 vars. 
trends["num_soc_att_vars"] = trends[soc_att_vars].notna().sum(axis=1)
trends.loc[trends["num_soc_att_vars"] < 3, "social_attitude_raw"] = np.nan
trends["num_soc_att_vars"].value_counts()


###############
# Socialization
###############
soc_vars = ["socbar", "socommun", "socfrend", "socrel"]
trends["soc_raw"] = trends[soc_vars].mean(axis=1, skipna=True)

# omit for R's with < 2 vars. 
trends["num_soc_vars"] = trends[soc_vars].notna().sum(axis=1)
trends.loc[trends["num_soc_vars"] < 2, "soc_raw"] = np.nan

trends["num_soc_vars"].value_counts() # Cf. appendix for threshold test


###################
# Work-Life Balance 
###################
# Recode work hours
trends["hrs1"].describe()

trends["hrs_trim"] = trends["hrs1"]
# subset minimum hours to ensure people are actually part of the work force
trends.loc[trends["hrs1"] < 10, "hrs_trim"] = np.nan
# trim outliers at 3 std * mean
outlier_threshold = trends["hrs1"].mean() + (3 * trends["hrs1"].std())
trends.loc[trends["hrs1"] > outlier_threshold, "hrs_trim"] = np.nan # setting outliers and <10 hrs to np.nan to preserve rows

# reverse direction of hours for index; so high hours = negative score
max_hours = trends["hrs_trim"].max(skipna = True)

trends["hrs_rev"] = (max_hours + 1) - trends["hrs_trim"] # +1 means that there is not a zero point. 
print(trends["hrs_rev"].describe())

trends["satjob"].describe()

# construct index as satjob * hrs_rev
trends["wlb_raw"] = trends["satjob"] * trends["hrs_rev"]
trends["wlb_raw"].describe()

trends.drop(["hrs_rev", "hrs_trim"], axis = 1, inplace = True)


#################
# Quality of Life
#################
# Standardize first
qol_vars = ["educ", "health", "soc_raw", "wlb_raw"]

def standardize(x):
    return (x - x.mean()) / x.std()

for var in qol_vars: 
    trends[var + "_z"] = standardize(trends[var])

# Rename columns to z-score equivalents
qol_z_vars = [var + "_z" for var in qol_vars]

trends["qol_raw"] = trends[qol_z_vars].mean(axis = 1, skipna = True)

trends["num_qol_vars"] = trends[qol_z_vars].notna().sum(axis=1)
trends["num_qol_vars"].value_counts()

trends.loc[trends["num_qol_vars"] < 4, "qol_raw"] = np.nan

trends["qol_raw"].describe() 


#############################################################################
# Part 1B
## ii. Visualize Index Distributions
#############################################################################

measures = ["educ", "health", "conf_raw", "conf_gen", "conf_gov", "conf_media", 
            "religiosity_raw", "happiness_raw", "social_attitude_raw", 
            "soc_raw", "wlb_raw", "qol_raw"]

for i in measures: 
    plt.figure(figsize = (8,5))
    plt.hist(trends[i].dropna(), bins=12, color='salmon', edgecolor='black')
    plt.title(f'Distribution of {i}')
    plt.grid(axis='y')
    plt.show()


#############################################################################
# Part 1C. 
## Clean up to save the data table as raw, recoded measures
#############################################################################
# remove Z scores from raw data table

trends.drop(qol_z_vars, axis = 1, inplace = True)
trends.drop(relig_vars_z, axis = 1, inplace = True)


# remove na counts
na_counts = ["num_conf_vars", "num_relig_vars", "num_hap_vars", 
                "num_soc_att_vars", "num_soc_vars", "num_qol_vars", 
                "num_gen_vars", "num_gov_vars", "num_med_vars"]
trends.drop(na_counts, axis = 1, inplace = True)


##############################################################################
## Short save ################################################################
trends.to_csv("dashboard_raw.csv", index=False)
trends = pd.read_csv("dashboard_raw.csv")
##############################################################################


##############################################################################
### Part II - Create Table of Z values for all measures ######################
##############################################################################

### Part 2a.
### STANDARDIZE All MEASURES #################################################

for var in conf_vars: 
    trends[var + "_z"] = standardize(trends[var])
    
conf_indexes = ["conf_gen", "conf_gov", "conf_media"]
for var in conf_indexes: 
    trends[var + "_z"] = standardize(trends[var])

for var in relig_vars:
    trends[var + "_z"] = standardize(trends[var])

for var in hap_vars:
    trends[var + "_z"] = standardize(trends[var])

for var in soc_att_vars:
    trends[var + "_z"] = standardize(trends[var])

for var in soc_vars:
    trends[var + "_z"] = standardize(trends[var])

wlb_vars = ["hrs1", "satjob"]
for var in wlb_vars:
    trends[var + "_z"] = standardize(trends[var])
    
qol_vars = ["educ", "health"]
for var in qol_vars:
    trends[var + "_z"] = standardize(trends[var])

# Standardize indexes
indexes = ["conf_raw", "religiosity_raw", "happiness_raw", 
           "social_attitude_raw", "soc_raw","wlb_raw", "qol_raw"] 

# religiosity and qol are re-standardized

# rename indexes as z scores
for var in indexes: 
    new_var = var.replace("_raw", "_z")
    trends[new_var] = standardize(trends[var])


### Part 2b. #################################################################
## Clean up table ############################################################

# Drop original raw vars. 
trends.drop(conf_vars, axis=1, inplace = True)
trends.drop(conf_indexes, axis = 1, inplace = True)
trends.drop(relig_vars, axis=1, inplace = True)
trends.drop(hap_vars, axis=1, inplace = True)
trends.drop(soc_att_vars, axis=1, inplace = True)
trends.drop(soc_vars, axis=1, inplace = True)
trends.drop(qol_vars, axis=1, inplace = True)
trends.drop(wlb_vars, axis=1, inplace = True)

mar_vars = ["hapmar", "hapcohab"]
trends.drop(mar_vars, axis=1, inplace = True)
trends.drop(indexes, axis=1, inplace=True)

### KEEP CATEGORICAL DATA IN THE Z TABLE 

### Visualize Again ###############################################################
measures_z = ["educ_z", "health_z", "conf_z", "conf_gen_z", "conf_gov_z", "conf_media_z", 
            "religiosity_z", "happiness_z", "social_attitude_z", 
            "soc_z", "wlb_z", "qol_z"]

for i in measures_z: 
    plt.figure(figsize = (8,5))
    plt.hist(trends[i].dropna(), bins=12, color='salmon', edgecolor='black')
    plt.title(f'Distribution of {i}')
    plt.grid(axis='y')
    plt.show()

#### Part 2C. ############################################################
## Short save ###############################################################
trends.to_csv("dashboard_z_measures.csv", index=False)
#############################################################################

##############################################################################
## Part 3
## 3a. Impute ################################################################
""" Impute method is to use the yearly means of raw variables to fill in missing 
values. The purpose of imputing is create smoother trend lines, so this method seems
simple and intuitive. It requires constructing yearly aggregates, imputing aggregate 
values for missing years, then merging that aggregated data and using it to impute 
for respondent level scores. I drop the aggregated data column, but use the r-level 
scores to reconstruct the indexes with a complete (imputed) dataset. 
"""
##############################################################################

# USE RAW DATA - i.e., Impute THEN standardize for Trend lines
trends = pd.read_csv("dashboard_raw.csv")


### 3a i. Conf Index #########################################################
# calculate yearly confidence means in new data frame
conf_means = trends.groupby("YEAR")[conf_vars].mean()

# fill missing year means with previous (if not) then subsequent year means
conf_means[conf_vars] = conf_means[conf_vars].fillna(method="ffill").fillna(method="bfill")

# merge yearly means as new columns
trends = trends.merge(conf_means, on="YEAR", suffixes = ("", "_yearly_mean"))

# fill r-level missing confidence scores with yearly means
for var in conf_vars: 
    trends[var] = trends[var].fillna(trends[var + "_yearly_mean"])
    
# drop the merged yearly mean values (they now exist at r - level)
trends.drop(columns=[var + "_yearly_mean" for var in conf_vars], inplace=True)

# check work 
trends[conf_vars].isna().groupby(trends["YEAR"]).sum()

# NOW construct the Imputed Conf Indices
trends["conf_trend"] = trends[conf_vars].mean(axis=1, skipna=True)

# C.f., Factor Analysis (appendix)
factor1 = ["conmedic", "conarmy", "conbus", "confinan", "conjudge"]
factor2 = ["confed", "conlegis"]
factor3 = ["conpress", "contv"]

trends["conf_gen_trend"] = trends[factor1].mean(axis = 1)
trends["conf_gov_trend"] = trends[factor2].mean(axis = 1)
trends["conf_media_trend"] = trends[factor3].mean(axis = 1)

# Rename the columns in the DataFrame
rename_dict = {var: var + "_trend" for var in conf_vars}
trends = trends.rename(columns=rename_dict)
trends.drop(["conf_raw", "conf_gen", "conf_gov", "conf_media"], axis = 1, inplace = True)


### 3a ii. Religiosity #######################################################
# Check missing values
trends[relig_vars].isna().sum()
trends[relig_vars].isna().groupby(trends["YEAR"]).sum()

## IMPUTE Religiosity Variables
relig_means = trends.groupby("YEAR")[relig_vars].mean()
relig_means[relig_vars] = relig_means[relig_vars].fillna(method="ffill").fillna(method="bfill")

trends = trends.merge(relig_means, on="YEAR", suffixes = ("", "_yearly_mean"))

for var in relig_vars: 
    trends[var] = trends[var].fillna(trends[var + "_yearly_mean"])

trends.drop(columns=[var + "_yearly_mean" for var in relig_vars], inplace=True)

# Religiosity variables have to be standardized before constructing the index 
def standardize(x):
    return (x - x.mean()) / x.std()

for var in relig_vars: 
    trends[var + "_z"] = standardize(trends[var])

relig_vars_z = [var + "_z" for var in relig_vars]

trends["religiosity_trend"] = trends[relig_vars_z].mean(axis=1, skipna=True)
trends["religiosity_trend"].describe()
# Will Standardize Again in part b. 

# Rename the columns in the DataFrame
rename_dict = {var: var + "_trend" for var in relig_vars}
trends = trends.rename(columns=rename_dict)

trends.drop(relig_vars_z, axis = 1, inplace = True)
trends.drop(["religiosity_raw"], axis = 1, inplace = True)


#### 3a iii. HAPPINESS #######################################################
#### Happy variables are all on the same scale

happy_vars = ["happy", "life", "haprelate"]
trends[happy_vars].isna().groupby(trends["YEAR"]).sum()

# Impute happy variables including haprelate
happy_means = trends.groupby("YEAR")[happy_vars].mean()
happy_means[happy_vars] = happy_means[happy_vars].fillna(method="ffill").fillna(method="bfill")

trends = trends.merge(happy_means, on="YEAR", suffixes = ("", "_yearly_mean"))

for var in happy_vars: 
    trends[var] = trends[var].fillna(trends[var + "_yearly_mean"])

trends.drop(columns=[var + "_yearly_mean" for var in happy_vars], inplace=True)

# NOW construct Index with haprelate
trends["happiness_trend"] = trends[happy_vars].mean(axis=1, skipna=True)

# post_check
trends[happy_vars].isna().groupby(trends["YEAR"]).sum()
trends["happiness_trend"].isna().groupby(trends["YEAR"]).sum()
trends["happiness_trend"].describe()

# Rename the columns in the DataFrame
rename_dict = {var: var + "_trend" for var in happy_vars}
trends = trends.rename(columns=rename_dict)

# drop raw 
trends.drop(["happiness_raw"], axis = 1, inplace = True)


### 3a iv. SOCIAL ATTITUDES ##################################################

trends[soc_att_vars].isna().groupby(trends["YEAR"]).sum()

## IMPUTE Social Attitudes  
soc_att_means = trends.groupby("YEAR")[soc_att_vars].mean()
soc_att_means[soc_att_vars] = soc_att_means[soc_att_vars].fillna(method="ffill").fillna(method="bfill")

trends = trends.merge(soc_att_means, on="YEAR", suffixes = ("", "_yearly_mean"))

for var in soc_att_vars: 
    trends[var] = trends[var].fillna(trends[var + "_yearly_mean"])

trends.drop(columns=[var + "_yearly_mean" for var in soc_att_vars], inplace=True)

## NOW construct the index
trends["social_attitude_trend"] = trends[soc_att_vars].mean(axis=1, skipna=True)

trends["social_attitude_trend"].describe()
trends["social_attitude_trend"].isna().value_counts()

# Rename the columns in the DataFrame
rename_dict = {var: var + "_trend" for var in soc_att_vars}
trends = trends.rename(columns=rename_dict)

# drop raw 
trends.drop(["social_attitude_raw"], axis = 1, inplace = True)


### 3a v. Social Relationships ###############################################

trends[soc_vars].isna().groupby(trends["YEAR"]).sum()

# Impute
soc_means = trends.groupby("YEAR")[soc_vars].mean()
soc_means[soc_vars] = soc_means[soc_vars].fillna(method="ffill").fillna(method="bfill")

trends = trends.merge(soc_means, on="YEAR", suffixes = ("", "_yearly_mean"))

for var in soc_vars: 
    trends[var] = trends[var].fillna(trends[var + "_yearly_mean"])

trends.drop(columns=[var + "_yearly_mean" for var in soc_vars], inplace=True)

## NOW construct the index
trends["soc_trend"] = trends[soc_vars].mean(axis=1, skipna=True)
trends["soc_trend"].describe()

# Rename the columns in the DataFrame
rename_dict = {var: var + "_trend" for var in soc_vars}
trends = trends.rename(columns=rename_dict)

# drop raw 
trends.drop(["soc_raw"], axis = 1, inplace = True)


### 3a vi. Work/Life Balance #####################################################

wlb_vars = ["hrs1", "satjob"]
trends[wlb_vars].isna().groupby(trends["YEAR"]).sum()

wlb_means = trends.groupby("YEAR")[wlb_vars].mean()
wlb_means[wlb_vars] = wlb_means[wlb_vars].fillna(method="ffill").fillna(method="bfill")

trends = trends.merge(wlb_means, on="YEAR", suffixes = ("", "_yearly_mean"))

for var in wlb_vars: 
    trends[var] = trends[var].fillna(trends[var + "_yearly_mean"])

trends.drop(columns=[var + "_yearly_mean" for var in wlb_vars], inplace=True)

# copy/paste hrs recodes from above
trends["hrs_trim"] = trends["hrs1"]
# subset minimum hours to ensure people are actually part of the work force
trends.loc[trends["hrs1"] < 10, "hrs_trim"] = np.nan
# trim outliers at 3 std * mean
outlier_threshold = trends["hrs1"].mean() + (3 * trends["hrs1"].std())
trends.loc[trends["hrs1"] > outlier_threshold, "hrs_trim"] = np.nan # setting outliers and <10 hrs to np.nan to preserve rows

# reverse direction of hours for index; so high hours = negative score
max_hours = trends["hrs_trim"].max(skipna = True)

trends["hrs_rev"] = (max_hours + 1) - trends["hrs_trim"] # +1 means that there is not a zero point. 
print(trends["hrs_rev"].describe())

## NOW construct the index
trends["wlb_trend"] = trends["hrs_rev"] * trends["satjob"]
trends["wlb_trend"].describe()

# Rename the columns in the DataFrame
rename_dict = {var: var + "_trend" for var in wlb_vars}
trends = trends.rename(columns=rename_dict)

trends.drop(["hrs_rev", "hrs_trim", "wlb_raw"], axis=1, inplace=True)


### 3a vii. QOL ################################################################

qol_vars = ["health", "educ"] # soc trend and wlb trend already made up 

trends[qol_vars].isna().groupby(trends["YEAR"]).sum()

qol_means = trends.groupby("YEAR")[qol_vars].mean()
qol_means[qol_vars] = qol_means[qol_vars].fillna(method="ffill").fillna(method="bfill")

trends = trends.merge(qol_means, on="YEAR", suffixes = ("", "_yearly_mean"))

for var in qol_vars: 
    trends[var] = trends[var].fillna(trends[var + "_yearly_mean"])

trends.drop(columns=[var + "_yearly_mean" for var in qol_vars], inplace=True)


# Rename the columns in the DataFrame
rename_dict = {var: var + "_trend" for var in qol_vars}
trends = trends.rename(columns=rename_dict)


## NOW construct the index, C.f., recodes 
qol_vars = ["educ_trend", "health_trend", "soc_trend", "wlb_trend"]

def standardize(x):
    return (x - x.mean()) / x.std()

for var in qol_vars: 
    trends[var + "_z"] = standardize(trends[var])

qol_zs = [var + "_z" for var in qol_vars]

trends["qol_trend"] = trends[qol_zs].mean(axis = 1, skipna = True)

trends["qol_trend"].describe()


trends.drop(qol_zs, axis = 1, inplace = True)
trends.drop(["qol_raw"], axis = 1, inplace = True)

#############################################################################
##### Clean Data Table - Drop temp vars

trends.drop(["hapmar", "hapcohab"], axis = 1, inplace = True)

categoricals = ["party_centered", "pol_centered", "gender", "party", "polview", 
        "region", "mobile16", "race", "degree", "age_group", "ses_category", "place_category"]

trends.drop(categoricals, axis = 1, inplace = True)

#############################################################################
## Part 3b. Standardize Imputed Trends ######################################
##############################################################################

scaler = StandardScaler()

exclude_vars = ["YEAR"]
trends_numeric = trends.select_dtypes(include='number').columns.difference(exclude_vars)

trends[trends_numeric] = scaler.fit_transform(trends[trends_numeric])

### 3C. Finally, Aggregate #######################################################
######### Indices By year
yr_trends = trends.groupby("YEAR").mean().reset_index()

## 3D. Save ############################################
yr_trends.to_csv("dashboard_yearly_trends.csv", index=False)



