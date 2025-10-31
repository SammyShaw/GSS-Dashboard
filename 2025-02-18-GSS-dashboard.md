---
layout: post
title: "GSS Dashboard"
image: "/posts/GSS_dashboard.png"
tags: [GSS, Data Cleaning, Visualization, Tableau]
---

## Summary & Context
This project demonstrates 1. Visualization using Tableau, and 2. Preprocessing and analyzing a large public dataset. 

I chose this project to practice Tableau and to represent my background in the social sciences. It is also an effort to celebrate publicaly available data and an attempt to make that data even more accessible. The General Social Survey, which receives its primary funding from the National Science Foundation, has been collecting data on American political opinions and social behaviors since 1972. While the analyses here tell a story of Americans' declining trust in instutions alongside rising political polarization, current administration efforts to slash public spending puts funding for projects like the GSS in the crosshairs. 

## The General Social Survey Dashboard

The General Social Survey consists of 34 waves of representative surveys. The entire dataset contains over 72,000 rows and over 6,600 columns. See the [GSS](https://gss.norc.org/) website for further details. 

My purpose here is not to provide extensive analysis of any one measure, but users can interactively explore basic relationships through visualization. I constructed a series of measures at my discretion, including: 

* Education
* Health
* Happiness
* Social Attitudes
* Social Relationships
* Confidence in Institutions
* Confidence in Government
* Confidence in Media
* Work-Life Balance
* Quality of Life
* Religiosity
* Polarization

Click on a measure button to see its description, distribution, and trends. Toggle a demographic dimension. Select a specific year or all years. 

<iframe seamless frameborder="0" src="https://public.tableau.com/views/GSS_2_17387981106750/GSSDashboard?:embed=yes&:display_count=yes&:showVizHome=no" width = '1100' height = '900'></iframe>
Unfortunately, I have difficulty right-sizing Tableau's dashboards for github pages. Automatic scaling in Tableau alters objects in the view, and github's default width does not correspond with my desktop. 
Visit my [Tableau Public Profile](https://public.tableau.com/app/profile/samuel.shaw2748/vizzes) for a more responsive Dashboard:
[50 Years of the General Social Survey](https://public.tableau.com/views/GSS_2_17387981106750/GSSDashboard?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

Details for each measure are shared below. Briefly, most measures are individual-level indices (except Education and Health, which are raw measures, and Polarization, which is an aggregation). All measures except Polarization have been standardized for better comparability. Global trend line data include imputed data for the sake of drawing a continuous trend line - but those imputed data do not carry over to other charts. 

Python coding scripts are available in my [Github repository](https://github.com/SammyShaw/GSS-Dashboard). Examples are shown below. 

<br> 

### Measure and Dimension details

| **Variable Name**               | **Variable Type**   | **Description**                                                                                  | **GSS Variables** |
|----------------------------------|---------------------|--------------------------------------------------------------------------------------------------|------------------|
| Education                        | Single Measure     | Years of school, standardized.                                                                   | EDUC             |
| Health                           | Single Measure     | Self-reported Health, scale 1 (poor health) - 4 (excellent health), standardized.                | HEALTH           |
| Happiness                        | Index Measure      | Average happiness, life excitement, and happy relationships, standardized.                      | HAPPY, LIFE, HAPMAR, HAPCOHAB |
| Religiosity                      | Index Measure      | Average religious service attendance, prayer, importance of religion, and strength of beliefs, standardized. | ATTEND, PRAY, RELITEN, GOD, BIBLE |
| Social Relationships             | Index Measure      | Average time spent with family, friends, neighbors, and out at bars, standardized.              | SOCREL, SOCFREND, SOCOMMUN, SOCBAR |
| Social Attitudes                 | Index Measure      | Average trust, and feelings about others being helpful and fair.                                | TRUST, HELPFUL, FAIR |
| Work-Life Balance                | Index Measure      | Product of job satisfaction and reversed hours/week worked for respondents with at least 10 hr/week, standardized. | SATJOB, HRS1 |
| Quality of Life                  | Index Measure      | Average standardized Education, Health, Social Relationships, and Work-Life Balance.            | EDUC, HEALTH, SOCREL, SOCFREND, SOCOMMUN, SOCBAR, SATJOB, HRS1 |
| Confidence in Institutions        | Index Measure      | Average confidence in 13 institutions.                                                          | CONEDUC, CONFED, CONMEDIC, CONARMY, CONBUS, CONCLERG, CONFINAN, CONJUDGE, CONLABOR, CONLEGIS, CONPRESS, CONSCI, CONTV |
| Confidence in Government          | Index Measure      | Average confidence in congress and executive branch.                                            | CONFED, CONLEGIS |
| Confidence in Media               | Index Measure      | Average confidence in television and the press.                                                 | CONPRESS, CONTV |
| Age Group                         | Dimension          | Respondent's age categorized for comparability.                                                 | AGE |
| Degree                            | Dimension          | Respondent's education level (note correlation with Education, above).                          | DEGREE |
| Race                              | Dimension          | Respondent's race.                                                                              | RACE |
| Socioeconomic Status              | Index Dimension    | Standardized education (years of school), occupational prestige, and total family income, split: 30%(low) / 40%(mid) / 30%(high). | EDUC, PRESTG10, REALINC |
| Gender                            | Dimension          | Respondent's gender.                                                                            | SEX |
| Race * Gender                     | Dimension          | Respondent's Race/Gender.                                                                      | RACE, SEX |
| Race * SES                        | Dimension          | Respondent's Race/Socioeconomic status.                                                        | RACE, EDUC, REALINC, PRESTG10 |
| Place                             | Dimension          | Type of Social Environment, recoded using SIZE and XNORCSIZ.                                   | SIZE, XNORCSIZ |
| Region                            | Dimension          | Region of interview. All nine categories are used in the map. Collapsed to four categories for dimension comparison. | REGION |

___

## Sample Processing, Recoding & Index Construction
In the code below, I will
* Select variables from GSS for the dashboard.
* Demonstrate key variable recodes. 
* Construct indices, demonstrating reliability analysis, factor analysis, and a minimum variable threshold.
* Impute measures for continuous trend lines.

<br>

### Select Variables

```python
# import required packages and libraries
import os 
import pyreadstat # reads sas files
import pandas as pd
import numpy as np
import pingouin as pg
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler # I'll also write my own standardization function. 
import matplotlib.pyplot as plt

os.chdir("C:/Dat_Sci/Data Projects/GSS/dashboard project")

df, meta = pyreadstat.read_sas7bdat('C:/Dat_Sci/Datasets/GSS_sas/gss7222_r3.sas7bdat')

df.shape
Out[5]: (72390, 6691)
```

6,691 is a lot of columns. To identify relevant columns, I used the GSS website's variable explorer dashboard, which allows you search variables by topic area, or by module, including their "core" module - measures that are collected at every wave, and many others. The following are selected after extensive digging, researching and iteratively realizing what I am interested in. 

```python
trends = df[["YEAR", "SIZE", "XNORCSIZ", "AGE", "SEX", "EDUC", "PRESTG10", "REALINC",
           "DEGREE", "RACE", "HAPPY", "TRUST", "HELPFUL", "FAIR", "HEALTH", "LIFE", 
           "HAPMAR", "HAPCOHAB", "RELITEN", "GOD", "BIBLE", "REGION", "ATTEND", 
           "PRAY", "HRS1", "SATJOB", "MOBILE16", "POLVIEWS", "PARTYID", "SOCREL", 
           "SOCOMMUN", "SOCBAR", "SOCFREND", "CONEDUC", "CONFED", "CONMEDIC", "CONARMY",
           "CONBUS", "CONCLERG", "CONFINAN", "CONJUDGE", "CONLABOR", 
           "CONLEGIS", "CONPRESS", "CONSCI", "CONTV"]].copy()
```
<br>

### Key Variable Recodes

I demonstrate recodes and index construction for two key measures: Confidence in Institutions & Work-Life Balance
<br>

#### Confidence in Institutions

The GSS provides 13 items about confidence in institutions. The questionaire script reads: "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them?"

And the GSS codes these responses:
1: "a great deal"
2: "only some"
3: "hardly any"

To create an index out of these I'll first need to recode these questions in a positive direction so that outcomes are easily interpretatble: high numbers = high confidence. I'll then analyze how internally reliable the set of 13 questions is using an alpha analysis. I'll use factor analysis to see whether and how respondents cohere around these questions. Because 13 is a lot of items, we'll then want to know if there is a minimum item cut-off if some rows contain missing values. That is, we want to minimize the number of index values that contain few item answers, but maximize our total responses (n). Finally, this index, along with all the others, will be standardized for easy comparison.  
```python
# use a list to recode all items at once
conf_vars = ["CONEDUC", "CONFED", "CONMEDIC", "CONARMY", "CONBUS", 
             "CONCLERG", "CONFINAN", "CONJUDGE", "CONLABOR",
             "CONLEGIS", "CONPRESS", "CONSCI", "CONTV"]

# I change the case to keep the variable name while indicating it has been recoded.
for var in conf_vars: 
    trends[var.lower()] = 4 - trends[var]  # Subtracting 4 from the original yeilds the reverse order.
# And then drop the original. 
trends.drop(conf_vars, axis = 1, inplace=True)

# Now construct the Index
conf_vars = [var.lower() for var in conf_vars] # First I want the same list in lower case because I renamed the variables to lower case.
trends["conf_index"] = trends[conf_vars].mean(axis = 1, skipna = True) # take the mean, instead of the sum, so that rows with missing values don't bring down the totals. I'll deal with missing data below.

trends["conf_index"].describe()
Out[12]: 
count    48957.000000
mean         2.031674
std          0.352033
min          1.000000
25%          1.800000
50%          2.000000
75%          2.230769
max          3.000000
Name: conf_index, dtype: float64

# Now we have an index, but how well to the component items cohere, and how do respondents' answers cohere?

## A. Alpha Analysis for inter-item reliability

confidence_columns = trends[conf_vars].copy() # set the component columns apart

reliability = pg.cronbach_alpha(data = confidence_columns)
print(reliability) # Alpha = .79

# An alpha of .79 indicates high inter-item reliability. The items are all getting after one underlying construct, confidence in institutions. 

## B. Factor Analysis 

fa = FactorAnalyzer(n_factors=3, rotation="varimax")  # Adjust n_factors as needed
fa.fit(confidence_columns)

# Get eigenvalues to determine the optimal number of factors
ev, v = fa.get_eigenvalues()
print("Eigenvalues:", ev)
Eigenvalues: [3.54629544 1.41743876 1.00764509 0.96894189 0.86242839 0.80770145
 0.75664672 0.70908479 0.67070858 0.61520566 0.58425706 0.55188097
 0.50176519]
# eigenvalues > 1 indicate a useful factor. In this case there are 3 strong factors

# Print factor loadings to observe which items cohere in each factor.
loadings = fa.loadings_
print("Factor Loadings:\n", loadings)
Factor Loadings:
 [[ 0.36118129  0.24410226  0.23097058]
 [ 0.19306445  0.59500797  0.1643183 ]
 [ 0.62462896 -0.00825083  0.17924858]
 [ 0.45896996  0.17788212 -0.03423817]
 [ 0.41959102  0.25217145  0.09374704]
 [ 0.28677819  0.22292182  0.11038094]
 [ 0.38402004  0.25974974  0.17509192]
 [ 0.45309956  0.3305685   0.10799279]
 [ 0.06857161  0.2834985   0.27623394]
 [ 0.16888011  0.6249543   0.29079604]
 [ 0.09311359  0.15853253  0.66356791]
 [ 0.52642256  0.03028766  0.04476395]
 [ 0.1125664   0.16761342  0.54758137]]
# factor loadings > .4 indicate a useful inclusion. The rows represent the variables (in the order in which I listed them), the columns represent the factors. 

# next step: separate and name three domains 
factor1 = ["conmedic", "conarmy", "conbus", "confinan", "conjudge"] # generic confidence in institutions
factor2 = ["confed", "conlegis"] # confidence in the government
factor3 = ["conpress", "contv"] # confidence in the media

# subsequent reliability analysis does not show an improvement of factor1 (a subset of 5) over the total index measuring confidence in 13 institutions.
# For the dashboard, I'll keep the Confidence in Institutions Index, and I'll add seperate indices for Confidence in Government and Confidence in Media.

```
<br>

##### Number of Items Sensitivity Test
Earlier I mentioned that taking the mean of the component measures is better than summing to the get the index because if there are missing values in some of the items, then a respondent's total index score is artificially lowered. Averaging any number of variables of range 1-3 still yields a range of 1-3, which doesn't feel like a composite index, but we would standardize in the end anyway, which will (ideally) yeild us a zero-centered range of -3 to 3 either way. But when we do take the mean, we should be careful that the output represents the same thing for all respondents. If one respondent answered a 3 for one item and then refused to continue, that would be much different than a respondent who answered carefully for all 13. We can count the number of items a respondent answered, and compare means, distributions, and n for respondents that answered k, k-1, k-2, etc. 

```python
trends["num_conf_vars"] = trends[conf_vars].notna().sum(axis=1) # count the number of missing items per row. 

thresholds = [6, 8, 10, 11, 12] # I'll compare distributions for each of these minimum questions answered thresholds. 
results = {} 
for threshold in thresholds:
    temp_df = indexes.copy() # Create a copy of the dataset to avoid modifying the original
    temp_df["conf_index"] = temp_df[confidence_vars].mean(axis=1, skipna=True) # Recalculate the confidence index based on the threshold
    temp_df.loc[temp_df["num_conf_vars"] < threshold, "conf_index"] = np.nan # Assign rows with missing items < threshold to 'nan'. 
    
    # Summarize the results
    results[threshold] = temp_df["conf_index"].describe()

# Display the summaries for each threshold
for threshold, summary in results.items():
    print(f"Threshold: {threshold}")
    print(summary)
    print("\n" + "="*50 + "\n")

Threshold: 6
count    48665.000000
mean         2.031716
std          0.350796
min          1.000000
25%          1.800000
50%          2.000000
75%          2.230769
max          3.000000
Name: conf_index, dtype: float64

==================================================

Threshold: 8
count    48290.000000
mean         2.031468
std          0.350350
min          1.000000
25%          1.800000
50%          2.000000
75%          2.230769
max          3.000000
Name: conf_index, dtype: float64

==================================================

Threshold: 10
count    47510.000000
mean         2.031037
std          0.349782
min          1.000000
25%          1.800000
50%          2.000000
75%          2.230769
max          3.000000
Name: conf_index, dtype: float64

==================================================

Threshold: 11
count    46674.000000
mean         2.030616
std          0.349311
min          1.000000
25%          1.818182
50%          2.000000
75%          2.230769
max          3.000000
Name: conf_index, dtype: float64

==================================================

Threshold: 12
count    44891.000000
mean         2.029900
std          0.348355
min          1.000000
25%          1.769231
50%          2.000000
75%          2.230769
max          3.000000
Name: conf_index, dtype: float64

==================================================
# Fortunately, there is very little change in n, and amost no change in the mean. Using a minimum of 8 variables for the index provides the best trade off. 

# Use minimum 8 variables

# trends["conf_index"] = trends[confidence_vars].mean(axis=1, skipna=True) # index construction again
trends.loc[trends["num_conf_vars"] < 8, "conf_index"] = np.nan # calling all rows with less than 8 answers "nan."

```
<br>

#### Work-Life Balance

A less obvious measure is Work-Life Balance. Rather than providing a subjective measure, the GSS offers several items relevant to this construct. Past researchers have constructed this index in myriad ways, often including domestic responsibilities (especially for working women that double as primary caretakers) among the factors that make it up. However, work-life balance is not the primary focus here, so I'll elect to keep it simple and define it as a function of job satisfaction and number of hours worked. As far as trends go, I'm simply interested to see if the average American is working longer and harder hours.

* Job Satistfaction (SATJOB): The GSS asks: "On the whole, how satisfied are you with the work you do--would you say you are very satisfied, moderately satisfied, 
a little dissatisfied, or very dissatisfied?"

GSS Codes: 
1: "Very satisfied"
2: "Moderately satisfied"
3: "Not very satisfied"
4: "Very dissatisfied"

* Work Hours (HRS1): The GSS asks: "A. IF WORKING, FULL OR PART TIME:  How many hours did you work last week, at all jobs?

GSS Codes: Continuous numeric, Range 0-89. 

To construct this index I'll first recode these variables so that they are numerically interpretable within our construct. I'll reverse code both job satisfaction so that high numbers = high satisfaction = 'good for work-life balance', and work hours, so that low work hours = good for work-life balance. Even though the GSS only asks work hours among those with a Full-time or part-time job, I'll exclude those working less than 10 hours to ensure that people actually are working. And because there are a number of respondents who have apparently worked 89+ hours, I identified outliers (>80 hrs) and removed these from the distribution, for a practical range of 10-80 hrs/work/week. Multiplying hours worked by job satisfaction yields a possible index score of range 10-320. Because these component variables are on completely different scales, alpha analysis and factor analysis do not apply. And because there are only two variables that make up the construct, it will be necessary to include both in the measure. Like Confidence in institutions, Work-Life Balance will be standardized so it is comparable with other measures on the same scale. 

```python

# recode job satisfaction
trends["satjob"] = 5 - trends["SATJOB"] # reverse order so "very satisfied" is highest.

trends["hrs_trim"] = np.where(trends["HRS1"] < 10, np.nan, trends["HRS1"]) # subset minimum hours to ensure people are actually part of the work force

# trim outliers at 3 std * mean
outlier_threshold = trends["HRS1"].mean() + (3 * trends["HRS1"].std()) # Find the treshold. 3 standard deviations above the mean is one way. 
trends.loc[trends["HRS1"] > outlier_threshold, "hrs_trim"] = np.nan # setting outliers and <10 hrs to np.nan to preserve rows.

# reverse direction of hours for index; so high hours worked = poor work/life balance
max_hours = trends["hrs_trim"].max(skipna = True)
trends["hrs_rev"] = (max_hours + 1) - trends["hrs_trim"] # max_hours + 1 means that there is not a zero point. 

# print(trends["hrs_rev"].describe())
# print(trends["satjob"].describe())

# construct index as satjob * hrs_rev
trends["wlb_raw"] = trends["satjob"] * trends["hrs_rev"]
# trends["wlb_raw"].describe()

trends.drop(["hrs_rev", "hrs_trim"], axis = 1, inplace = True)

```
<br>

#####  Standardize
Because I'll end up with a dozen or so index measures, I will standardize for easy comparison. That is, I'll convert raw index scores (which in some cases range from 1-3 and in other cases from 1-320) to z-scores, giving each a range of roughly -3 to 3. Each individual measure is then interpretable as the number of standard deviations from the mean. While z-scores don't mean anything specifically in terms of confidence in institutions or work-life balance, for example, they do reveal which scores are **relatively** high or low. The convenience of being able to put standardized measures on the same graphs because their scales align is a good trade-off. 

One way to standardize is to use scikitlearn's StandardScaler() function, another is to simply subtract the mean and divide by the standard deviation. Because I'm not pipelining any data, and because StandardScaler() and ordinary math end up treating missing values all the same, I'll use a simple function here. 

```python

def standardize(x):
    return (x - x.mean()) / x.std()

dashboard_measures = ["education", "health", "religiosity", "social_attitudes", "social_relationships", "work_life_balance", "quality_of_life, conf_index", "conf_gov", "conf_media"]

for measure in dashboard_measures:
    trends[measure + "_z"] = standardize(trends[measure]) # the z indicates that it is now standardized. 

```
<br>

### Data Storytelling vs. Comprehensiveness vs. Interactivity in Tableau

Tableau is a powerful tool because it allows almost limitless ways to visualize information, which gives the author incredible creative power to tell stories with data. Its interactive features allow users to assume some of that power as well. However, I find that the more information that is crammed onto a dashboard, the more difficult it becomes to make a data story cohere. A simple bivariate relationship can be given a well-designed graph, exciting visuals, KPI-style output text, and accompanying description or analytic detail. And it is often these simple dashboards that make Tableau's viz-of-the-day list. But my purpose was more a celebration of GSS's awesome breadth. It was hard to choose what NOT to include. I make no appologies that I crammed a lot in; in the future I would like to add more. Unfortunately, this means that there is not a singular storyline in the dashboard, but multiple, and the user is given the ability to uncover what narrative they will. 

In any case the key to creating an appealing dashboard has as much to do with design decisions as what data are available. I chose to center the trend lines and include "50 years" in the title to emphasize the theme of changes over time. I created index buttons in different colors with thumbnail trend lines attached to draw the user's attention to the several different measures, while emphasizing that change is still the theme. Unfortunately, I was not able to reconcile Tableau's filter and pages shelves to allow users to toggle between years seamlessly. I love that the pages feature allows users to fast forward through the years, which dynamically updates the barchart, map, and political distribution graphs. But the pages shelf does not allow an "all years" page, which is frustrating. I could include both a year filter and a pages button, but then there are two "Year" switches and when the Year filter is set to "all years" and a user toggles the pages, the filter parameter does not change, thus indicating "all years" even as the pages are changing. A user could stop the pages, but then they would have to unselect all years from the filters shelf and reselect it to get back to an "all years" view, which is not intuitive, and thus I decided was a bad design feature. Despite extensive research, I have not yet been able figure this out. I have not seen another dashboard that accomplishes this. I found no results in a google search, and chatGTP-o3 could not figure it out either. I ended up with just a filter switch for the years, with the default set to "all years". Still, it is not immediately apparent that selecting a specific year will automatically update the bar charts, political distributions, and map details. 

What did work well was the several parameter controls and calculated fields that control which data are being visualized. Because I included all of my measures and all of the dimensions in each chart, I used the same parameters and calculated fields (pointing the parameter to the data) for each. 

I'm excited to continue to use this awesome tool in future projects, with an eye to keeping measures and outcomes simple while allowing for more impactful design. 
