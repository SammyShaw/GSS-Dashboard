## General Social Survey (GSS) Dashboard

This repository contains data cleaning scripts, index construction, and visualization materials for my **Tableau dashboard** based on the [General Social Survey (GSS)](https://gss.norc.org/).  
It demonstrates skills in **data preprocessing, index creation, reliability testing, and interactive visualization**.

---

## Getting Started

### Prerequisites

| Package | Version | Description |
|----------|----------|--------------|
| Python | 3.10+ | Primary language for preprocessing |
| pandas | ≥1.5 | Data manipulation and transformation |
| numpy | ≥1.24 | Numerical computing |
| pyreadstat | ≥1.2 | Read SAS-format GSS data files |
| pingouin | ≥0.5 | Reliability analysis (Cronbach’s α) |
| factor_analyzer | ≥0.5 | Exploratory factor analysis |
| scikit-learn | ≥1.3 | Standardization and scaling |
| matplotlib | ≥3.7 | Visual checks and diagnostics |

### Installation

```bash
# Clone this repository
git clone https://github.com/SammyShaw/GSS-Dashboard.git
cd GSS-Dashboard

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


import pandas as pd
import pyreadstat
from sklearn.preprocessing import StandardScaler

# Read raw GSS SAS file (1972–2022 cumulative)
df, meta = pyreadstat.read_sas7bdat("gss7222_r3.sas7bdat")

# Select variables used in the dashboard
columns = [
    "YEAR", "AGE", "SEX", "EDUC", "DEGREE", "RACE", "REALINC", "PRESTG10",
    "HAPPY", "LIFE", "HAPMAR", "HAPCOHAB", "HEALTH", "RELITEN", "GOD", "BIBLE",
    "ATTEND", "PRAY", "TRUST", "HELPFUL", "FAIR", "HRS1", "SATJOB",
    "CONEDUC", "CONFED", "CONMEDIC", "CONARMY", "CONBUS", "CONCLERG",
    "CONFINAN", "CONJUDGE", "CONLABOR", "CONLEGIS", "CONPRESS", "CONSCI", "CONTV"
]
gss = df[columns].copy()

# Example: standardized "Happiness" index
happiness_vars = ["HAPPY", "LIFE", "HAPMAR", "HAPCOHAB"]
gss["happiness_index"] = gss[happiness_vars].mean(axis=1)
scaler = StandardScaler()
gss["happiness_index_z"] = scaler.fit_transform(gss[["happiness_index"]])

# Export a 10% sample (used in this repo)
sample = gss.sample(frac=0.10, random_state=42)
sample.to_csv("data/gss_z_sample.csv", index=False)

```

## Summary & Context

This project demonstrates:
1. **Visualization using Tableau**
2. **Preprocessing and analysis of a large public dataset**

I chose this project to practice Tableau and to represent my background in the social sciences.  
It also celebrates publicly available data and aims to make it more accessible.  

The GSS, funded primarily by the National Science Foundation, has been collecting data on American political opinions and social behaviors since 1972.  
While the analyses here show Americans’ declining trust in institutions alongside rising political polarization, the project also highlights the importance of maintaining federal support for longitudinal public data.

---

## Variable Composition Table

| **Index / Measure** | **Type** | **Description** | **GSS Variables Included** |
|----------------------|----------|-----------------|-----------------------------|
| **Education** | Single Measure | Years of school completed (standardized). | `EDUC` |
| **Health** | Single Measure | Self-reported health, 1 (poor) – 4 (excellent). | `HEALTH` |
| **Happiness Index** | Composite | Average of happiness, life excitement, and relationship happiness. | `HAPPY`, `LIFE`, `HAPMAR`, `HAPCOHAB` |
| **Religiosity Index** | Composite | Attendance, prayer frequency, importance of religion, and belief strength. | `ATTEND`, `PRAY`, `RELITEN`, `GOD`, `BIBLE` |
| **Social Relationships Index** | Composite | Time spent with family, friends, neighbors, and social outings. | `SOCREL`, `SOCFREND`, `SOCOMMUN`, `SOCBAR` |
| **Social Attitudes Index** | Composite | Interpersonal trust and perceptions of fairness/helpfulness. | `TRUST`, `HELPFUL`, `FAIR` |
| **Work-Life Balance Index** | Composite | Product of reversed work hours and job satisfaction for respondents working ≥10 hrs/week. | `HRS1`, `SATJOB` |
| **Quality of Life Index** | Composite | Average of Education, Health, Social Relationships, and Work-Life Balance indices. | Derived from other indices |
| **Confidence in Institutions** | Composite | Average confidence in 13 institutions (reversed coding). | `CONEDUC`, `CONFED`, `CONMEDIC`, `CONARMY`, `CONBUS`, `CONCLERG`, `CONFINAN`, `CONJUDGE`, `CONLABOR`, `CONLEGIS`, `CONPRESS`, `CONSCI`, `CONTV` |
| **Confidence in Government** | Composite | Confidence in federal government and congress. | `CONFED`, `CONLEGIS` |
| **Confidence in Media** | Composite | Confidence in television and the press. | `CONPRESS`, `CONTV` |
| **Polarization** | Aggregate | Standard deviation of selected opinion measures over time. | Derived (no raw columns) |

---

## Visualization

<iframe seamless frameborder="0" src="https://public.tableau.com/views/GSS_2_17387981106750/GSSDashboard?:embed=yes&:display_count=yes&:showVizHome=no" width="1100" height="900"></iframe>

*(Note: automatic scaling in Tableau can distort object sizes when embedded in GitHub Pages. For the most responsive version, visit my [Tableau Public profile](https://public.tableau.com/app/profile/samuel.shaw2748/vizzes):*  
[**50 Years of the General Social Survey**](https://public.tableau.com/views/GSS_2_17387981106750/GSSDashboard?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link) *)


## Key Skills Demonstrated

- Data cleaning and transformation (`pandas`, `numpy`)
- Reliability and factor analysis (`pingouin`, `factor_analyzer`)
- Index construction and standardization
- Data storytelling and Tableau dashboard design
- Large-scale data handling (70k+ rows, 6k+ columns)
- Documentation and portfolio presentation with Markdown and GitHub Pages

---

## Reflections on Tableau Design

Tableau enables powerful data storytelling but requires tradeoffs between **comprehensiveness, clarity, and interactivity**.  
While many dashboards focus on one storyline, this project intentionally embraces breadth — highlighting multiple social trends across 50 years of GSS data.

**Key design choices:**
- Centered trend lines to emphasize temporal change  
- Color-coded index buttons with miniature sparklines  
- Parameter-driven filters and calculated fields for modular control  

Some technical limitations (e.g., `pages` vs `filter` interaction) prevented seamless toggling between “All Years” and yearly playback — a design constraint I plan to revisit in future versions.

---

## Related Links

- **Dashboard:** [50 Years of the General Social Survey](https://public.tableau.com/views/GSS_2_17387981106750/GSSDashboard?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
- **Portfolio:** [https://sammyshaw.github.io/](https://sammyshaw.github.io/)
- **Data Source:** [GSS NORC at the University of Chicago](https://gss.norc.org/)

---

*Author: Samuel Shaw, PhD*  
*Seattle, WA*  
*[LinkedIn](https://www.linkedin.com/in/sammyshaw/)*
