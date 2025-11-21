#!/usr/bin/env python
# coding: utf-8

# # AI Adoption & Job Threat Index Analysis  | Capstone Project
# # Jake Gapusan
# # CIS 480
# # Professor Kashihara

# # I.Introduction 🤖
# ## In this project, I will clean and visualize the gathered data from legitimate sources about how the growth of AI applications such as "ChatGpt" influences numerous fields such as healthcare, education, and the job market. 
# 
# ## I am also obligated to answer the following questions for this capstone project.

# ## Following questions to be answered:
# 
# ### 	•	What do usage and engagement trends of tools like ChatGPT show?
# ### 	•	Which countries and age groups are using it the most?
# ### 	•	Which age groups are using ChatGPT the most?
# 

# # II.Data Process ⏳

# ## Step 1: Cleaning each dataset 🧹

# #### 1A: Clean the data 'chatgpt_user_growth' dataset

# In[17]:


# Step 1A: Clean & Inspect ChatGPT User Growth Dataset

import pandas as pd

# Load dataset (use the exact filename you uploaded)
growth_raw = pd.read_csv("chatgpt_user_growth.csv")

# Inspect the first few rows and column names
print("Preview:")
display(growth_raw.head())

print("\nColumns:")
print(growth_raw.columns.tolist())

# Optional: Convert 'month' column to datetime if present
if 'month' in growth_raw.columns:
    growth_raw['date'] = pd.to_datetime(growth_raw['month'])

# Drop missing values (if any)
growth_clean = growth_raw.dropna()

# Save cleaned version (in same directory)
growth_clean.to_csv("chatgpt_user_growth_clean.csv", index=False)

print("\nCleaned dataset saved as 'chatgpt_user_growth_clean.csv'")
display(growth_clean.head())


# #### 1B: Clean the data 'Public_Opinion' dataset

# In[27]:


import pandas as pd

# Load raw dataset
df = pd.read_csv("Public_Opinion_Dataset.csv")

# Standardize column names a bit
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

print("Columns after basic cleaning:\n", df.columns.tolist())

# --- Find key columns dynamically ---

# 1) Column with "% of respondents that agree"
agree_cols = [c for c in df.columns if "respondents" in c and "agree" in c]
print("\nCandidate 'agree' columns:", agree_cols)
agree_col = agree_cols[0]   # pick the first match

# 2) Opinion text column
opinion_cols = [c for c in df.columns if "opinion_on_products" in c]
print("Candidate 'opinion' columns:", opinion_cols)
opinion_col = opinion_cols[0]

# 3) Country column
country_cols = [c for c in df.columns if "country" in c]
print("Candidate 'country' columns:", country_cols)
country_col = country_cols[0]

# 4) Year & source columns (should exist already)
year_col = [c for c in df.columns if "year" in c][0]
source_col = [c for c in df.columns if "source_file" in c][0]

print("\nUsing columns:")
print("agree_col :", agree_col)
print("opinion_col :", opinion_col)
print("country_col :", country_col)
print("year_col :", year_col)
print("source_col :", source_col)

# --- Split US vs International ---

# US rows = no country value
us_df = df[df[country_col].isna()][[
    agree_col,
    opinion_col,
    year_col,
    source_col
]].dropna(subset=[agree_col])

# International rows = have country
intl_df = df[df[country_col].notna()][[
    country_col,
    agree_col,
    year_col,
    source_col
]].dropna(subset=[country_col])

# Save cleaned versions
us_df.to_csv("Public_Opinion_US_clean.csv", index=False)
intl_df.to_csv("Public_Opinion_International_clean.csv", index=False)

print("\nUS survey shape:", us_df.shape)
print("International survey shape:", intl_df.shape)

display(us_df.head())
display(intl_df.head())


# #### 1C: Clean the data 'clean_chatgpt_reviews' dataset

# In[29]:


import pandas as pd

df = pd.read_csv("clean_chatgpt_reviews.csv", encoding='utf-8')

print("Preview:")
display(df.head(10))

print("\nColumns:", df.columns.tolist())
print("\nShape before cleaning:", df.shape)

# Drop fully empty rows or columns
df = df.dropna(how="all")           # remove empty rows
df = df.dropna(axis=1, how="all")   # remove empty columns

# Remove duplicates
df = df.drop_duplicates()

print("Shape after removing empty/duplicate rows:", df.shape)

# Standardize column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

print("\nStandardized Columns:")
print(df.columns.tolist())

df.to_csv("chatgpt_reviews_clean.csv", index=False)
print("Saved as chatgpt_reviews_clean.csv")


# #### 1D: Clean the data 'country_share' & 'monthly_visit' dataset

# In[30]:


import pandas as pd

df_country = pd.read_csv("chatgpt_country_share.csv")

print("Preview:")
display(df_country.head())

print("\nColumns:", df_country.columns.tolist())
print("Shape before cleaning:", df_country.shape)

# Drop duplicate rows
df_country = df_country.drop_duplicates()

# Drop fully empty rows
df_country = df_country.dropna(how="all")

# Standardize column names
df_country.columns = [c.strip().lower().replace(" ", "_") for c in df_country.columns]

print("\nCleaned Columns:", df_country.columns.tolist())
print("Shape after cleaning:", df_country.shape)

df_country.to_csv("chatgpt_country_share_clean.csv", index=False)
print("\nSaved as: chatgpt_country_share_clean.csv")


# In[31]:


df_visits = pd.read_csv("chatgpt_monthly_visits.csv")

print("Preview:")
display(df_visits.head())

print("\nColumns:", df_visits.columns.tolist())
print("Shape before cleaning:", df_visits.shape)

# standard cleaning
df_visits = df_visits.drop_duplicates()
df_visits = df_visits.dropna(how="all")

# normalize col names
df_visits.columns = [c.strip().lower().replace(" ", "_") for c in df_visits.columns]

# if month exists as text → keep it
# if visits include commas → fix it
if 'visits' in df_visits.columns:
    df_visits['visits'] = (
        df_visits['visits']
        .astype(str)
        .str.replace(",", "")
        .astype(float)
    )

print("\nColumns After standardizing:", df_visits.columns.tolist())
print("Shape after cleaning:", df_visits.shape)

df_visits.to_csv("chatgpt_monthly_visits_clean.csv", index=False)
print("\nSaved as: chatgpt_monthly_visits_clean.csv")


# #### 1E: Clean the data 'job_threat_index'

# In[37]:


import pandas as pd

# Step 1E: Clean job_threat_index dataset

# 1. Load raw dataset
df_jobs = pd.read_csv("job_threat_index.csv")

print("Original columns:")
print(df_jobs.columns.tolist())
print("Original shape:", df_jobs.shape)

# 2. Drop fully empty rows/columns
df_jobs = df_jobs.dropna(how="all")
df_jobs = df_jobs.dropna(axis=1, how="all")

# 3. Drop duplicate rows
df_jobs = df_jobs.drop_duplicates()

# 4. Standardize column names (lowercase, underscores)
df_jobs.columns = [c.strip().lower().replace(" ", "_") for c in df_jobs.columns]

# 👉 Fix known typo: 'job_titiles' → 'job_titles'
df_jobs = df_jobs.rename(columns={"job_titiles": "job_titles"})

print("\nStandardized columns (after manual correction):")
print(df_jobs.columns.tolist())
print("Shape after basic cleaning:", df_jobs.shape)

# 5. Auto-detect the AI risk / threat column
#    We look for any column whose name contains one of these keywords.
risk_keywords = ["risk", "threat", "score", "impact", "exposure", "autom", "ai"]

possible_risk_cols = [
    c for c in df_jobs.columns
    if any(k in c.lower() for k in risk_keywords)
]

print("\nPossible AI-risk related columns:", possible_risk_cols)

risk_col = None
if possible_risk_cols:
    risk_col = possible_risk_cols[0]
    print(f"\nUsing '{risk_col}' as AI risk column.")

    # 6. Convert risk column to numeric (remove %, commas, text)
    if df_jobs[risk_col].dtype == object:
        df_jobs[risk_col] = (
            df_jobs[risk_col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.extract(r"([\d\.]+)", expand=False)  # keep only number-like part
            .astype(float)
        )

    # 7. Sort by highest risk first
    df_jobs = df_jobs.sort_values(by=risk_col, ascending=False)
else:
    print("\n⚠ WARNING: No AI risk column auto-detected.")
    print("Check the printed column names above and manually set 'risk_col' if needed.")
    # Example if you see something like 'ai_threat_index':
    # risk_col = 'ai_threat_index'
    # df_jobs[risk_col] = ...

# 8. Save cleaned dataset
df_jobs.to_csv("job_threat_index_clean.csv", index=False)
print("\nCleaned dataset saved as 'job_threat_index_clean.csv'")

print("\nPreview of cleaned data:")
display(df_jobs.head(10))


# #### 1F: Clean the data 'Normalized_Average_AI_Impact_by_Domain'

# In[38]:


import pandas as pd

# Step 1F: Clean PolicyMap Education dataset

# 1. Load raw dataset
edu_raw = pd.read_csv("policymap_education_county.csv")  # change name if yours is different

print("Original columns:")
print(edu_raw.columns.tolist())
print("Original shape:", edu_raw.shape)

# 2. Basic cleanup: drop fully empty rows/columns
edu = edu_raw.dropna(how="all").dropna(axis=1, how="all")

# 3. Standardize column names (for easier matching)
edu.columns = [c.strip() for c in edu.columns]

print("\nStandardized columns:")
print(edu.columns.tolist())

# 4. Detect key columns

# FIPS column: usually "Formatted GeoID" or something similar
fips_candidates = [c for c in edu.columns if "formatted geoid" in c.lower() or "geoid" in c.lower() or "fips" in c.lower()]
print("\nFIPS candidates:", fips_candidates)
FIPS_COL = fips_candidates[0] if fips_candidates else None

# Geography / County name
name_candidates = [c for c in edu.columns if "geography name" in c.lower() or "name" in c.lower()]
print("Name candidates:", name_candidates)
NAME_COL = name_candidates[0] if name_candidates else None

# State column (often "Sits in State")
state_candidates = [c for c in edu.columns if "state" in c.lower()]
print("State candidates:", state_candidates)
STATE_COL = state_candidates[0] if state_candidates else None

# Education value column: look for 'bachelor' or 'percent'
value_candidates = [c for c in edu.columns if "bachelor" in c.lower() or "percent" in c.lower()]
print("Education value candidates:", value_candidates)
VALUE_COL = value_candidates[0] if value_candidates else None

print("\nUsing columns:")
print("FIPS_COL  :", FIPS_COL)
print("NAME_COL  :", NAME_COL)
print("STATE_COL :", STATE_COL)
print("VALUE_COL :", VALUE_COL)

# 5. Build cleaned dataframe

selected_cols = {}

if FIPS_COL:
    selected_cols["FIPS"] = edu[FIPS_COL].astype(str).str.zfill(5)

if NAME_COL:
    selected_cols["County_Name"] = edu[NAME_COL]

if STATE_COL:
    selected_cols["State"] = edu[STATE_COL]

if VALUE_COL:
    selected_cols["Bachelors_Degree_Rate"] = edu[VALUE_COL]

edu_clean = pd.DataFrame(selected_cols)

# Drop rows where FIPS or value is missing
if "FIPS" in edu_clean.columns and "Bachelors_Degree_Rate" in edu_clean.columns:
    edu_clean = edu_clean.dropna(subset=["FIPS", "Bachelors_Degree_Rate"])

print("\nCleaned education dataframe shape:", edu_clean.shape)
display(edu_clean.head())

# 6. Save cleaned version
edu_clean.to_csv("policymap_education_clean.csv", index=False)
print("\nSaved as 'policymap_education_clean.csv'")


# #### 1G: Clean the data 'policymap_education_county'

# In[39]:


import pandas as pd

# Step 1G: Clean PolicyMap Income dataset
income_raw = pd.read_csv("policymap_income_county.csv")

print("Original columns:")
print(income_raw.columns.tolist())
print("Original shape:", income_raw.shape)

# Drop fully empty rows/cols
inc = income_raw.dropna(how="all").dropna(axis=1, how="all")

# Standardize names a bit
inc.columns = [c.strip() for c in inc.columns]

print("\nStandardized columns:")
print(inc.columns.tolist())

# Detect FIPS / Name / State / Income value

fips_candidates = [c for c in inc.columns if "formatted geoid" in c.lower() or "geoid" in c.lower() or "fips" in c.lower()]
print("\nFIPS candidates:", fips_candidates)
FIPS_COL = fips_candidates[0] if fips_candidates else None

name_candidates = [c for c in inc.columns if "geography name" in c.lower() or "name" in c.lower()]
print("Name candidates:", name_candidates)
NAME_COL = name_candidates[0] if name_candidates else None

state_candidates = [c for c in inc.columns if "state" in c.lower()]
print("State candidates:", state_candidates)
STATE_COL = state_candidates[0] if state_candidates else None

value_candidates = [c for c in inc.columns if "median household income" in c.lower() or "income" in c.lower()]
print("Income value candidates:", value_candidates)
VALUE_COL = value_candidates[0] if value_candidates else None

print("\nUsing columns:")
print("FIPS_COL  :", FIPS_COL)
print("NAME_COL  :", NAME_COL)
print("STATE_COL :", STATE_COL)
print("VALUE_COL :", VALUE_COL)

# Build cleaned dataframe
selected_cols = {}

if FIPS_COL:
    selected_cols["FIPS"] = inc[FIPS_COL].astype(str).str.zfill(5)

if NAME_COL:
    selected_cols["County_Name"] = inc[NAME_COL]

if STATE_COL:
    selected_cols["State"] = inc[STATE_COL]

if VALUE_COL:
    selected_cols["Median_Income"] = inc[VALUE_COL]

income_clean = pd.DataFrame(selected_cols)

# Drop rows missing FIPS or income
if "FIPS" in income_clean.columns and "Median_Income" in income_clean.columns:
    income_clean = income_clean.dropna(subset=["FIPS", "Median_Income"])

print("\nCleaned income dataframe shape:", income_clean.shape)
display(income_clean.head())

# Save
income_clean.to_csv("policymap_income_clean.csv", index=False)
print("\nSaved as 'policymap_income_clean.csv'")


# #### 1H: Clean the data 'policymap_income_county'

# In[40]:


# Step 1H: Clean PolicyMap Unemployment dataset
unemp_raw = pd.read_csv("policymap_unemployment_county.csv")

print("Original columns:")
print(unemp_raw.columns.tolist())
print("Original shape:", unemp_raw.shape)

# Drop fully empty rows/cols
unemp = unemp_raw.dropna(how="all").dropna(axis=1, how="all")

# Standardize names
unemp.columns = [c.strip() for c in unemp.columns]

print("\nStandardized columns:")
print(unemp.columns.tolist())

# Detect FIPS / Name / State / Unemployment value

fips_candidates = [c for c in unemp.columns if "formatted geoid" in c.lower() or "geoid" in c.lower() or "fips" in c.lower()]
print("\nFIPS candidates:", fips_candidates)
FIPS_COL = fips_candidates[0] if fips_candidates else None

name_candidates = [c for c in unemp.columns if "geography name" in c.lower() or "name" in c.lower()]
print("Name candidates:", name_candidates)
NAME_COL = name_candidates[0] if name_candidates else None

state_candidates = [c for c in unemp.columns if "state" in c.lower()]
print("State candidates:", state_candidates)
STATE_COL = state_candidates[0] if state_candidates else None

value_candidates = [c for c in unemp.columns if "unemployment rate" in c.lower() or "unemployment" in c.lower()]
print("Unemployment value candidates:", value_candidates)
VALUE_COL = value_candidates[0] if value_candidates else None

print("\nUsing columns:")
print("FIPS_COL  :", FIPS_COL)
print("NAME_COL  :", NAME_COL)
print("STATE_COL :", STATE_COL)
print("VALUE_COL :", VALUE_COL)

# Build cleaned dataframe
selected_cols = {}

if FIPS_COL:
    selected_cols["FIPS"] = unemp[FIPS_COL].astype(str).str.zfill(5)

if NAME_COL:
    selected_cols["County_Name"] = unemp[NAME_COL]

if STATE_COL:
    selected_cols["State"] = unemp[STATE_COL]

if VALUE_COL:
    selected_cols["Unemployment_Rate"] = unemp[VALUE_COL]

unemp_clean = pd.DataFrame(selected_cols)

# Drop rows missing FIPS or value
if "FIPS" in unemp_clean.columns and "Unemployment_Rate" in unemp_clean.columns:
    unemp_clean = unemp_clean.dropna(subset=["FIPS", "Unemployment_Rate"])

print("\nCleaned unemployment dataframe shape:", unemp_clean.shape)
display(unemp_clean.head())

# Save
unemp_clean.to_csv("policymap_unemployment_clean.csv", index=False)
print("\nSaved as 'policymap_unemployment_clean.csv'")


# #### 1I: Clean the data 'policymap_unemployment_county'

# In[41]:


# Step 1H: Clean PolicyMap Unemployment dataset
unemp_raw = pd.read_csv("policymap_unemployment_county.csv")

print("Original columns:")
print(unemp_raw.columns.tolist())
print("Original shape:", unemp_raw.shape)

# Drop fully empty rows/cols
unemp = unemp_raw.dropna(how="all").dropna(axis=1, how="all")

# Standardize names
unemp.columns = [c.strip() for c in unemp.columns]

print("\nStandardized columns:")
print(unemp.columns.tolist())

# Detect FIPS / Name / State / Unemployment value

fips_candidates = [c for c in unemp.columns if "formatted geoid" in c.lower() or "geoid" in c.lower() or "fips" in c.lower()]
print("\nFIPS candidates:", fips_candidates)
FIPS_COL = fips_candidates[0] if fips_candidates else None

name_candidates = [c for c in unemp.columns if "geography name" in c.lower() or "name" in c.lower()]
print("Name candidates:", name_candidates)
NAME_COL = name_candidates[0] if name_candidates else None

state_candidates = [c for c in unemp.columns if "state" in c.lower()]
print("State candidates:", state_candidates)
STATE_COL = state_candidates[0] if state_candidates else None

value_candidates = [c for c in unemp.columns if "unemployment rate" in c.lower() or "unemployment" in c.lower()]
print("Unemployment value candidates:", value_candidates)
VALUE_COL = value_candidates[0] if value_candidates else None

print("\nUsing columns:")
print("FIPS_COL  :", FIPS_COL)
print("NAME_COL  :", NAME_COL)
print("STATE_COL :", STATE_COL)
print("VALUE_COL :", VALUE_COL)

# Build cleaned dataframe
selected_cols = {}

if FIPS_COL:
    selected_cols["FIPS"] = unemp[FIPS_COL].astype(str).str.zfill(5)

if NAME_COL:
    selected_cols["County_Name"] = unemp[NAME_COL]

if STATE_COL:
    selected_cols["State"] = unemp[STATE_COL]

if VALUE_COL:
    selected_cols["Unemployment_Rate"] = unemp[VALUE_COL]

unemp_clean = pd.DataFrame(selected_cols)

# Drop rows missing FIPS or value
if "FIPS" in unemp_clean.columns and "Unemployment_Rate" in unemp_clean.columns:
    unemp_clean = unemp_clean.dropna(subset=["FIPS", "Unemployment_Rate"])

print("\nCleaned unemployment dataframe shape:", unemp_clean.shape)
display(unemp_clean.head())

# Save
unemp_clean.to_csv("policymap_unemployment_clean.csv", index=False)
print("\nSaved as 'policymap_unemployment_clean.csv'")


# ## Step 2: Merge & master Tables 🧮

# #### 2A: Merge ChatGPT usage datasets

# In[58]:


import pandas as pd

# Step 2A: Merge ChatGPT usage datasets
# 1. Load cleaned datasets
growth = pd.read_csv("chatgpt_user_growth_clean.csv")
visits = pd.read_csv("chatgpt_monthly_visits_clean.csv")

print("Growth columns:", growth.columns.tolist())
print("Visits columns:", visits.columns.tolist())
print("Growth shape:", growth.shape)
print("Visits shape:", visits.shape)

# 2. Detect date/month columns
growth_date_candidates = [c for c in growth.columns if "month" in c.lower() or "date" in c.lower()]
visits_date_candidates = [c for c in visits.columns if "month" in c.lower() or "date" in c.lower()]

if not growth_date_candidates or not visits_date_candidates:
    raise ValueError(f"Could not find date/month columns.\n"
                     f"Growth columns: {growth.columns.tolist()}\n"
                     f"Visits columns: {visits.columns.tolist()}")

growth_date_col = growth_date_candidates[0]
visits_date_col = visits_date_candidates[0]

print(f"\nUsing growth date column: {growth_date_col}")
print(f"Using visits date column: {visits_date_col}")

print("\nSample growth dates:")
print(growth[growth_date_col].head())
print("\nSample visits dates:")
print(visits[visits_date_col].head())

# 3. Helper function: try multiple date formats safely
def parse_dates(series):
    # Common possible formats
    formats = [
        "%Y-%m-%d",  # 2023-01-01
        "%Y-%m",     # 2023-01
        "%m/%d/%Y",  # 01/15/2023
        "%Y/%m/%d",  # 2023/01/15
        "%b %Y",     # Jan 2023
        "%B %Y",     # January 2023
    ]
    for fmt in formats:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="raise")
            print(f"Parsed dates successfully with format: {fmt}")
            return parsed
        except Exception:
            continue
    # Fallback: let pandas infer if none of the above worked
    print("⚠ Could not match known formats; falling back to generic parser.")
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

# 4. Parse dates robustly
growth["date"] = parse_dates(growth[growth_date_col])
visits["date"] = parse_dates(visits[visits_date_col])

# Drop rows where date couldn't be parsed
growth = growth.dropna(subset=["date"])
visits = visits.dropna(subset=["date"])

print("\nAfter parsing & dropping NaT:")
print("Growth shape:", growth.shape)
print("Visits shape:", visits.shape)

# 5. Optionally sort by date
growth = growth.sort_values("date")
visits = visits.sort_values("date")

# 6. Merge on date (inner join keeps common dates only)
growth_merged = growth.merge(visits, on="date", how="inner")

print("\nMerged growth & visits shape:", growth_merged.shape)
display(growth_merged.head())

# 7. Save merged dataset
growth_merged.to_csv("chatgpt_growth_and_visits.csv", index=False)
print("\nSaved merged file as 'chatgpt_growth_and_visits.csv'")


# #### 2B: Merge policy data with job threat index  

# In[62]:


import pandas as pd

# 2B: Merge PolicyMap data into a single county_features table
# Load the datasets that needs to be merged
edu = pd.read_csv("policymap_education_county_clean.csv")
income = pd.read_csv("policymap_income_county_clean.csv")
unemp = pd.read_csv("policymap_unemployment_county_clean.csv")

# Print the data
print("Education columns:", edu.columns.tolist())
print("Income columns   :", income.columns.tolist())
print("Unemp columns    :", unemp.columns.tolist())

# ---- Helper to find columns by keyword ----
def find_col(cols, keywords):
    cols = list(cols)
    cols_low = [c.lower() for c in cols]
    for i, c in enumerate(cols_low):
        if all(k in c for k in keywords):
            return cols[i]
    return None

# ---- Detect FIPS/GEOID column in each df and create a standard 'FIPS' ----
def add_fips_column(df, label):
    fips_col = (
        find_col(df.columns, ["fips"]) or
        find_col(df.columns, ["geoid"]) or
        find_col(df.columns, ["geo_id"])
    )
    if fips_col is None:
        raise ValueError(f" Could not find a FIPS/GEOID column in {label}: {df.columns.tolist()}")
    df["FIPS"] = df[fips_col].astype(str).str.zfill(5)
    print(f"{label}: using '{fips_col}' as FIPS source")
    return df

edu   = add_fips_column(edu,   "Education")
income = add_fips_column(income, "Income")
unemp  = add_fips_column(unemp,  "Unemployment")

# ---- Identify key columns in education (base table) ----
county_col = (
    find_col(edu.columns, ["county"]) or
    find_col(edu.columns, ["geography", "name"]) or
    find_col(edu.columns, ["name"])
)
state_col = find_col(edu.columns, ["state"])
edu_val_col = (
    find_col(edu.columns, ["bachelor"]) or
    find_col(edu.columns, ["degree"]) or
    find_col(edu.columns, ["education"]) or
    find_col(edu.columns, ["college"])
)

print("\nEducation base columns chosen:")
print("County name column    ->", county_col)
print("State column          ->", state_col)
print("Education value col   ->", edu_val_col)

# ---- Identify income and unemployment value columns ----
inc_val_col = (
    find_col(income.columns, ["median", "income"]) or
    find_col(income.columns, ["income"])
)
print("\nIncome value column   ->", inc_val_col)

unemp_val_col = (
    find_col(unemp.columns, ["unemployment"]) or
    find_col(unemp.columns, ["unemploy"])
)
print("Unemployment value col ->", unemp_val_col)

# ---- Build base county_features from education ----
data = {
    "FIPS": edu["FIPS"]
}
if county_col:
    data["County_Name"] = edu[county_col]
if state_col:
    data["State"] = edu[state_col]
if edu_val_col:
    data["Bachelors_Degree_Rate"] = edu[edu_val_col]

county_features = pd.DataFrame(data)

# ---- Merge income (if found) ----
if inc_val_col:
    income_small = income[["FIPS", inc_val_col]].rename(
        columns={inc_val_col: "Median_Income"}
    )
    county_features = county_features.merge(income_small, on="FIPS", how="left")
else:
    print("\n⚠ Skipping income merge (no income value column found).")

# ---- Merge unemployment (if found) ----
if unemp_val_col:
    unemp_small = unemp[["FIPS", unemp_val_col]].rename(
        columns={unemp_val_col: "Unemployment_Rate"}
    )
    county_features = county_features.merge(unemp_small, on="FIPS", how="left")
else:
    print("\n⚠ Skipping unemployment merge (no unemployment value column found).")

# ---- Final cleanup & save ----
county_features = county_features.dropna(subset=["FIPS"])

print("\nFinal county_features shape:", county_features.shape)
display(county_features.head())

county_features.to_csv("county_features.csv", index=False)
print("\nSaved merged dataset as 'county_features.csv'")


# #### 2C: Build AI-Socioeconomic modeling dataset

# In[68]:


import pandas as pd

# Load the cleaned job threat file
jobs = pd.read_csv("job_threat_index_clean.csv")
print("Jobs shape:", jobs.shape)
print("Columns:", jobs.columns.tolist())

# Identify key columns
title_col = [c for c in jobs.columns
             if "title" in c.lower() or ("job" in c.lower() and "id" not in c.lower())][0]

risk_col = [c for c in jobs.columns
            if any(k in c.lower() for k in ["risk", "impact", "threat", "ai"])][0]

# Optional extra columns if they exist
extra_cols = []
for key in ["tasks", "ai_models", "ai_workload_ratio", "domain"]:
    matches = [c for c in jobs.columns if key in c.lower()]
    if matches:
        extra_cols.append(matches[0])

print("\nUsing:")
print("  job title column:", title_col)
print("  risk column     :", risk_col)
print("  extra columns   :", extra_cols)

# Build a detailed job-level dataset
cols_to_keep = [title_col, risk_col] + extra_cols
jobs_detailed = jobs[cols_to_keep].copy()
jobs_detailed.rename(columns={risk_col: "AI_Risk_Score"}, inplace=True)

# (Optional) add a risk band for easier visuals
jobs_detailed["Risk_Band"] = pd.cut(
    jobs_detailed["AI_Risk_Score"],
    bins=[0, 33, 66, 100],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

print("\nDetailed AI job-risk dataset preview:")
display(jobs_detailed.head())

# Save it
jobs_detailed.to_csv("ai_job_risk_detailed.csv", index=False)
print("\n✅ Saved dataset: 'ai_job_risk_detailed.csv'")


# # III. How AI Apps such as Chatgpt influences today's technology. 💻

# ## Question 1: How fast is ChatGPT growing over time? 

# In[70]:


# ==============================
# Part III — Q1: ChatGPT Growth Over Time
# ==============================

import pandas as pd
import matplotlib.pyplot as plt

# 3A — Load dataset
growth = pd.read_csv("chatgpt_growth_and_visits.csv")

print("Growth dataset preview:")
display(growth.head())
print("\nShape:", growth.shape)
print("Columns:", growth.columns.tolist())

# 3B — Parse dates & sort
growth['date'] = pd.to_datetime(growth['date'])
growth = growth.sort_values("date")


# ### Visualize the Data

# In[71]:


# 3C — Plot: User growth
plt.figure(figsize=(10,5))
plt.plot(growth['date'], growth['Users (millions)'], marker='o')
plt.title("ChatGPT User Growth Over Time")
plt.xlabel("Date")
plt.ylabel("Users (millions)")
plt.grid(True)
plt.show()

# 3D — Plot: Monthly visits
plt.figure(figsize=(10,5))
plt.plot(growth['date'], growth['monthly_visits_(millions)'], marker='s', color='orange')
plt.title("Monthly Visits to ChatGPT")
plt.xlabel("Date")
plt.ylabel("Visits (millions)")
plt.grid(True)
plt.show()

# 3E — Combined Plot
plt.figure(figsize=(10,6))
plt.plot(growth['date'], growth['Users (millions)'], marker='o', label="Users")
plt.plot(growth['date'], growth['monthly_visits_(millions)'], marker='s', label="Visits")
plt.title("ChatGPT Adoption & Usage Over Time")
plt.xlabel("Date")
plt.ylabel("Millions")
plt.legend()
plt.grid(True)
plt.show()

# 3F — Growth stats
start_users = growth['Users (millions)'].iloc[0]
end_users = growth['Users (millions)'].iloc[-1]

growth_rate = end_users - start_users
growth_percent = (growth_rate / start_users) * 100

print(f"\nChatGPT users grew from {start_users}M to {end_users}M.")
print(f"Total growth: +{growth_rate}M users")
print(f"Relative growth: {growth_percent:.2f}%")

# 3G — Month-to-month growth
growth['User_Growth'] = growth['Users (millions)'].diff()
growth['Visit_Growth'] = growth['monthly_visits_(millions)'].diff()

print("\nMonth-to-Month Growth:")
display(growth[['date', 'User_Growth', 'Visit_Growth']])


# ### Insight

# ### ChatGPT experienced explosive growth throughout 2023.
# ### Based on the dataset, the platform grew from 100 million users in December 2022 to 210 million users by August 2023, representing a 110% increase in less than a year.
# ### Monthly visits also climbed from 60 million to 130 million, showing a parallel rise in engagement.
# 
# ### This trend indicates that ChatGPT is one of the fastest-growing AI platforms in history, rapidly reshaping how users interact with artificial intelligence tools.

# ## Question 2: Which age groups are using ChatGPT the most?

# In[73]:


# Part III — Q2: ChatGPT Age Demographics
import pandas as pd
import matplotlib.pyplot as plt

# 2A — Load dataset
age_df = pd.read_csv("chatgpt_age_demographics.csv")

print("Age demographics dataset preview:")
display(age_df.head())
print("\nShape:", age_df.shape)
print("Columns:", age_df.columns.tolist())

# 2B — Clean column names
age_df.columns = [c.strip().replace(" ", "_").lower() for c in age_df.columns]

# Look for % column
pct_col = [c for c in age_df.columns if "%" in c or "percent" in c or "share" in c][0]
age_col = [c for c in age_df.columns if "age" in c][0]

print("\nUsing columns:")
print("Age column:", age_col)
print("Percent column:", pct_col)

# Clean numeric %
age_df[pct_col] = (
    age_df[pct_col]
    .astype(str)
    .str.replace("%", "")
    .str.strip()
    .astype(float)
)

# 2C — Sort descending
age_sorted = age_df.sort_values(pct_col, ascending=False)

# 2D — Plot
plt.figure(figsize=(8,5))
plt.bar(age_sorted[age_col], age_sorted[pct_col])
plt.title("ChatGPT Usage by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Percent of Respondents (%)")
plt.grid(axis='y')
plt.show()

# 2E — Print highest usage group
top_age = age_sorted.iloc[0]
print(f"\nHighest usage age group: {top_age[age_col]} at {top_age[pct_col]}%")

print("\nFull ranking:")
display(age_sorted)


# ### Survey data shows that younger users dominate ChatGPT usage, with the 18–29 age group having the highest adoption rate (often 40–55% depending on the source).
# 
# ### Usage gradually declines with age:
# ### • 30–44 remain active but lower
# ### • 45–60 drops more
# ### • 60+ has minimal adoption
# 
# ### This confirms that Gen Z & younger Millennials are the power users driving AI conversational tool growth.

# ## Question 3: Which countries are using ChatGPT the most?

# In[74]:


# Part III — Q3: ChatGPT Country Adoption
import pandas as pd
import matplotlib.pyplot as plt

# 3A — Load dataset
country_df = pd.read_csv("chatgpt_country_share_clean.csv")

print("Country dataset preview:")
display(country_df.head())
print("\nShape:", country_df.shape)
print("Columns:", country_df.columns.tolist())

# 3B — Standardize column names
country_df.columns = [c.strip().lower().replace(" ", "_") for c in country_df.columns]

# Identify columns
country_col = [c for c in country_df.columns if "country" in c][0]
pct_col = [c for c in country_df.columns if "%" in c or "percent" in c or "share" in c][0]

print("\nDetected columns:")
print("Country column:", country_col)
print("Percent column:", pct_col)

# Clean % values
country_df[pct_col] = (
    country_df[pct_col]
    .astype(str)
    .str.replace("%", "")
    .str.strip()
    .astype(float)
)

# 3C — Sort descending
sorted_countries = country_df.sort_values(pct_col, ascending=False)

# 3D — Plot top 10
top10 = sorted_countries.head(10)

plt.figure(figsize=(10,5))
plt.bar(top10[country_col], top10[pct_col])
plt.title("Top 10 Countries Using ChatGPT")
plt.xlabel("Country")
plt.ylabel("Usage %")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y")
plt.show()

# 3E — Print strongest adopters
print("\nTop 5 Countries Using ChatGPT:")
display(sorted_countries.head(5))

print("\nBottom 5 Countries:")
display(sorted_countries.tail(5))


# #### According to global usage data, ChatGPT adoption is not evenly distributed across countries.
# #### India leads the world, with 11.7% of the population reporting that they have used ChatGPT — more than any other country. The United States follows at 9.5%, while Indonesia (7.2%) and the Philippines (5.9%) demonstrate particularly strong adoption in Southeast Asia. Brazil (5.4%) represents the highest adoption rate in Latin America.
# 
# #### This suggests that ChatGPT is gaining rapid traction in emerging economies, especially where:
# #### 	•	English proficiency is high
# #### 	•	digitally-native populations are growing
# #### 	•	access to AI tools is improving
# 
# #### The data reveals that ChatGPT is not just a Western phenomenon — it is global, driven especially by students, freelancers, and tech workers in developing regions.
# 

# ## Question 4: What are the top reasons why people use ChatGPT?

# In[75]:


import pandas as pd

df = pd.read_csv("Public_Opinion_US_clean.csv")  # or International

df.head()


# In[78]:


# Correct column names
agree_col = '%_of_respondents_that_“agree”'
reason_col = 'opinion_on_products_and_services_using_ai'

# Sort by % who agree
df_sorted = df.sort_values(by=agree_col, ascending=False)

top_reasons = df_sorted[[agree_col, reason_col]].head(10)

print("Top reasons people use ChatGPT/AI tools:")
display(top_reasons)


# ## Bonus: Predicting Future ChatGPT Users (Next 12 Months)

# In[79]:


# Bonus: Predicting Future ChatGPT Users (Next 12 Months)
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load & prep data
growth = pd.read_csv("chatgpt_growth_and_visits.csv")
growth['date'] = pd.to_datetime(growth['date'])
growth = growth.sort_values("date")

print("Historical growth data:")
display(growth)

# Use an index as "time" (0, 1, 2, ..., n-1)
growth['t'] = np.arange(len(growth))

X = growth['t'].values
y = growth['Users (millions)'].values

# 2. Fit a simple linear trend model: Users = a*t + b
coeffs = np.polyfit(X, y, deg=1)
a, b = coeffs
print(f"\nTrend line: Users ≈ {a:.2f} * t + {b:.2f}")

# Compute in-sample R^2 to see how well the line fits
y_pred = a * X + b
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"R² (fit quality): {r2:.3f}")

# 3. Create next 12 months of dates
last_date = growth['date'].max()
future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=12, freq='MS')

# Time index for future months (continue sequence)
t_future = np.arange(len(growth), len(growth) + 12)

# Predict future users
future_users = a * t_future + b

# 4. Build forecast DataFrame
future_df = pd.DataFrame({
    "date": future_dates,
    "Predicted_Users_Millions": future_users
})

print("\nNext 12-month forecast:")
display(future_df)

# 5. Combine historical + forecast for plotting
hist_plot_df = growth[['date', 'Users (millions)']].copy()
hist_plot_df.rename(columns={"Users (millions)": "Users_Millions"}, inplace=True)

plt.figure(figsize=(10,6))
plt.plot(hist_plot_df['date'], hist_plot_df['Users_Millions'],
         marker='o', label='Historical Users')
plt.plot(future_df['date'], future_df['Predicted_Users_Millions'],
         marker='s', linestyle='--', label='Forecast (Next 12 Months)')

plt.title("ChatGPT Users: Historical Growth and 12-Month Forecast")
plt.xlabel("Date")
plt.ylabel("Users (millions)")
plt.legend()
plt.grid(True)
plt.show()


# ### To estimate future adoption, I fit a simple linear trend model using monthly user counts from December 2022 through August 2023. Time (in months) was used as the predictor, and total users (in millions) was the response. The model achieved an R² of [your printed value], indicating that a large share of the variance in user growth is explained by a stable upward trend.
# 
# ### Using this model, I forecasted ChatGPT’s user base for the next 12 months. If the historical trend continues, the model projects that ChatGPT could reach approximately X–Y million users within a year. This forecast highlights how quickly generative AI tools can scale globally, though real-world growth may deviate based on competition, regulation, and market saturation.

# # IV. Job-Threat Index 🚨

# ## This data shows what jobs that could be possibly replaced by AI in the upcoming years

# ### 1) Top 5 Jobs that are in danger of AI replacement

# In[81]:


# Job-Threat Analysis: Top 5 High-Risk Jobs
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the detailed file
jobs = pd.read_csv("ai_job_risk_detailed.csv")

print("Job dataset preview:")
display(jobs.head())

# Identify the AI risk score column
risk_col = "AI_Risk_Score" if "AI_Risk_Score" in jobs.columns else jobs.columns[1]
title_col = jobs.columns[0]  # first column is job title

# Sort by AI risk descending
high_risk_jobs = jobs.sort_values(by=risk_col, ascending=False).head(5)

print("\nTop 5 Jobs Most at Risk of AI Replacement:")
display(high_risk_jobs)


# In[82]:


# Plot
plt.figure(figsize=(10,5))
plt.barh(high_risk_jobs[title_col], high_risk_jobs[risk_col], color='red')
plt.title("Top 5 Jobs Most at Risk of AI Replacement")
plt.xlabel("AI Risk Score")
plt.ylabel("Job Title")
plt.gca().invert_yaxis()
plt.grid(axis='x')
plt.show()


# ### 2) Top 5 Jobs that are least replaced by AI 

# In[83]:


# Job-Threat Analysis: Top 5 Safest Jobs from AI
import pandas as pd
import matplotlib.pyplot as plt

# Load the detailed job dataset
jobs = pd.read_csv("ai_job_risk_detailed.csv")

# Identify columns
risk_col = "AI_Risk_Score" if "AI_Risk_Score" in jobs.columns else jobs.columns[1]
title_col = jobs.columns[0]

# Sort ascending (safest jobs first)
low_risk_jobs = jobs.sort_values(by=risk_col, ascending=True).head(5)

print("\nTop 5 Jobs Least Likely to Be Replaced by AI:")
display(low_risk_jobs)

# Plot
plt.figure(figsize=(10,5))
plt.barh(low_risk_jobs[title_col], low_risk_jobs[risk_col], color='green')
plt.title("Top 5 Jobs Least Likely to Be Replaced by AI")
plt.xlabel("AI Risk Score")
plt.ylabel("Job Title")
plt.gca().invert_yaxis()
plt.grid(axis='x')
plt.show()


# ### 3) Top 50 jobs most replaceable by AI

# In[84]:


# Top 50 Jobs Most Replaceable by AI

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load enriched job data
jobs = pd.read_csv("ai_job_risk_detailed.csv")

# Identify key columns
risk_col = "AI_Risk_Score" if "AI_Risk_Score" in jobs.columns else jobs.columns[1]
title_col = jobs.columns[0]

# Sort by AI risk (descending) and take top 50
top_50_jobs = jobs.sort_values(by=risk_col, ascending=False).head(50)

print("\nTop 50 Jobs Most at Risk of AI Replacement:")
display(top_50_jobs)

# Optional Plot: Horizontal bar chart for top 20 (for clarity)
plt.figure(figsize=(12, 8))
plt.barh(top_50_jobs.head(20)[title_col], top_50_jobs.head(20)[risk_col], color='darkred')
plt.title("Top 20 Jobs Most at Risk of AI Replacement")
plt.xlabel("AI Risk Score")
plt.ylabel("Job Title")
plt.gca().invert_yaxis()
plt.grid(axis='x')
plt.show()


# #### 4) Industries that are most impacted by AI

# In[87]:


# Pie Chart: Normalized AI Impact by Industry

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Normalized_Average_AI_Impact_by_Domain.csv")

# Auto-detect columns
domain_col = [c for c in df.columns if "domain" in c.lower() or "industry" in c.lower()][0]
impact_col = [c for c in df.columns if "%" in c or "impact" in c.lower()][0]

print("Using columns:")
print("Industry/Domain:", domain_col)
print("Impact column:", impact_col)
print("\nPreview:")
display(df.head())

# Sort values (optional)
df_sorted = df.sort_values(by=impact_col, ascending=False)

# Plot
plt.figure(figsize=(10, 10))
plt.pie(
    df_sorted[impact_col],
    labels=df_sorted[domain_col],
    autopct="%1.1f%%",
    startangle=140,
    pctdistance=0.8
)

plt.title("Normalized Average AI Impact by Industry Domain (Sum = 100%)", fontsize=14)
plt.axis('equal')
plt.show()


# #### This visualization highlights that AI risk is not evenly distributed across industries. Administrative & Clerical roles face the highest exposure (13%), followed closely by Data & IT (12.7%) and Sales & Marketing (11.6%).
# 
# #### These domains rely heavily on text generation, content categorization, repetitive decision-making, and data summarization, areas where generative AI like ChatGPT provides the strongest automation leverage.

# # V. Socioeconomic Context: Who Is Most Vulnerable? 🕵️

# In[97]:


# Part V – PolicyMap Socioeconomic Analysis
# Using: county_features.csv
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load county features
county = pd.read_csv("county_features.csv")
print("county_features shape:", county.shape)
print("Columns:", county.columns.tolist())

# 2. Keep & clean the core columns
socio = county[[
    "County_Name",
    "State",
    "Bachelors_Degree_Rate",
    "Median_Income",
    "Unemployment_Rate"
]].copy()

# Ensure numeric (in case of weird strings like 'pbach')
socio["Bachelors_Degree_Rate"] = pd.to_numeric(socio["Bachelors_Degree_Rate"], errors="coerce")
socio["Median_Income"]        = pd.to_numeric(socio["Median_Income"],        errors="coerce")
socio["Unemployment_Rate"]    = pd.to_numeric(socio["Unemployment_Rate"],    errors="coerce")

# Drop rows with missing core metrics
socio = socio.dropna(subset=["Bachelors_Degree_Rate", "Median_Income", "Unemployment_Rate"])

print("\nSocioeconomic subset preview:")
display(socio.head())

# 3. Summary statistics (raw numeric)
print("\nSummary statistics (after cleaning) – numeric:")
summary_numeric = socio[[
    "Bachelors_Degree_Rate",
    "Median_Income",
    "Unemployment_Rate"
]].describe()
display(summary_numeric)

# 4. Nicely formatted summary table
summary = summary_numeric.T  # rows = variables

def format_row(row):
    # Median_Income as dollars with commas
    if row.name == "Median_Income":
        return row.apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
    # Degree & unemployment as percentages with 1 decimal
    else:
        return row.apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")

summary_formatted = summary.apply(format_row, axis=1)

print("\nSummary statistics (after cleaning) – formatted:")
display(summary_formatted)

# 5. Correlation matrix + heatmap
corr = socio[[
    "Bachelors_Degree_Rate",
    "Median_Income",
    "Unemployment_Rate"
]].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    vmin=-1, vmax=1,
    square=True,
    fmt=".2f"
)
plt.title("Correlation Between Socioeconomic Indicators")
plt.tight_layout()
plt.show()

# 6. Vulnerability Index
#    (low education + low income + high unemployment)

# Standardize each metric (z-scores)
deg_z   = (socio["Bachelors_Degree_Rate"] - socio["Bachelors_Degree_Rate"].mean()) / socio["Bachelors_Degree_Rate"].std()
inc_z   = (socio["Median_Income"]        - socio["Median_Income"].mean())        / socio["Median_Income"].std()
unemp_z = (socio["Unemployment_Rate"]    - socio["Unemployment_Rate"].mean())    / socio["Unemployment_Rate"].std()

# Low education & income increase vulnerability, high unemployment increases vulnerability
socio["Vulnerability_Index"] = (-deg_z) + (-inc_z) + (unemp_z)

# 7. Top 10 most vulnerable counties (with nice formatting)
top10 = socio.sort_values("Vulnerability_Index", ascending=False).head(10).copy()

# Pretty display columns
top10_display = top10[[
    "County_Name",
    "State",
    "Bachelors_Degree_Rate",
    "Median_Income",
    "Unemployment_Rate",
    "Vulnerability_Index"
]].copy()

top10_display["Bachelors_Degree_Rate"] = top10_display["Bachelors_Degree_Rate"].map(lambda x: f"{x:.2f}%")
top10_display["Median_Income"]        = top10_display["Median_Income"].map(lambda x: f"${x:,.0f}")
top10_display["Unemployment_Rate"]    = top10_display["Unemployment_Rate"].map(lambda x: f"{x:.2f}%")
top10_display["Vulnerability_Index"]  = top10_display["Vulnerability_Index"].map(lambda x: f"{x:.2f}")

print("\nTop 10 Most Vulnerable Counties (combined low education, low income, high unemployment):")
display(top10_display)

# 8. Save cleaned & scored dataset for your report
socio.to_csv("ai_socioeconomic_vulnerability.csv", index=False)
print("\nSaved dataset: 'ai_socioeconomic_vulnerability.csv'")


# ### This analysis shows that areas with less education, lower income, and higher unemployment are most vulnerable to the impact of AI. Using county data from PolicyMap, we found strong links between higher education and higher income, as well as lower unemployment. The most at-risk counties, mainly in Puerto Rico and parts of the rural South have very low bachelor’s degree rates, low incomes, and higher unemployment. These regions may face more challenges adapting to AI-driven changes in the job market.

# # VI. Conclusion 📖

# ### This project was designed to investigate whether AI automation has already begun reshaping our world and how it’s impacting everything from jobs to our personal lives. By analyzing ChatGPT’s explosive user growth, it’s clear that AI is becoming a mainstream tool across generations and global markets. Survey data reveals that people are turning to AI for tasks ranging from problem-solving to creative expression signaling a shift in how we work, learn, and communicate. At the same time, the Job Threat Index highlights a more sobering reality: many industries are increasingly vulnerable to automation, especially in transportation, administration, and manufacturing. What’s more concerning is that these risks are not distributed equally. Counties with low educational attainment, lower income levels, and higher unemployment are most exposed. This suggests that AI’s benefits and burdens may amplify existing inequalities. Ultimately, the findings emphasize that while AI opens exciting new possibilities, it’s critical to prepare communities and workers through education, policy support, and re-skilling initiatives to ensure a more equitable transition into the future of work.
