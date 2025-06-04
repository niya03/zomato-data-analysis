import pandas as pd
import numpy as np
import ast
import re

# read dataset
df = pd.read_csv("../DATASETS/BangaloreZomatoData.csv", encoding='latin-1')
print(df.columns)

# make columns lowercase
df.columns = df.columns.str.lower()

# drop duplicates based on url
df.drop_duplicates(subset='url', inplace=True)

# clean dinner ratings column
df['dinner ratings'] = df['dinner ratings'].astype(str)
df['dinner ratings'] = df['dinner ratings'].apply(lambda x: x.split('/')[0] if '/' in x else x)
df['dinner ratings'] = df['dinner ratings'].replace(['NEW', '-', 'nan'], np.nan)
df['dinner ratings'] = pd.to_numeric(df['dinner ratings'], errors='coerce')

# clean delivery ratings column
df['delivery ratings'] = df['delivery ratings'].astype(str)
df['delivery ratings'] = df['delivery ratings'].apply(lambda x: x.split('/')[0] if '/' in x else x)
df['delivery ratings'] = df['delivery ratings'].replace(['NEW', '-', 'nan'], np.nan)
df['delivery ratings'] = pd.to_numeric(df['delivery ratings'], errors='coerce')

# clean cost column
df['averagecost'] = df['averagecost'].astype(str)
df['averagecost'] = df['averagecost'].str.replace(',', '')
df['averagecost'] = pd.to_numeric(df['averagecost'], errors='coerce')

# clean phone number
df['phonenumber'] = df['phonenumber'].astype(str)
df['phonenumber'] = df['phonenumber'].replace('nan', '', regex=False)
df['phonenumber'] = df['phonenumber'].replace('Not Available', '', regex=False)
df['phonenumber'] = df['phonenumber'].apply(lambda x: re.sub(r'[^\+0-9\s]', '', x))

# clean knownfor
df['knownfor'] = df['knownfor'].astype(str)
df['knownfor'] = df['knownfor'].replace('nan', '', regex=False)

# clean cuisines
df['cuisines'] = df['cuisines'].astype(str)
df['cuisines'] = df['cuisines'].replace('nan', '', regex=False)

# clean popular dishes
df['populardishes'] = df['populardishes'].astype(str)
df['populardishes'] = df['populardishes'].replace('nan', '', regex=False)

# clean people known for
df['peopleknownfor'] = df['peopleknownfor'].astype(str)
df['peopleknownfor'] = df['peopleknownfor'].replace('nan', '', regex=False)

# clean timing
df['timing'] = df['timing'].astype(str)
df['timing'] = df['timing'].replace('nan', '', regex=False)

# clean full address
df['full_address'] = df['full_address'].astype(str)
df['full_address'] = df['full_address'].replace('nan', '', regex=False)

# clean area
df['area'] = df['area'].astype(str)
df['area'] = df['area'].replace('nan', '', regex=False)

# convert boolean columns
bool_cols = ['ishomedelivery', 'istakeaway', 'isindoorseating', 'isvegonly']
for col in bool_cols:
    df[col] = df[col].astype(bool)

# convert category columns
cat_cols = ['area', 'cuisines', 'knownfor']
for col in cat_cols:
    df[col] = df[col].astype('category')

# save cleaned dataset
# df.to_csv("zomato_cleaned.csv", index=False)