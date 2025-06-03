import pandas as pd
import numpy as np
import ast  

df = pd.read_csv("zomato.csv", encoding='latin-1') 

print("Shape of dataset:", df.shape)
print("\nColumn names:\n", df.columns)
print("\nFirst 5 rows:")
print(df.head())

#missing values
print("\nMissing values per column:\n", df.isnull().sum())

df = df.drop_duplicates(subset='url') #removing duplicates

print(df['rate'].unique())
print(df['votes'].unique())
df['rate'] = df['rate'].replace('-', '').fillna('') #replace with blanks
print("\nMissing values per column:\n", df.isnull().sum())

# NaN values in 'votes'
print("Missing values before replacement:", df['votes'].isnull().sum())

# Replace NaN values with blank 
df['votes'].replace(np.nan, '', inplace=True)
df['votes'] = df['votes'].replace('-', '').replace(np.nan, '')

# 'phone' that do NOT contain digits or '+'
print("Phone values without valid numbers:")
print(df[~df['phone'].str.contains('[0-9+]', na=False)]['phone'].unique())
df['phone'].replace(np.nan, '', regex=True, inplace=True)

# Replace 'Not Available' with blank
df['phone'].replace('Not Available', '', regex=True, inplace=True)

# if any invalid phone values remain
print("After cleaning, unique values in phone column that are still invalid:")
print(df[~df['phone'].str.contains('[0-9+]', na=False)]['phone'].unique())

print(df['location'].isnull().sum())
print(df['location'].unique())
df['rest_type'] = df['rest_type'].replace(np.nan, '', regex=True)
df['location'] = df['location'].replace(np.nan, '', regex=True)
print("Missing values in rest_type:", df['rest_type'].isnull().sum())
print("Missing values in location:", df['location'].isnull().sum())
df['cuisines'] = df['cuisines'].replace(np.nan, '', regex=True)
df['dish_liked'] = df['dish_liked'].replace(np.nan, '', regex=True)
print("Missing values in cuisines:", df['cuisines'].isnull().sum())
print("Missing values in dish_liked:", df['dish_liked'].isnull().sum())

# Check number of nulls and blanks
print("NULL values:", df['approx_cost(for two people)'].isnull().sum())
print("Blank values:", (df['approx_cost(for two people)'] == "").sum())

# if all non-null rows contain "two people"
print("Rows not containing 'two people':", df[~df['approx_cost(for two people)'].str.contains('two people', na=False)].shape)

#Use regex to extract just the numeric cost part
print("Nulls after regex extraction:")
print(df['approx_cost(for two people)'].str.extract(r'(\d{1,3}(?:,\d{3})*)').isnull().sum())

# View rows that had no extractable cost (i.e., original nulls)
print("Preview of rows with no cost info:")
print(df[df['approx_cost(for two people)'].str.extract(r'(\d{1,3}(?:,\d{3})*)').isnull().iloc[:, 0]].head())

# Extract the cost and rename column
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].str.extract(r'(\d{1,3}(?:,\d{3})*)')
print("NULL values:", df['reviews_list'].isnull().sum())
print("Blank (i.e., '[]') values:", (df['reviews_list'] == "[]").sum())

# Convert string representation of list of tuples into actual Python list
df['reviews_list'] = df['reviews_list'].apply(ast.literal_eval)
# checking type
print("Type check (first value):", type(df['reviews_list'].iloc[0]))
# total number of individual reviews across all restaurants
total_reviews = df['reviews_list'].apply(len).sum()
print("Total number of individual reviews:", total_reviews)

# NULL and blank values
print("NULL values : ", df['menu_item'].isnull().sum())
print("Blank : ", (df['menu_item'] == "[]").sum())

# Convert string representation of list into actual list objects
df['menu_item'] = df['menu_item'].apply(ast.literal_eval)

# View first few rows
df.head()
print("\nMissing values per column:\n", df.isnull().sum())

#############################

# Keep only +, digits, \r, \n, and spaces
df['phone'] = df['phone'].str.replace(r'[^\+0-9\r\n\s]', '', regex=True)

# if any invalid characters still exist
invalid_phones = df['phone'].str.contains(r'[^\+0-9\r\n\s]', na=False).sum()
print("Invalid Phone entries remaining:", invalid_phones)
# Remove non-English characters 
df['rest_type'] = df['rest_type'].str.replace(r'[^A-Za-z\s,]', '', regex=True).str.replace('Cafee', 'Cafe')
print("Unique values in Rest_type:", df['rest_type'].unique())
# Mapping restaurant types to readable names
type_of_meal = {
    'delivery_restaurants' : 'Delivery',
    'dineout_restaurants' : 'Dine-out',
    'drinks&nightlife' : 'Drinks & nightlife',
    'pubs&bars' : 'Pubs and bars',
    'buffet_restaurants' : 'Buffet',
    'cafe_restaurants' : 'Cafes',
    'desserts&bakes' : 'Desserts'
}

df['listed_in(type)'] = df['listed_in(type)'].map(type_of_meal)
print("Mapped Listed_in(type) values:", df['listed_in(type)'].unique())

print("Unique locations in Listed_in(city):", df['listed_in(city)'].unique())
print("Number of unique cities:", df['listed_in(city)'].nunique())

#####################
#Correcting data types
print(df.dtypes)
# Convert relevant columns to 'category'
df['online_order'] = df['online_order'].astype('category')
df['book_table'] = df['book_table'].astype('category')
df['location'] = df['location'].astype('category')
df['rest_type'] = df['rest_type'].astype('category')
df['listed_in(type)'] = df['listed_in(type)'].astype('category')
df['listed_in(city)'] = df['listed_in(city)'].astype('category')

# Fill missing values in 'Votes' column with 0
df['votes'] = df['votes'].fillna(0)
# Convert 'Votes' from float to int
df['votes'] = df['votes'].astype(int)

df.to_csv("zomato_cleaned.csv", index=False)
