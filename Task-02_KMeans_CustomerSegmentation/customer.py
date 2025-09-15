# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")
        
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df =  pd.read_csv("../data/Mall_Customers.csv")

def check_missing(df):
    """Return the number of missing values per column."""
    return df.isnull().sum()

def basic_info(df):
    """Print basic info about the DataFrame."""
    return df.info()

def statistical_summary(df):
    """Return transposed statistical summary."""
    return df.describe().T

def check_duplicated(df , drop = False):
    num_duplicates = df.duplicated().sum()

    if drop : 
        df_cleaned = df.drop_duplicates()
        print(f"Removed {num_duplicates} duplicated rows.")
        return df_cleaned
    else:
        print(f"Found {num_duplicates} duplicated rows.")
        return num_duplicates


def numerical_categorical_columns(df):
    """Return numerical and categorical column lists."""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    return numeric_cols, categorical_cols

def summary_dataframe(df):
    """Return summary of data types, unique values, and null ratios."""
    d_types = df.dtypes
    n_uniq = df.nunique()
    ratio = df.isnull().sum() / len(df)
    return pd.DataFrame({
        'DTypes': d_types,
        'N_Uniq': n_uniq,
        'Null_Ratio': ratio
    })

def overview(df):

    print("----- BASIC INFO -----")
    basic_info(df)

    print("\n----- MISSING VALUES -----")
    print(check_missing(df))

    print("\n----- DUPLICATED ROWS -----")
    print(f"Duplicated Rows: {check_duplicated(df , drop = True)}")

    print("\n----- NUMERIC & CATEGORICAL COLUMNS -----")
    num_cols, cat_cols = numerical_categorical_columns(df)
    print(f"Numerical Columns: {num_cols}")
    print(f"Categorical Columns: {cat_cols}")

    print("\n----- STATISTICAL SUMMARY -----")
    print(statistical_summary(df))

    print("\n----- DATAFRAME SUMMARY -----")
    print(summary_dataframe(df))

# overview(df)

print(df.rename(columns = {'Spending Score (1-100)' : 'Spending Score'   , 'Annual Income (k$)' : 'Annual Income' } , inplace = True))
print(df)

def handle_outliers(df):

    df_cleaned = df.copy()
    numeric_cols = df_cleaned.select_dtypes(include='number').columns.tolist()

    for col in numeric_cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_cleaned[col] = np.where(
            df_cleaned[col] < lower_bound, lower_bound,
            np.where(df_cleaned[col] > upper_bound, upper_bound, df_cleaned[col])
        )

    return df_cleaned

df_new = handle_outliers(df)
df_cleaned = df_new.drop(['CustomerID','Gender','Age' ] , axis = 1 )

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_cleaned)
df_cleaned.plot(kind='box', figsize=(10, 5), title='Before Scaling')
plt.show()

scaled_df = pd.DataFrame(scaled_data, columns=df_cleaned.columns)
scaled_df.plot(kind='box', figsize=(10, 5), title='After Robust Scaling')
plt.show()

inertia = [] 
for i in range (1 ,11):
    kmeans = KMeans (n_clusters = i , random_state = 42)
    kmeans.fit(scaled_df)
    inertia.append(kmeans.inertia_)


kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_data)


df_cleaned['Cluster'] = clusters

plt.figure(figsize=(8, 6))

# Plot using two original (or scaled) features
plt.scatter(
    df_cleaned['Annual Income'],
    df_cleaned['Spending Score'],
    c=df_cleaned['Cluster'],
    cmap='Set2',
    s=60
)

plt.title("Customer Segments (No PCA)")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.grid(True)