## Beginners' introduction to Pandas - Pycon APAC 2018
# Recommended installation
The Anaconda distribution package is one of the most popular packages for data science and contains all of the packages you will need for this workshop and for further experimentation and usage.  
The distribution package can be downloaded here (600~mb): `https://www.anaconda.com/download/`  
(You can also get the distribution package from me as the file is rather big)
# Advanced installation
For advanced users who wish to install only the necessary modules, these are the modules needed for this workshop.  
* Python 3.6
* Jupyter notebook `pip3 install jupyter`
* Pandas >= 0.20.3 `pip3 install pandas`
* Numpy >= 1.13.3 `pip3 install numpy`


## Setup
Start up jupyter notebook in a clean folder:  
`$ jupyter notebook`

Create a new 'Python 3' notebook:  
![jupyter notebook instruction](https://i.imgur.com/FSV3X5V.png)

Import the libraries you will use for this workshop in your current notebook environment:  
```
import pandas as pd
import numpy as np
```
(Tip: You can press `shift + enter` to execute the cell)

# Let's begin

## Basic Pandas Data-structures(Series, DataFrame)
## [Series](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html)
A `Series` is a **one-demensional ndarray with axis labels**  
Values within a `Series` object can be accessed by index: `a[0]`  
You can pass in `index` to create your own index  

Ways to create `Series` objects:
```
# Array
s1 = pd.Series([0.5, 1.0, 1.5, 2.0])

s1[0] # Returns 0.5

# Array and index
s2 = pd.Series([0.5, 1.0, 1.5, 2.0], index=['a', 'd', 'b', 'c')

s2['d'] # returns 1.0 (You can still access by numerical index: s2[1])

# Dictionary
s3 = pd.Series({'a': 15, 'd': 18, 'c': 20, 'b': 9})

s3 # Returns created series sorted by key
```

You can apply multiple different functions to `Series`
```
# Filtering
s3[s3 < 16] # Returns filtered Series with values of s3 less than 16

# Multiplication
s3 * 2 # Returns Series with all values multiplied by 2

# Apply
s3.apply(lambda x: True if x < 16 else False)

# Absolute
s3.abs() # Returns Series with all values transformed to absolute

# Check if index exists in Series
'a' in s3 # True
'e' in s3 # False

# Get values in Series
s3['a', 'c'] # Returns Series with only 'a' and 'c'

# Other examples:
s3.mean()
s3.std() # Standard deviation
s3.min()
s3.max()
# And more...
```

## [DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html#pandas.DataFrame)
A `DataFrame` is a **2-dimensional table with labeled axes**.  
Acts like a dict-like container for `Series` objects.  

Creating `DataFrame`:
```
 data = {
    "city": ["Paris", "London", "Berlin"],
    "density": [3550, 5100, 3750],
    "area": [2723, 1623, 984],
    "population": [9645000, 8278000, 3675000],
  }
  df = pd.DataFrame(data)
```

You get a `Series` if you access a `DataFrame`'s index
```
df['area']
df.area
# Returns Series object of the 'area' column
```

## Working with datasets (Pt. 1)
Place the folders `data` and `movie` in the same folder as your jupyter notebook, they contain the datasets needed for the workshop.
**read_csv(with proper labels)**  
```
df1 = pd.read_csv('data/users_with_age_data.csv')
df2 = pd.read_csv('data/users_with_gender_data.csv')
```

**Merging DataFrames**  
```
# Left merge using the 'user_id' column as keys (this only retains rows which exist in df1)
df1.merge(df2, on='user_id', how='left')

# Outer merge using the 'user_id' column as keys (all rows are kept, empty values are filled with 'null's)
users = df1.merge(df2, on="user_id", how="outer")
```

**Manipulating DataFrames**  
```
# Fills the empty values in 'gender' with 'F'
users['gender'] = users['gender'].fillna("'F'")

# Fills the empty values in 'age' with the available mean age
users['age'] = users['age'].fillna(users['age'].mean())
```

**Grouping and calculations**  
```
# Groups rows by 'gender' column
grouped = users.groupby('gender')

# Aggregates number of rows after grouping
grouped.count()

# Access grouped columns and apply mean function
grouped['age'].mean()
```

**Adding Columns**  
```
# Adds a new column based on function applied to another column
users['minor'] = users['age'].apply(lambdax: True if x < 18 else False)
```

## Working with datasets (Pt. 2)
**read_csv(without proper labels)**  
users.dat: dataset containing info of users who added 1 or more ratings  
movies.dat: dataset containing name and id of movies  
ratings.dat: dataset containing all ratings given (linked by `user_id` and `movie_id`)  
```
# delimiter(character used to separate data), names(array of names to use as label order-specific)
users = pd.read_csv('movie/users.dat', delimiter='::', names=['user_id','gender','age','occupation_code','zip'], engine='python')
ratings = pd.read_csv('movie/ratings.dat', delimiter='::', names=['user_id','movie_id','rating','timestamp'], engine='python')
movies = pd.read_csv('movie/movies.dat', delimiter='::', names=['movie_id','title','genre'], engine='python')
```

**Merging DataFrames**  
```
# joins the all the dataframes
merged = users.merge(ratings, on="user_id").merge(movies, on="movie_id")
```

**Sort by number of ratings**  
```
# Group ratings by title and aggregate the values, then sort the values in descending order
sorted_movies = merged.groupby("title").count().sort_values(by='rating', ascending=False)

# Gets top 5 rated movies
sorted_movies.head(5)
```

**Applying query to DataFrame**  
```
# list moviess with >= 250 ratings
active_titles = sorted_movies.query('rating >= 250')
# active_titles = top_movies[top_movies['rating'] >= 250]
```

**Ungrouping DataFrames**  
```
# Filter original DataFrame with index of filtered DataFrame 
ungrouped_sorted_movies = merged[merged['title'].isin(active_titles.index)]
```

**Grouping by multiple columns calculating mean values**  
```
# Groups ratings by 'gender' and title'
separated_rating = ungrouped_sorted_movies.groupby(['gender','title'], as_index=False)

# Aggregates mean value of 'rating' using np.mean()
separated_rating = separated_rating.agg({'rating': np.mean})
```

**Getting sorted, grouped data**  
```
# If you do not group the data, it will only return the 1st 3 items
separated_rating.sort_values(['gender','rating'],ascending=False).head(3)

# Grouped DataFrames will give result of head split by groupings
separated_rating.sort_values(['gender','rating'],ascending=False).groupby('gender').head(3)
```

**Pivoting tables**  
```
# Pivots ratings table to use values of 'title' as index and values of 'gender' as columns
pivoted_ratings = separated_rating.pivot('title', 'gender')
```

**Applying function to all rows of a DataFrame**  
```
# Applies 'F' - 'M' to each row and sorts the result
pivoted_ratings['rating'].apply(lambda x: x['F'] - x['M'], axis=1).sort_values()
```

**Calculating standard deviation**  
```
# Regroups movies and calculates the standard deviation of each title
ratings_std = ungrouped_sorted_movies.groupby('title')['rating'].std()

# Using nlargest instead of sorting the values and calling head(n) to get top n values
ratings_std.nlargest(5)
```
