# Beginners' introduction to Pandas - Pycon APAC 2018
## Recommended installation
The Anaconda distribution package is one of the most popular packages for data science and contains all of the packages you will need for this workshop and for further experimentation and usage.  
The distribution package can be downloaded here (600~mb): `https://www.anaconda.com/download/`  
(You can also get the distribution package from me as the file is rather big)
## Advanced installation
For advanced users who wish to install only the necessary modules, these are the modules needed for this workshop.  
* Python 3.6
* Jupyter notebook `pip3 install jupyter`
* Pandas >= 0.20.3 `pip3 install pandas`
* Numpy >= 1.13.3 `pip3 install numpy`
## What is Pandas
[**Pandas**](https://pandas.pydata.org/) is a Python package that provides fast, expressive, data structures that makes working with relational data intuitive. It can be used along with other packages such as [Numpy](http://www.numpy.org/) and [scikit-learn](http://scikit-learn.org/stable/) to further extend the uses of the package.


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
s2 = pd.Series([0.5, 1.0, 1.5, 2.0], index=['a', 'd', 'b', 'c'])

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
s3[['a', 'c']] # Returns Series with only 'a' and 'c'

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

For the first part you will learn to merge datasets and make some basic manipulations and calculations.
**read_csv(with proper labels)**  
With a properly defined csv file, importing csv files is as simple as calling `read_csv(<filename>)`  
You can also output csv files from `DataFrame`s by simply calling `DataFrame.to_csv(<filename>)`
```
df1 = pd.read_csv('data/users_with_age_data.csv')
df2 = pd.read_csv('data/users_with_gender_data.csv')
```

**Merging DataFrames**  
Having multiple different `DataFrame`s isn't very helpful for data comparison or calculations. Here, you will merge the `DataFrame`s together to start making sense of the datasets. 
```
# Left merge using the 'user_id' column as keys (this only retains rows which exist in df1)
df1.merge(df2, on='user_id', how='left')

# Outer merge using the 'user_id' column as keys (all rows are kept, empty values are filled with 'null's)
users = df1.merge(df2, on="user_id", how="outer")
```

**Manipulating DataFrames**  
You will sometimes encounter datasets with gaps in the data. In this case, some users do not have 'gender' or 'age' filled in.  
In certain cases you will choose to leave them as None. But here, you will fill them with default values using the `fillna()` function.
```
# Fills the empty values in 'gender' with 'F'
users['gender'] = users['gender'].fillna("'F'")

# Fills the empty values in 'age' with the available mean age
users['age'] = users['age'].fillna(users['age'].mean())
```

**Grouping and calculations**  
A common thing to do when evaluating datasets is to group the data by certain criteria and making calculations based off that. This is possible by using the `groupby()` function to group the data by certain columns and then evaluating the grouped data.
```
# Groups rows by 'gender' column
grouped = users.groupby('gender')

# Aggregates number of rows after grouping
grouped.count()

# Access grouped columns and apply mean function
grouped['age'].mean()
```

**Adding Columns**  
You will often run into cases where certain derived values are used multiple times. It is easy to add new columns to `DataFrame`s.
```
# Adds a new column based on function applied to another column
users['minor'] = users['age'].apply(lambda x: True if x < 18 else False)
```

## Working with datasets (Pt. 2)
Now that you know how to work with `DataFrame`s let's move to a bigger dataset.  

For this example, you will work on the given dataset of movie ratings to determine a few things:
* What are the favorite movies of males vs females
* What are the movies with the most discrepancy between reviews

**read_csv(without proper labels)**  
The given datasets do not come with labels. In such cases, you should determine the labels through the source if possible, or make assumptions if not.  
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
As the `merge()` function returns a `DataFrame` you are able to chain merges together as long as you can merge on a key.
```
# joins the all the dataframes
merged = users.merge(ratings, on="user_id").merge(movies, on="movie_id")
```

**Sort by number of ratings**  
There are times where the amount of data is too low to be meaningful. In such cases, it is better to set a threshold and remove data that you would deem inconsequential.  
Here, you will group the movie ratings together by title and make some judgements on the threshold.
```
# Group ratings by title and aggregate the values, then sort the values in descending order
sorted_movies = merged.groupby("title").count().sort_values(by='rating', ascending=False)

# Gets top 5 rated movies
sorted_movies.head(5)
```

**Applying query to DataFrame**  
We will deem movies with less than 250 ratings to be inactive and would therefore not give good results.  
You can easily filter `DataFrame`s with the `query()` function or make use of the [] filter functionality of `Series` to filter out offending data.
```
# list moviess with >= 250 ratings
active_titles = sorted_movies.query('rating >= 250')
# active_titles = top_movies[top_movies['rating'] >= 250]
```

**Ungrouping DataFrames**  
Grouped `DataFrame`s are useful for determining values from groups but not so for evaluating other information.  
It is useful to ungroup the data again for further evaluation.  
(You can also ungroup with the `reset_index()` function, but it will not be covered here.
```
# Filter original DataFrame with index of filtered DataFrame 
ungrouped_sorted_movies = merged[merged['title'].isin(active_titles.index)]
```

**Grouping by multiple columns and calculating mean values**  
To calculate the mean of the ratings separated by gender and title, we will group the `DataFrame` by more than one columns.  
Once the `DataFrame` is grouped, you can call `agg()` to make aggregated values of the columns.
```
# Groups ratings by 'gender' and title'
separated_rating = ungrouped_sorted_movies.groupby(['gender','title'], as_index=False)

# Aggregates mean value of 'rating' using np.mean()
separated_rating = separated_rating.agg({'rating': np.mean})
```

**Getting sorted, grouped data**  
To get the favorite movies of both genders, we can either filter the genders and get the top movies of the filtered `DataFrame`s, or we can make use of a feature of grouped `DataFrame`s to get the top movies of each gender.
```
# If you do not group the data, it will only return the 1st 3 items
separated_rating.sort_values(['gender','rating'],ascending=False).head(3)

# Grouped DataFrames will give result of head split by groupings
separated_rating.sort_values(['gender','rating'],ascending=False).groupby('gender').head(3)
```

**Pivoting tables**  
Sometimes, you would need to use the rows of a `DataFrame` as columns to calculate some values.  
In order to calculate the difference in rating between genders of a movie, you would need to be able to access the 'M' and 'F' rows of data as columns.  
The `pivot()` functions allows you to use all the values of a column as the columns of a new pivoted `DataFrame`.
```
# Pivots ratings table to use values of 'title' as index and values of 'gender' as columns
pivoted_ratings = separated_rating.pivot('title', 'gender')
```

**Applying function to all rows of a DataFrame**  
`apply()` allows you to apply a function to each row or column of a `DataFrame`.  
Whether the function is applied to the row or column of the `DataFrame` is determined by the `axis` parameter. (0 for col, 1 for row)
```
# Applies 'F' - 'M' to each row and sorts the result
pivoted_ratings['rating'].apply(lambda x: x['F'] - x['M'], axis=1).sort_values()
```

**Calculating standard deviation**  
Standard deviation is a common value to look for to see how controversial a thing is.  
This higher the deviation, the more controversial it is.  
`std()` is a helper aggregator functions available in Pandas that calculates and returns the standard deviation value of a `Series`
```
# Regroups movies and calculates the standard deviation of each title
ratings_std = ungrouped_sorted_movies.groupby('title')['rating'].std()

# Using nlargest instead of sorting the values and calling head(n) to get top n values
ratings_std.nlargest(5)
```

# End
This marks the end of the workshop.  
I hope this gave you a good beginner's tour of Pandas and manipulation of `DataFrame`s to arrive at meaningful values.  

Feel free to reach out to me at [pengyu@theartling.com](mailto:pengyu@theartling.com)
