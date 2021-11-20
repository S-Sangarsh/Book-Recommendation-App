import streamlit as st 
import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors

# reading the the csv file
df = pd.read_csv('RecommendationEngine.csv')
df1 = df.copy()
publisher= list(df['publisher'].unique())
author = list(df['authors'].unique())
language = list(df['language_code'].unique())
book = list(df['title'].unique())


def num_into_obj(x):
    if x>=0 and x<=1:
        return 'between 0 and 1'
    elif x>1 and x<=2:
        return 'between 1 and 2'
    elif x>2 and x<=3:
        return 'between 2 and 3'
    elif x>3 and x<=4:
        return 'between 3 and 4'
    else:
        return 'between 4 and 5'
    
df['rating_obj'] = df['average_rating'].apply(num_into_obj)
rating_df = pd.get_dummies(df['rating_obj'])
language_df = pd.get_dummies(df['language_code'])


# Let's concat both the data frames and set the title column as the index 
features = pd.concat([rating_df, language_df, df['average_rating'], df['ratings_count'],df['title']], axis=1)
features.set_index('title', inplace=True)

# scaling down the values of the data frame
min_max_scaler = MinMaxScaler()
features_scaled = min_max_scaler.fit_transform(features)


# training the model
model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree', metric='euclidean')
model.fit(features_scaled)
dist, idlist = model.kneighbors(features_scaled)


# based on the book you have read
def BookRecommender(x):
    book_list_name = []
    book_id = df[df['title'] == x].index
    book_id = book_id[0]
    for newid in idlist[book_id]:
        book_list_name.append(df.loc[newid].title)
    return book_list_name[1:]


# recommending books based on authors
def recommend_books_on_publishers(x):
    a = df[df['publisher']==x][['title','average_rating']]
    a = a.sort_values(by = 'average_rating', ascending=False)
    return a.head(10)

# recommending books based on authors
def recommend_books_on_authors(x):
    a = df[df['authors']==x][['title','average_rating']]
    a = a.sort_values(by = 'average_rating', ascending=False)
    return a.head(10)

# recommend books based on languages
def recommend_books_on_languages(x):
    a = df[df['language_code']==x][['title','average_rating']]
    a = a.sort_values(by = 'average_rating', ascending=False)
    return a.head(10)




#Book Recommendation System

# Title
st.title("Bookkeeda")
st.header("One Stop Solution to all the Book Recommendation...")
st.subheader("Lets have look at top books based some attribute....This will give generally idea before using recommendation system here")
st.write('')
st.write('')
st.write('')
st.write('')
# top 10 best book based on rating_count and average rating
st.subheader('Top 10 Book Based on Highest Ratings Count and Average Rating')

st.write(df1[['title','authors','average_rating','ratings_count']].sort_values(by = ['ratings_count'],ascending = False).head(10).style.background_gradient(cmap = 'coolwarm'),width=1200)



st.write('')
st.write('')
st.write('')
st.write('')

st.subheader('Visualise the top 10 authors with maximum number of books')
# Visualise the top 10 authors with maximum number of books
fig, ax = plt.subplots()
ax=sns.countplot(x = "authors", data = df1,order = df1['authors'].value_counts().iloc[:10].index, palette = "coolwarm")
plt.xticks(fontsize =8,rotation=75)
st.pyplot(fig)
st.write('')
st.write('')
st.write('')
st.write('')


# Most occuring books in the data

st.subheader('Most occuring books in the data')
fig, ax = plt.subplots()
book1 = df1['title'].value_counts()[:15]
ax = sns.barplot(y=book1, x = book1.index, palette = 'winter_r') 

plt.xlabel("Number of occurences")
plt.ylabel("Books")
plt.xticks(rotation = 75,fontsize = 8)
st.pyplot(fig)


st.write('')
st.write('')
st.write('')
st.write('')

# distribution plot for average rating column
st.subheader('Distribution plot for average rating column')
fig, ax = plt.subplots()
ax = sns.distplot(df1['average_rating'])
st.pyplot(fig)

# Top 10  Book Suggested based on Publication wrt their average rating
st.write('')
st.write('')
st.write('')
st.write('')
# Top publishers with maximum books
st.subheader('Top publishers with maximum books')
fig, ax = plt.subplots()
publisher1 = df1['publisher'].value_counts()[:15]
sns.barplot(y=publisher1, x = publisher1.index, palette = 'Wistia')
plt.xlabel("Number of occurences")
plt.ylabel("Publishers")
plt.xticks(rotation = 75)
st.pyplot(fig)




st.write('')
st.write('')
st.write('')
st.write('')
st.title('Recommendation Engine')

if st.checkbox("Book Recommended Based on Publisher"):
    st.subheader("Select the Publisher Name")
    x = st.selectbox('Publisher Name',publisher,index=0)
    if st.button("Suggest"):
        result = recommend_books_on_publishers(x)
        st.write(f'Result of Top 10 Book by {x} Publications')
        st.write(result,height=1100)
        
# Top 10  Book Suggested based on Author wrt their average rating
if st.checkbox("Book Recommended Based on Author"):
    st.subheader("Select the Author Name")
    x = st.selectbox('Author Name',author,index=1)
    if st.button("Suggest"):
        result = recommend_books_on_authors(x)
        st.write(f'Result of Top 10 Book by {x}')
        st.dataframe(result,width=1100,height=1100)
        
# Top 10  Book Suggested based on Author wrt their average rating
if st.checkbox("Book Recommended Based on Language"):
    st.subheader("Select the Language Code")
    x = st.selectbox('Language Code',language,index=1)
    if st.button("Suggest"):
        result = recommend_books_on_languages(x)
        st.write(f'Result of Top 10 Book based on the language code {x}')
        st.dataframe(result,width=1100,height=1100)
	

# Top 10  Book Suggested based on book you have read
if st.checkbox("Book Recommended Based on Book Read  Before"):
    st.subheader("Select the Book Name that you have read Before")
    x = st.selectbox('Book Name',book,index=1)
    if st.button("Recommend"):
        result = BookRecommender(x)
        st.write(f'Result of Top Book based on before read book by you is {x}')
        st.dataframe(result,width=1100,height=1100)

st.sidebar.subheader("Book Recommendation App")
st.sidebar.info("Beta Phase(some functionality might have some issue)")
    

st.sidebar.subheader("By")
st.sidebar.text("Sangarsh")

