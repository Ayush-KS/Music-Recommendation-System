#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import warnings
import time
import streamlit as st


# In[2]:
warnings.filterwarnings("ignore")

# In[3]:

triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'

songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'

@st.cache
def load_data(fraction):
    songs_metadata = pd.read_csv(songs_metadata_file)
    triplets = pd.read_csv(triplets_file, sep="\t")
    triplets.columns = ["user_id", "song_id", "listen_count"]
    
    cnt = int(len(songs_metadata) * fraction)
    songs_metadata = songs_metadata[:cnt]
    triplets = triplets[:cnt]
    
    df = pd.merge(triplets, songs_metadata, on="song_id")
    df = df[df['listen_count'] >= 10]
    df["listen_count"] = (df["listen_count"]-df["listen_count"].min())/(df["listen_count"].max()-df["listen_count"].min())

    ratings = pd.DataFrame(df.groupby('title').mean()['listen_count'])
    ratings["number of ratings"] = df.groupby('title').count()['listen_count']

    songs_with_less_nor = ratings[ratings["number of ratings"] <= 10]
    print(len(songs_with_less_nor), len(ratings))
    to_remove = list(songs_with_less_nor.index)
    ratings = ratings[~ratings.index.isin(to_remove)]
    df = df[~df["title"].isin(to_remove)]
    # print(len(new_ratings), len(df), len(new_df))

    songs_mat = df.pivot_table(index="user_id", columns="title", values="listen_count")
    mp = df.set_index('title')["artist_name"].T.to_dict()
    return ratings, songs_mat, mp

# In[5]:


fraction = 1

# Create a text element and let the reader know the data is loading.
# data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
ratings, songs_mat, mp = load_data(fraction)
# Notify the reader that the data was successfully loaded.
# data_load_state.text("Done! (using st.cache)")




def recommend(song_title):
    song_ratings = songs_mat[song_title]
    similar = songs_mat.corrwith(song_ratings)
    corr_song = pd.DataFrame(similar, columns=["Correlation"])
    corr_song.dropna(inplace=True)
    corr_song = corr_song.join(ratings["number of ratings"])
    predictions = corr_song[corr_song["number of ratings"] > 100].sort_values(["Correlation", "number of ratings"], ascending=[False, False])
    print(len(predictions))
    predictions["Artist Name"] = pd.Series(mp, index = predictions.index)
    # predictions.drop('Correlation', axis=1, inplace=True)
    cols =  predictions.columns.tolist()
    predictions = predictions[cols[::-1]]
    return predictions



st.title('AstroRecommends')
option = st.sidebar.selectbox('Select a song that you like!', ratings.index)

'You selected:', option

output = recommend(option)
st.write(output)