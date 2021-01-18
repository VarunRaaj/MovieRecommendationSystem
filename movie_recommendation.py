#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Movie Recommendation System
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
columns_name=['user_id','item_id','rating','timestamp']
df=pd.read_csv('u.data',sep="\t",names=columns_name)
movies_title=pd.read_csv('u.item',sep="\|",header=None)
movies_titles=movies_title[[0,1]]
movies_titles.columns=["item_id","title"]
df=pd.merge(df,movies_titles,on="item_id")
ratings=pd.DataFrame(df.groupby('title').mean()['rating'])
ratings['num of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])
moviemat=df.pivot_table(index="user_id",columns="title",values="rating")
def predict_movies(movie_name):
    movie_user_ratings=moviemat[movie_name]
    similar_to_movie=moviemat.corrwith(movie_user_ratings)
    correlation_movie=pd.DataFrame(similar_to_movie,columns=['correlation'])
    correlation_movie.dropna(inplace=True)
    correlation_movie=correlation_movie.join(ratings['num of ratings'])
    
    predictions=correlation_movie[correlation_movie['num of ratings']>100].sort_values('correlation',ascending=False)
    
    return predictions

predict_my_movie=predict_movies("River Wild, The (1994)")
predict_my_movie.head()


# In[ ]:




