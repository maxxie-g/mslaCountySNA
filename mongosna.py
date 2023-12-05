import json
import string

# Plotting
import matplotlib.pyplot as plt
from numpy import double
import pandas as pd
from wordcloud import WordCloud

# Utilities
import googlemaps
import pymongo

# NLP
import re
# import nltk.corpus
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Analysis
import math
from scipy.stats import norm

# API Clients
gmaps_client = None
db_client = None
tweepy_client = None

# Collections and DataFrames
pFollower = None
sFollower = None
sFollowing = None
combined = None
sna_database = None
df_pFollower = None
df_sFollower = None
df_sFollowing = None

# Method for cleaning up text, making it easier for NLP
def nlp_preprocessing(group,dataset,stemOutputPath,lemmaOutputPath):
    # get the text from group where the description is not empty
    
    list_temp = list(dataset['description'])
    stop = stopwords.words('english')

    for i in range(len(list_temp)):
        # normalize text
        list_temp[i] = list_temp[i].lower()
            
        # remove unicode characters    
        list_temp[i] = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?" + "\\n", " ", list_temp[i])

        # remove stopwords
        list_temp[i] = " ".join([word for word in list_temp[i].split() if word not in (stop)])
    
    # we need to iterate back through the list and remove any empty strings
    text = [item for item in list_temp if item != "" if item != "n/a"]
    
    # stemming
    text_stem = text
    stemmer = PorterStemmer()

    for s in range(len(text_stem)):
        text_stem[s] = stemmer.stem(text_stem[s])

    # lemmatization
    text_lemma = text
    lemmatizer = WordNetLemmatizer()

    for l in range(len(text_lemma)):
        text_lemma[l] = lemmatizer.lemmatize(text_lemma[l])
    
    # output the stemmed and lemmatized texts to csv for storage
    stem_series = pd.Series(text_stem)
    stem_series.to_csv(stemOutputPath,index=False)

    lemma_series = pd.Series(text_lemma)
    lemma_series.to_csv(lemmaOutputPath,index=False)

    return print("Complete Group " + str(group))

# Method for creating the wordcloud from the results of the nlp_preprocessing()
def wordcloud_vis(lemmaInput,stemInput,lemmaOutput,stemOutput):
    stop = stopwords.words('english')
    stop.extend(['missoula','montana','montanan','mt'])

    text_lemma = " ".join(lemmaInput)
    wordcloud = WordCloud(stopwords=stop,width=1600,height=1600).generate(text_lemma)

    plt.axis("off")
    wordcloud.to_file(lemmaOutput)

    text_stem = " ".join(stemInput)
    wordcloud = WordCloud(stopwords=stop,width=1600,height=1600).generate(text_stem)

    plt.axis("off")
    wordcloud.to_file(stemOutput)

# Statistical analysis of a given DataFrame
def stats(df: pd.DataFrame,output_path: string) -> None:
    """
    param df: Pandas DataFrame containing information on a set of Twitter users
    type df: pandas.DataFrame
    param output_path: string containing the path to output the file
    type output_path: string
    returns: None
    rtype: None
    """
    series_drop = ['_id','verified']
    quantiles = [0.1,0.25,0.5,0.75,0.9]
    output = {
        'followers': {},
        'following': {},
        'tweet_count': {}
    }

    mean = df.mean(numeric_only=True)
    mean.drop(series_drop,inplace=True)

    median = df.median(numeric_only=True)
    median.drop(series_drop,inplace=True)

    std = df.std(numeric_only=True)
    std.drop(series_drop,inplace=True)

    var = df.var(numeric_only=True)
    var.drop(series_drop,inplace=True)

    min = df.min(numeric_only=True)
    min.drop(series_drop,inplace=True)

    max = df.max(numeric_only=True)
    max.drop(series_drop,inplace=True)

    sum = df.sum(numeric_only=True)
    sum.drop(series_drop,inplace=True)

    quantileFollowers = df['followers'].dropna().quantile(q=quantiles)
    quantileFollowing = df['following'].dropna().quantile(q=quantiles)
    quantileTweetCount = df['tweet_count'].dropna().quantile(q=quantiles)

    output['followers'].update({'mean': math.floor(mean['followers'])})
    output['followers'].update({'mode': math.floor(df['followers'].mode())})
    output['followers'].update({'median': math.floor(median['followers'])})
    output['followers'].update({'min': math.floor(min['followers'])})
    output['followers'].update({'max': math.floor(max['followers'])})
    output['followers'].update({'std': math.floor(std['followers'])})
    output['followers'].update({'var': math.floor(var['followers'])})
    output['followers'].update({'sum': math.floor(sum['followers'])})
    output['followers'].update({'0.1': math.floor(quantileFollowers[0.10])})
    output['followers'].update({'0.25': math.floor(quantileFollowers[0.25])})
    output['followers'].update({'0.5': math.floor(quantileFollowers[0.5])})
    output['followers'].update({'0.75': math.floor(quantileFollowers[0.75])})
    output['followers'].update({'0.9': math.floor(quantileFollowers[0.9])})

    output['following'].update({'mean': math.floor(mean['following'])})
    output['following'].update({'mode': math.floor(df['following'].mode())})
    output['following'].update({'median': math.floor(median['following'])})
    output['following'].update({'min': math.floor(min['following'])})
    output['following'].update({'max': math.floor(max['following'])})
    output['following'].update({'std': math.floor(std['following'])})
    output['following'].update({'var': math.floor(var['following'])})
    output['following'].update({'sum': math.floor(sum['following'])})
    output['following'].update({'0.1': math.floor(quantileFollowing[0.10])})
    output['following'].update({'0.25': math.floor(quantileFollowing[0.25])})
    output['following'].update({'0.5': math.floor(quantileFollowing[0.5])})
    output['following'].update({'0.75': math.floor(quantileFollowing[0.75])})
    output['following'].update({'0.9': math.floor(quantileFollowing[0.9])})

    output['tweet_count'].update({'mean': math.floor(mean['tweet_count'])})
    output['tweet_count'].update({'mode': math.floor(df['tweet_count'].mode())})
    output['tweet_count'].update({'median': math.floor(median['tweet_count'])})
    output['tweet_count'].update({'min': math.floor(min['tweet_count'])})
    output['tweet_count'].update({'max': math.floor(max['tweet_count'])})
    output['tweet_count'].update({'std': math.floor(std['tweet_count'])})
    output['tweet_count'].update({'var': math.floor(var['tweet_count'])})
    output['tweet_count'].update({'sum': math.floor(sum['tweet_count'])})
    output['tweet_count'].update({'0.1': math.floor(quantileTweetCount[0.10])})
    output['tweet_count'].update({'0.25': math.floor(quantileTweetCount[0.25])})
    output['tweet_count'].update({'0.5': math.floor(quantileTweetCount[0.5])})
    output['tweet_count'].update({'0.75': math.floor(quantileTweetCount[0.75])})
    output['tweet_count'].update({'0.9': math.floor(quantileTweetCount[0.9])})

    with open(output_path,'w') as fp:
        json.dump(output,fp,indent=2)

# Next two methods obtain coordinates from location information via googlemaps geocoding api
def geocode_lat(add: string) -> float:
    """
    param gmaps_client: GoogleMaps API Client
    type gmaps_client: GoogleMaps.Client
    param add: the location string to geocode
    type add: string
    returns: lat object from the GoogleMaps response
    rtype: float
    """
    g = gmaps_client.geocode(add)
    lat = None
    try:
        lat = g[0]["geometry"]["location"]["lat"]
    except:
        pass
    return lat

def geocode_lng(add: string) -> float:
    """
    param gmaps_client: GoogleMaps API Client
    type gmaps_client: GoogleMaps.Client
    param add: the location string to geocode
    type add: string
    returns: lng object from the GoogleMaps response
    rtype: float
    """
    g = gmaps_client.geocode(add)
    lng = None
    try:
        lng = g[0]["geometry"]["location"]["lng"]
    except:
        pass
    return lng

# add coordinates to the database
def add_geo(lat: list,lng: list,geo_ids: list,user_ids: list,collection: pymongo.MongoClient) -> None:
    """
    param lat: list of latitude values
    type lat: list
    param lng: list of longitude values
    type lng: list
    param geo_ids: list of user IDs with geocoded location data
    type geo_ids: list
    param user_ids: list of all user IDs in the Database Collection
    type user_ids: list
    param collection: PyMongo Client to add the data into
    type collection: pymongo.MongoClient
    returns: None
    rtype: None
    """
    iter = 0

    for id in user_ids:
        if id in geo_ids:
            query = { '_id' : id }
            value = { '$set' : { 'lat' : lat[iter], 'lng': lng[iter]}}
            iter += 1
            collection.update_one(query,value)
        else:
            query = { '_id' : id }
            value = { '$set' : { 'lat' : None, 'lng': None}}
            collection.update_one(query,value)

# swap columns of a Pandas DataFame
def swap_columns(df: pd.DataFrame,col1: int,col2: int) -> pd.DataFrame:
    """
    param df: pandas DataFrame to swap columns in
    type df: pandas.DataFrame
    param col1: index value of the first column
    type col1: int
    param col2: index value of the second column
    type col2: int
    returns: df
    rtype: pandas.DataFrame
    """
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df
    
# create histogram from given list of data
def makeStatsGraph(data_list: list, xlabel: string, title: string, output_path: string):
    num_bins = 100

    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(data_list, num_bins,density=True,)

    mu, sigma = norm.fit(data_list)
    y = norm.pdf(bins,mu,sigma)
    l = plt.plot(bins, y, 'r--')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title % (mu,sigma))
    plt.savefig(output_path,dpi=250,bbox_inches='tight')