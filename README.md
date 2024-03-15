import pandas as pd
import nltk
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
# Load the dataset
df = pd.read_csv('dataset_6.csv')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# tokenize the lyrics into words and then store as a list in a new column
def preprocess(text):
  tokens = nltk.word_tokenize(str(text))

#Removes the stop words and stemming
  
  stop_words = set(stopwords.words('english'))
  stemmer = PorterStemmer()
  filtered_tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]

  return filtered_tokens

df['processed_lyrics'] = df['lyrics'].apply(preprocess)
df['song_length'] = df['processed_lyrics'].apply(len)

# Plotting song length vs year
sns.relplot(x='year', y='song_length', data=df)

# Min-Max Scaling
scaler = MinMaxScaler()
df['song_length_scaled'] = scaler.fit_transform(df[['song_length']])

![image](https://github.com/BrianHodges4/MusicLyricsTrends-DataScience/assets/163497160/333f68a3-0227-4270-8267-909dfecbf5ea)


X = df[['year']]
y = df['song_length_scaled']

# Remove NaN values
X = X.dropna()
y = y.loc[X.index]

lm = LinearRegression().fit(X, y)

print("Coefficient for year (based on song length):", lm.coef_[0])

df['unique_words'] = df['lyrics'].apply(lambda x: len(set(nltk.word_tokenize(str(x)))))
sns.relplot(x='year', y='unique_words', data=df)

df['unique_words_scaled'] = scaler.fit_transform(df[['unique_words']])

![image](https://github.com/BrianHodges4/MusicLyricsTrends-DataScience/assets/163497160/e50c8d0f-4d6c-49c0-bb45-cbd214d17bc8)

X = df[['year']]
y = df['unique_words_scaled']

# Remove NaN values
X = X.dropna()
y = y.loc[X.index]

lm = LinearRegression().fit(X, y)

print("Coefficient for year (based on unique words):", lm.coef_[0])

def calculate_ratio(row):
    if row['processed_lyrics']:
        return len(set(row['processed_lyrics'])) / len(row['processed_lyrics'])
    return float('nan')

df['unique_to_total_ratio'] = df.apply(calculate_ratio, axis=1)

# Replaces NaN and infinity values with 0
df['unique_to_total_ratio'].replace([float('inf'), float('-inf')], float('nan'), inplace=True)
df['unique_to_total_ratio'].fillna(0, inplace=True)


sns.relplot(x='year', y='unique_to_total_ratio', data=df)

df['ratio_scaled'] = scaler.fit_transform(df[['unique_to_total_ratio']])

![image](https://github.com/BrianHodges4/MusicLyricsTrends-DataScience/assets/163497160/1af8fdec-0e4d-4531-bf1c-0f047dce7b0e)

#Use linear regression model
X = df[['year']]
y = df['ratio_scaled']

# Remove NaN values
X = X.dropna()
y = y.loc[X.index]

lm = LinearRegression().fit(X, y)
#Print out the coefficient of the year feature
print("Coefficient for year (based on unique-to-total ratio):", lm.coef_[0])
