# Downoad all poter images from the given url in MovieGenre.csv

import pandas as pd
import os
import urllib.request
from tqdm import tqdm

df = pd.read_csv('MovieGenre.csv', encoding='latin-1')
# Remove all columns except for 'Poster' and 'Genre'
df = df[['imdbId', 'Genre', 'Poster']]
df = df.dropna()

# Create a directory to store the poster images
if not os.path.exists('posters'):
    os.makedirs('posters')

# Loop through 'Poster' column and download the images.
# Images should be stored as 'posters/{imdbId}.jpg'
# If the image cannot be downloaded, delete the row from the dataframe.
for index, row in tqdm(df.iterrows()):
    url = row['Poster']
    imdbId = row['imdbId']
    if not os.path.exists(f'posters/{imdbId}.jpg'):
        try:
            urllib.request.urlretrieve(url, f"posters/{imdbId}.jpg")
        except:
            df = df.drop(index)

# Save the dataframe as a csv file
df.to_csv('MovieGenre_cleaned.csv', index=False)


