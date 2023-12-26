import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
from torch import nn



class PosterDataset(Dataset):
    """
    Custom dataset for handling movie poster images and genre labels.
    """
    def __init__(self, genre_csv, poster_dir, transform=None):
        """
        Args:
            genre_csv (string): Path to the csv file with genre labels.
            poster_dir (string): Path to directory with all the movie posters as jpgs.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        data_csv = pd.read_csv(genre_csv, encoding='latin-1')
        self.movie_ids_and_genres = data_csv[['imdbId', 'Genre']]
        self.poster_dir = poster_dir
        self.transform = transform

        self._genre_set = sorted(set(self.movie_ids_and_genres['Genre'].str.split('|').explode()))
        self._genre2idx = {genre.casefold(): idx for idx, genre in enumerate(self._genre_set)}

    def genres2multihot(self, genres):
        """
        Converts list of genres to one-hot encoding. For more than one genre, the one-hot
        encoding is the sum of the one-hot encodings of the individual genres.

        Args:
            genres (list): List of genres (strings).

        Returns:
            tensor: multi-hot encoding of genres.
        """
        labels_idx = torch.tensor([self._genre2idx[genre.casefold()] for genre in genres], dtype=int)
        multihot = nn.functional.one_hot(labels_idx, num_classes=len(self._genre2idx)).sum(dim=0)
        return multihot.float() # Must be float to be passed to Linear layer
    
    def multihot2genres(self, multihot):
        """
        Converts one-hot encoding to list of genres.

        Args:
            multihot (tensor): One-hot encoding of genres.

        Returns:
            list: List of genres (strings).
        """
        idxs = torch.nonzero(multihot.int()).squeeze().tolist()
        genres = [self._genre_set[idx] for idx in idxs]
        return genres

    def __len__(self):
        return len(self.movie_ids_and_genres)

    def __getitem__(self, idx):
        poster_path = os.path.join(self.poster_dir, str(self.movie_ids_and_genres.iloc[idx, 0]) + '.jpg')
        poster = read_image(poster_path)
        genres = self.movie_ids_and_genres.iloc[idx, 1].split('|')
        genre_multihot = self.genres2multihot(genres)

        if poster.shape[0] == 1:
            poster = poster.repeat(3, 1, 1) # If poster only has 1 channel, repeat it 3 times to make it RGB
            
        if self.transform:
            poster = self.transform(poster)

        return poster, genre_multihot
