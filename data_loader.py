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
    def __init__(self, genre_csv, poster_dir, transform=None, device='cpu'):
        data_csv = pd.read_csv(genre_csv, encoding='latin-1')
        self.genre_labels = data_csv[['imdbId', 'Genre']]
        self.poster_dir = poster_dir
        self.transform = transform

        self.all_genres = sorted(set(self.genre_labels['Genre'].str.split('|').explode()))
        self.genre2idx = {genre.casefold(): idx for idx, genre in enumerate(self.all_genres)}

    def genres2onehot(self, genres):
        """
        Converts list of genres to one-hot encoding. For more than one genre, the one-hot
        encoding is the sum of the one-hot encodings of the individual genres.

        Args:
            genres (list): List of genres (strings).
        """
        labels_idx = torch.tensor([self.genre2idx[genre.casefold()] for genre in genres])
        genre_encoding = nn.functional.one_hot(labels_idx, num_classes=len(self.genre2idx)).sum(dim=0)
        return genre_encoding
    
    def onehot2genres(self, one_hot):
        """
        Converts one-hot encoding to list of genres.

        Args:
            one_hot (tensor): One-hot encoding of genres.
        """
        genres = []
        for idx, val in enumerate(one_hot):
            if val == 1:
                genres.append(list(self.all_genres)[idx])
        return genres

    def __len__(self):
        return len(self.genre_labels)

    def __getitem__(self, idx):
        poster_path = os.path.join(self.poster_dir, str(self.genre_labels.iloc[idx, 0]) + '.jpg')
        poster = read_image(poster_path)
        genres = self.genre_labels.iloc[idx, 1].split('|')
        genre_encoding = self.genres2onehot(genres)
        if poster.shape[0] == 1:
            poster = poster.repeat(3, 1, 1) # If poster only has 1 channel, repeat it 3 times to make it RGB
        if self.transform:
            poster = self.transform(poster)
        return poster, genre_encoding
