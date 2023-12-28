import time
import torch

from data_loader import PosterDataset
from networks import Generator


def delayed_print(text, delay=0.01):
    """Prints out text with a delay between each character."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()


def main(G_path):
    # Instantiate dataset and generator
    dataset = PosterDataset(genre_csv='MovieGenre_cleaned.csv', poster_dir='posters')
    poster_generator = Generator(latent_dim=100, n_classes=len(dataset.genre_set), class_embedding_dim=16)
    checkpoint = torch.load(G_path)
    poster_generator.load_state_dict(checkpoint['model_state_dict'])
    poster_generator.eval()

    print()
    delayed_print('Welcome to PosterGAN!')
    time.sleep(1)
    print()
    delayed_print('Please enter one or more genres from the list below and I will generate a movie poster from those genres:')
    print()

    # Print all genres with three genres to a line evenly separated so that each line is roughly 80 characters long
    for i in range(0, len(dataset.genre_set), 4):
        print(f'{dataset.genre_set[i]:<18}{dataset.genre_set[i + 1]:<18}{dataset.genre_set[i + 2]:<18}{dataset.genre_set[i + 3]:<18}')
        time.sleep(0.1)
    print()

    # Generate poster(s) until user is done
    done = False
    while not done:
        genres_are_valid = False
        while not genres_are_valid:
            # Get user input
            genres = input('Enter genre(s) separated by spaces: ').split(' ')
            genres = [genre.capitalize() for genre in genres]

            lower_genres = [genre.casefold() for genre in genres]
            lower_genre_set = [genre.casefold() for genre in dataset.genre_set]

            # Check if genres are valid
            if set(lower_genres) <= set(lower_genre_set):
                genres_are_valid = True
            else:
                delayed_print('Invalid genre(s). Please try again.')
        
        delayed_print('Generating movie poster...')
        print()
        time.sleep(1)
        genre_multihot = dataset.genres2multihot(genres).unsqueeze(0)
        poster = poster_generator.generate_poster(genre_multihot)
        poster.show()

        # Check if user wants to generate another poster
        response = input('Would you like to generate another poster? (y/n): ')
        print()
        if response.lower() == 'n':
            done = True

    delayed_print('Thanks for using PosterGAN!')
    delayed_print('Goodbye!') 


if __name__ == '__main__':
    main('G.pt')