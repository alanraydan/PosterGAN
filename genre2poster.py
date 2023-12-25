import time
import torch

from data_loader import PosterDataset
from networks import Generator


def delayed_print(text, delay=0.03):
    """Prints out text with a delay between each character."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()


def main():
    # Instantiate dataset and generator
    dataset = PosterDataset(genre_csv='MovieGenre_cleaned.csv', poster_dir='posters')
    poster_generator = Generator(latent_dim=100, n_classes=len(dataset.all_genres))
    poster_generator.load_state_dict(torch.load('80epochs/G.pt'))
    poster_generator.eval()

    print()
    delayed_print('Welcome to PosterGAN!')
    time.sleep(1)
    print()
    delayed_print('Please enter one or more genres from the list below and I will generate a movie poster from those genres:')
    print()

    # Print all genres with three genres to a line evenly separated so that each line is roughly 80 characters long
    for i in range(0, len(dataset.all_genres), 4):
        print(f'{dataset.all_genres[i]:<18}{dataset.all_genres[i + 1]:<18}{dataset.all_genres[i + 2]:<18}{dataset.all_genres[i + 3]:<18}')
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
            lower_all_genres = [genre.casefold() for genre in dataset.all_genres]

            # Check if genres are valid
            if set(lower_genres) <= set(lower_all_genres):
                genres_are_valid = True
            else:
                delayed_print('Invalid genre(s). Please try again.')
        
        delayed_print('Generating movie poster...')
        print()
        time.sleep(1)
        genre_embedding = dataset.genres2onehot(genres).unsqueeze(0)
        poster = poster_generator.generate_poster(genre_embedding)
        poster.show()

        # Check if user wants to generate another poster
        response = input('Would you like to generate another poster? (y/n): ')
        print()
        if response.lower() == 'n':
            done = True

    delayed_print('Thanks for using PosterGAN!')
    delayed_print('Goodbye!') 


if __name__ == '__main__':
    main()