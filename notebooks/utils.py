"""Some utils functions"""

from pathlib import Path
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

DATA_DIRECTORY_PATH = Path().absolute().parents[0] / "data"
AVAILABLE_FILES = ["answers.json", "questions.json", "users.json"]


def read_data(file_name: str) -> pd.DataFrame:
    """Read json file

    Args:
        file_name: Name of the file to read

    Returns:
        pd.DataFrame: DataFrame containing data from json
    """
    if file_name not in AVAILABLE_FILES:
        raise KeyError(f"File {file_name} doesn't exists\n Available files are: {''.join(AVAILABLE_FILES)}")

    return pd.read_json(DATA_DIRECTORY_PATH / Path(file_name), lines=True)


def parse_html_tags(text: str) -> str:
    """Get a text containing html tags and remove it

    Args:
        text: Unclean text containing html tags

    Returns:
        str: Clean text without html tags
    """
    to_return = text
    if isinstance(text, str):
        to_return = BeautifulSoup(text, "lxml").text
    return to_return


def word_cloud_generation(texts: list[str], title: str, max_words: int) -> None:
    """Generate a word cloud from texts

    Args:
        texts: Texts to parse to identify most represented words
        title: Title of the image
        max_words: Max number of words to take care
    """
    wc = WordCloud(background_color="black", max_words=max_words, stopwords=STOPWORDS)
    wc.generate(" ".join(texts))
    plt.figure(figsize=(8, 8))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.imshow(wc.recolor(colormap='gist_earth', random_state=244), alpha=0.98)
