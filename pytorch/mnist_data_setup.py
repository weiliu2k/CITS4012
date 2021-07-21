from pathlib import Path
import requests
import pickle
import gzip
import torch

def mnist_dataloader():

    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)

    URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
            content = requests.get(URL + FILENAME).content
            (PATH / FILENAME).open("wb").write(content)

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
            ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

   

    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    n, c = x_train.shape
    x_train, x_train.shape, y_train.min(), y_train.max()
    # print(x_train, y_train)
    # print(x_train.shape)
    # print(y_train.min(), y_train.max())
    return x_train, y_train, x_valid, y_valid        