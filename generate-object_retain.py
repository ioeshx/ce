from util.template import imagenet_classes
import pandas as pd


if __name__ == "__main__":
    df = pd.DataFrame({
        "id": list(range(1, len(imagenet_classes) + 1)),
        "class": imagenet_classes
    })
    df.to_csv("data/object.csv", index=False)