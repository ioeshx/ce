import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("data/coco-1k.csv")
    df = df.sort_values(by=["image_id"], ascending=True)
    # df.to_csv("data/coco-1k-sort.csv", index=False)

    # df_new =df.head(100)
    # df_new.to_csv("data/coco-100.csv", index=False)

    df_new =df.head(20)
    df_new.to_csv("data/coco-20.csv", index=False)

