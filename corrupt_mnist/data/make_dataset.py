import os


def download_data():
    base_path = "https://github.com/SkafteNicki/dtu_mlops/blob/main/data/corruptmnist/"

    os.system(f"mkdir -p data/raw")
    os.system(f"mkdir -p data/processed")

    os.system(f"wget {base_path}/test_images.pt")
    os.system("mv test_images.pt data/raw/test_images.pt")
    os.system(f"wget {base_path}/test_target.pt")
    os.system("mv test_target.pt data/raw/test_target.pt")

    for i in range(6):
        os.system(f"wget {base_path}/train_images_{i}.pt")
        os.system(f"mv train_images_{i}.pt data/raw/train_images_{i}.pt")
        os.system(f"wget {base_path}/train_target_{i}.pt")
        os.system(f"mv train_target_{i}.pt data/raw/train_target_{i}.pt")
    
if __name__ == "__main__":
    # Get the data and process it
    print('here I am')
    print(os.listdir('.'))

    import torch

    if os.path.exists("data/raw/train_images_0.pt"):
        print("Data already downloaded")
    else:   
        download_data()

    chunks = len([x for x in os.listdir("data/raw/") if "train_images" in x])
    imgs = []
    for i in range(chunks):
        img_set = torch.load(f"data/raw/train_images_{i}.pt")
        imgs.append(img_set)
    data = torch.cat(imgs, dim=0)

    # get mean and std
    mean = torch.mean(data)
    std = torch.std(data)

    # normalize all in imgs
    for i in range(chunks):
        normalized = (imgs[i] - mean) / std
        torch.save(normalized, f"data/processed/train_images_{i}.pt")

    # normalize test set
    test = torch.load("data/raw/test_images.pt")
    test = (test - mean) / std
    torch.save(test, "data/processed/test_images.pt")

    # finally copy the labels
    import os

    for i in range(chunks):
        os.system(f"cp data/raw/train_target_{i}.pt data/processed/train_target_{i}.pt")
    os.system("cp data/raw/test_target.pt data/processed/test_target.pt")
