import os 

if __name__ == '__main__':
    # Get the data and process it

    import torch
    chunks = len([x for x in os.listdir('data/raw/') if 'train_images' in x])
    imgs = []
    for i in range(chunks):
        img_set = torch.load(f'data/raw/train_images_{i}.pt')
        imgs.append(img_set)
    data = torch.cat(imgs, dim=0)

    # get mean and std
    mean = torch.mean(data)
    std = torch.std(data)

    # normalize all in imgs
    for i in range(chunks):
        normalized = (imgs[i] - mean) / std
        torch.save(normalized, f'data/processed/train_images_{i}.pt')

    # normalize test set
    test = torch.load(f'data/raw/test_images.pt')
    test = (test - mean) / std
    torch.save(test, 'data/processed/test_images.pt')

    # finally copy the labels
    import os
    for i in range(chunks):
        os.system(f'cp data/raw/train_target_{i}.pt data/processed/train_target_{i}.pt')
    os.system(f'cp data/raw/test_target.pt data/processed/test_target.pt')
