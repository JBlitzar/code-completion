from dataset import get_train_dataset, dataset

d = get_train_dataset()

for a, b in d:
    print(a)
    print(dataset.manager.decode(a))
