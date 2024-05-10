from core.dataset import DataBuilder, get_dataloader

data_builder = DataBuilder()
datasets = data_builder.build_dataset(save_to_disk=True)

train_dataloader, test_dataloader = get_dataloader()
for intent, description, label in train_dataloader:
    print("intent, description, label")
    print(intent, description, label)
    break
