import json
import random
import pandas as pd
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import Dataset, DatasetDict
from src.config import ALL_REWRITE_INTENT_JSON, DATA_PATH


class DataBuilder:
    def __init__(self):
        with open(ALL_REWRITE_INTENT_JSON, "r", encoding="utf-8") as file:
            self.raw_rewrite_data = json.load(file)

    @staticmethod
    def __generate_fake_index(fixed_value, length):
        while True:
            random_index = random.randint(0, length - 1)
            if random_index != fixed_value:
                return random_index

    def __build_real_data(self):
        return list(map(lambda x: {**x, "label": 1}, self.raw_rewrite_data))

    def __build_fake_data(self):
        unrelated_data = []
        for index in range(len(self.raw_rewrite_data)):
            intent = self.raw_rewrite_data[index]["name"]
            randomIndex = self.__generate_fake_index(index, len(self.raw_rewrite_data))
            description = self.raw_rewrite_data[randomIndex]['rewrite']
            unrelated_data.append({"name": intent, "rewrite": description, "label": 0})
        return unrelated_data

    def __contact_data(self):
        data = self.__build_real_data() + self.__build_fake_data()
        print("Number of Real example", len(self.__build_real_data()))
        print("Number of Fake example", len(data) - len(self.__build_real_data()))
        data = sorted(data, key=lambda x: x['name'])
        return data

    @staticmethod
    def __fold(data_list):
        num_folds = 5
        fold_size = len(data_list) // num_folds
        train_sets = []
        validation_sets = []
        for fold in range(num_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size
            validation_set = data_list[start_idx:end_idx]
            train_set = [item for item in data_list if item not in validation_set]
            train_sets.append(train_set)
            validation_sets.append(validation_set)
        return train_sets, validation_sets

    @staticmethod
    def __filter_data(selects, candidates):
        """
        检查candidates中是否含有selects中的元素,若包含则将该元素移动到selects中
        """
        for candidate in candidates:
            matched_item = [item for item in selects if item["name"] == candidate['name']]
            if matched_item:
                candidates.remove(candidate)
                selects.extend(matched_item)
        return selects, candidates

    def build_dataset(self, save_to_disk=False):
        data = self.__contact_data()
        train_sets, validate_sets = self.__fold(data)
        datasets = []
        for index, item in enumerate(train_sets):
            train_set, validate_set = self.__filter_data(item, validate_sets[index])
            dataset = DatasetDict()
            dataset["train"] = Dataset.from_list(train_set)
            dataset["test"] = Dataset.from_list(validate_set)
            datasets.append(dataset)
            if save_to_disk:
                pd.DataFrame(train_set).to_csv(f"{DATA_PATH}/dataset/train{index + 1}.csv", index=False,
                                               encoding="utf-8-sig")
                pd.DataFrame(validate_set).to_csv(f"{DATA_PATH}/dataset/test{index + 1}.csv", index=False,
                                                  encoding="utf-8-sig")
        return datasets


class SentenceDataset(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][0], self.dataset[index][1], self.dataset[index][-1]


batch_size = 1
train_dataset = pd.read_csv(f"{DATA_PATH}/dataset/train1.csv", encoding="utf-8-sig").values.tolist()
train_dataloader = DataLoader(SentenceDataset(train_dataset), batch_size=batch_size, shuffle=True)
test_dataset = pd.read_csv(f"{DATA_PATH}/dataset/test1.csv", encoding="utf-8-sig").values.tolist()
test_dataloader = DataLoader(SentenceDataset(test_dataset), batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    data_builder = DataBuilder()
    datasets = data_builder.build_dataset(save_to_disk=True)

    train_dataset = pd.read_csv(f"{DATA_PATH}/dataset/train1.csv", encoding="utf-8-sig").values.tolist()
    intent_dataset = SentenceDataset(train_dataset)

    for intent, description, label in train_dataloader:
        print("intent, description, label", intent, description, label)
        break
