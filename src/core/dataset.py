import json
import random
import pandas as pd
from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset, DataLoader

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
        true_data = self.__build_real_data()
        fake_data = self.__build_fake_data()
        data = true_data + fake_data
        print("Number of Real example", len(true_data))
        print("Number of Fake example", len(fake_data))
        return sorted(data, key=lambda x: x['name'])

    @staticmethod
    def __filter_data(selects, candidates):
        """
        The candidates are checked to see if the element in selects exists, and if so, the element is moved to selects.
        Args:
            selects: selected data
            candidates: candidate data

        Returns:

        """
        for candidate in candidates:
            matched_item = [item for item in selects if item["name"] == candidate['name']]
            if matched_item:
                candidates.remove(candidate)
                selects.extend(matched_item)
        return selects, candidates

    def build_dataset(self, save_to_disk=False):
        """
        The dataset is divided into training and test sets, and the data is saved to the disk.
        Args:
            save_to_disk: whether to save the data to the disk

        Returns:

        """
        data = self.__contact_data()
        random.shuffle(data)
        split_radio = 0.8
        train_size = int(len(data) * split_radio)
        train_set = data[:train_size]
        test_set = data[train_size:]
        dataset = DatasetDict()
        dataset["train"] = Dataset.from_list(train_set)
        dataset["test"] = Dataset.from_list(test_set)
        if save_to_disk:
            pd.DataFrame(train_set).to_csv(f"{DATA_PATH}/dataset/train.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(test_set).to_csv(f"{DATA_PATH}/dataset/test.csv", index=False, encoding="utf-8-sig")
        return dataset


class SentenceDataset(TorchDataset):
    """
    The dataset is used to load the data in the form of a sentence.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][0], self.dataset[index][1], self.dataset[index][-1]


def get_dataloader():
    """
    Get the training and test data loader.
    Returns:

    """
    batch_size = 1
    train_dataset = pd.read_csv(f"{DATA_PATH}/dataset/train.csv", encoding="utf-8-sig").values.tolist()
    train_dataloader = DataLoader(SentenceDataset(train_dataset), batch_size=batch_size, shuffle=True)
    test_dataset = pd.read_csv(f"{DATA_PATH}/dataset/test.csv", encoding="utf-8-sig").values.tolist()
    test_dataloader = DataLoader(SentenceDataset(test_dataset), batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader
