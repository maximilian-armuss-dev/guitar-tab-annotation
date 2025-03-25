from pathlib import Path
from typing import List
from PIL import Image
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
import torch


@dataclass
class TrainingExample:
    image_path: Path
    image: torch.Tensor
    label: torch.Tensor

    def to_tuple(self):
        return self.image, self.label

    def __str__(self):
        return f"Image path: {self.image_path}, label: {self.label}"


class TrainDataset(Dataset, ABC):
    def __init__(self, csv_path: Path, image_dir: Path, transform: callable = lambda x: x,
                 augmentations: callable = None, dataframe_keys: List[str] = None):
        super().__init__()
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.transform = transform
        self.augmentations = augmentations
        self.dataframe_keys = dataframe_keys
        self.data: List[TrainingExample] = self.from_csv()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        image, label = self.data[idx].to_tuple()
        if self.augmentations is not None:
            image = self.augmentations(image)
        return image, label

    def create_train_example(self, row) -> TrainingExample:
        image_path = self.image_dir / row["filename"]
        pil_image = Image.open(image_path)
        image = self.transform(pil_image)
        label = self.create_label(row)
        train_ex = TrainingExample(image=image, image_path=image_path, label=label)
        return train_ex

    def check_row_complete(self, row) -> bool:
        keys_in_row = [ not pd.isna(row[key]) for key in self.dataframe_keys ]
        all_rows_complete = all(keys_in_row)
        return all_rows_complete

    def from_csv(self) -> List[TrainingExample]:
        dataframe = pd.read_csv(self.csv_path)
        train_examples = [ self.create_train_example(row)
                           for index, row in dataframe.iterrows()
                           if self.check_row_complete(row) ]
        return train_examples

    @abstractmethod
    def create_label(self, row) -> torch.Tensor:
        pass


class TabTrainDataset(TrainDataset):
    def __init__(self, csv_path: Path, image_dir: Path, transform: callable = lambda x: x,
                 augmentations: callable = None, use_finger_labels = False) -> None:
        self.use_finger_labels = use_finger_labels
        super().__init__(csv_path, image_dir, transform, augmentations, ["fret_indices", "finger_indices"])

    def create_label(self, row) -> torch.Tensor:
        """
        Converts tab notation to class indices.
        :param row: pandas dataframe row containing
        * "fret_indices": List[str], e.g., ["x", "0", "3", "2", "2", "4"]
        * "finger_indices": List[str], e.g., ["1", "3", "2", "2"]
        :return: Tuple containing two tensors:
                 - fret indices: [0, 1, 4, 3, 3, 5]
                 - finger indices: [0, 0, 1, 3, 2, 2]
        Rules:
        * Frets:
          - Class index = 0 if fret == "x"
          - Class index = fret value + 1 otherwise
        * Fingers:
          - Class index = 0 if fret is "x" or "0"
          - Class index = corresponding finger value otherwise
        """
        frets: List[str] = row["fret_indices"].split()
        frets_int = [ -1 if fret == 'x' else int(fret)
                      for fret in frets ]
        if not self.use_finger_labels:
            return torch.tensor(frets_int, dtype=torch.long) + 1

        fingers: List[str] = row["finger_indices"].split()
        fingers_int = [ 0 if fret < 1 else int(fingers.pop(0))
                        for fret in frets_int ]
        return torch.tensor(fingers_int, dtype=torch.long)


class FretOffsetTrainDataset(TrainDataset):
    def __init__(self, csv_path: Path, image_dir: Path, transform: callable = lambda x: x,
                 augmentations: callable = None) -> None:
        super().__init__(csv_path, image_dir, transform, augmentations, ["fret_offset"])

    def create_label(self, row) -> torch.Tensor:
        offsets_int = [int(row["fret_offset"])]
        return torch.tensor(offsets_int, dtype=torch.long)


class TestDataset(Dataset):
    def __init__(self, data: List[torch.Tensor]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        train_im = self.data[idx]
        return train_im

    @classmethod
    def from_dir(cls, image_dir: Path, filenames: List[str], transform: callable) -> 'TestDataset':
        images = []
        if not filenames:
            filenames = image_dir.iterdir()
        for filename in filenames:
            image_path = image_dir / filename
            pil_image = Image.open(image_path)
            image = transform(pil_image)
            images.append(image)
        return cls(images)
