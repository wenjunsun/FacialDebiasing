from datasets.generic import GenericImageDataset
from pathlib import Path
from logger import logger
from PIL import Image
import pandas as pd
import os

class XRayPositiveDataset(GenericImageDataset):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.store = self.init_store()
        self.classification_label = 1

    def read_image(self, idx: int):
        # self.path_to_images = data/CheXpert_Images
        # self.store.iloc[idex]['Path'] = 'CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg'
        # return Image.open(os.path.join(
        #     self.path_to_images,
        #     self.store.iloc[idx]['Path']
        # ))
        return Image.open(self.path_to_images + '/' + self.store.iloc[idx]['Path']).convert("RGB")

    def init_store(self):
        # metadata tells us the path and label of each image.
        path_to_metadata = 'data/CheXpert_Images/CheXpert-v1.0-small/train_preprocessed.csv'
        metadata = pd.read_csv(path_to_metadata)
        data_has_fiding = metadata[metadata['Has Finding'] == 1]
        return data_has_fiding

    def __len__(self):
        return len(self.store)