import os
import torch
from torch.utils.data import Dataset, DataLoader
# from dataset import CustomImageDataset
import random

class CustomImageDatasetProcessed(Dataset):
    def __init__(self, img_dir, **kwargs):
        img_dir = os.path.abspath(img_dir)
        pt_dir = img_dir + '_processed'
        self.pt_files = [os.path.join(pt_dir, i) for i in os.listdir(pt_dir) if '.pt' in i]
        self.pt_files.sort()
        # super().__init__(img_dir, **kwargs)

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx, retries=3):
        try:
            pt_file = self.pt_files[idx]
            data = torch.load(pt_file, map_location='cpu')

            return data["img"], data["img_ids"], data["txt"], data["txt_ids"], data["vec"]

        except Exception as e:
            print(f"Error loading file {pt_file}: {e}")
            if retries > 0:
                retries = retries - 1
                return self.__getitem__(random.randint(0, len(self.pt_files) - 1))
            else:
                raise Exception(f"Failed to load file after {retries} attempts.")
            
def loader(train_batch_size, num_workers, **args):
    torch.multiprocessing.set_start_method('spawn', force=True)
    dataset = CustomImageDatasetProcessed(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
