import unittest
import os
import sys
import torch
import shutil
from torch.utils.data import DataLoader
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# from image_datasets.dataset import loader
from image_datasets.dataset_processed import loader

class TestDataLoader(unittest.TestCase):
    def test_loader(self):
        # 使用 loader 函数创建 DataLoader
        self.test_dir = "/datasets/flux_shortsentence"
        dataloader = loader(train_batch_size=4, num_workers=2, img_dir=self.test_dir)
        print("len(dataloader) = ", len(dataloader))
        # 检查 DataLoader 是否返回正确的批次大小
        for batch in dataloader:
            print("len(batch) = ", len(batch))
            print(batch[0].shape)
            self.assertEqual(batch[0].shape, (4, 1024, 64))
            break  # 只检查一个批次

    def test_loader_with_exception(self):
        self.test_dir = "/datasets/flux_shortsentence"

        dataloader = loader(train_batch_size=2, num_workers=0, img_dir=self.test_dir)
        
        # 检查 DataLoader 是否能够继续工作并且不抛出异常
        for batch in dataloader:
            self.assertEqual(batch[0].shape, (2, 1024, 64))
            break  # 只检查一个批次

if __name__ == '__main__':
    unittest.main()