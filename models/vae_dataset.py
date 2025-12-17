import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

class VAENpDataset(Dataset):
    def __init__(self, path_to_np_files, img_size):
        self.data = self.prepare(path_to_np_files)
        self.transform = transforms.Compose([transforms.Resize(img_size),
                                             transforms.ToTensor(),

                                             ])

    def __len__(self):
        return self.data.shape[0]

    @staticmethod
    def prepare(path_to_np_files:str):
        path = Path(str(path_to_np_files)).glob('**/*')
        files = [x for x in path if x.is_file()]

        collected_data = {}
        for pth in tqdm(files):
            with np.load(str(pth)) as data:
                for key in data.files:
                    if key not in collected_data:
                        collected_data[key] = []

                    collected_data[key].append(data[key])
        final_dataset = {}
        for key, arrays_list in collected_data.items():
            # axis=0 склеивает по длине (добавляет новые строки/кадры вниз)
            final_dataset[key] = np.concatenate(arrays_list, axis=0)

        return final_dataset['obs']

    def __getitem__(self, idx):
        obs = self.data[idx]
        image_obs = Image.fromarray(obs)
        image_obs = self.transform(image_obs)
        image_obs = image_obs*2-1
        return image_obs
