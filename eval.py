import argparse
from typing import List
from pathlib import Path

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr

import lpips
from pytorch_fid import fid_score
from pytorch_msssim import ssim
import re

def tag(name:str):
    def wrapper(func):
        func.tag = name
        return func
    return wrapper

class Factory(object):
    def __init__(self, name: List[str]):
        self.name = name
        methods = {func for func in dir(self) if callable(getattr(self, func)) and hasattr(getattr(self, func), 'tag')}
        self.tagged_method = {getattr(self, func).tag : getattr(self, func) for func in methods}
        self._call_func = self.get_method(name)

    # def retrieve(self, input_dir, pred_dir):
    #     def get_num(path):
    #         nums = re.findall(r'\d+', path.stem)
    #         return int(nums[0]) if nums else None
    #     input_path = list(Path(input_dir).glob('*.png')) + list(Path(input_dir).glob('*.jpg'))
    #     pred_path = list(Path(pred_dir).glob('*.png')) + list(Path(pred_dir).glob('*.jpg'))

    #     input_path = sorted(input_path, key=get_num)
    #     pred_path = sorted(pred_path, key=get_num)
        
    #     # print(input_path, pred_path)

    #     return input_path, pred_path

    def retrieve(self, input_dir, pred_dir):
        # 定义获取数字的逻辑
        def get_num(path):
            nums = re.findall(r'\d+', path.stem)
            return int(nums[0]) if nums else 0

        input_path_obj = Path(input_dir)
        pred_path_obj = Path(pred_dir)

        # 处理 input_path
        if input_path_obj.is_file():
            in_files = [input_path_obj]
        else:
            in_files = list(input_path_obj.glob('*.png')) + list(input_path_obj.glob('*.jpg'))
            in_files = sorted(in_files, key=get_num)

        # 处理 pred_path
        if pred_path_obj.is_file():
            pred_files = [pred_path_obj]
        else:
            pred_files = list(pred_path_obj.glob('*.png')) + list(pred_path_obj.glob('*.jpg'))
            pred_files = sorted(pred_files, key=get_num)

        # 增加防御性判断：如果两边数量不一致且不是单张对多张，打印警告
        if len(in_files) != len(pred_files):
            print(f"Warning: Number of images mismatch! ({len(in_files)} vs {len(pred_files)})")

        return in_files, pred_files

    def __call__(self, *args, **kwargs):
        output = []
        for _func in self._call_func:
            output.append(_func(*args, **kwargs))
        return output

    def get_method(self, name: List[str]):
        methods = []
        for n in name:
            if n not in self.tagged_method:
                raise ValueError(f'Cannot find {self.__class__.__name__} ({n})')
            else:
                methods.append(self.tagged_method[n])
        return methods

class Metric(Factory):
    @tag('psnr')
    def _psnr(self, input_path, pred_path, transform=None, data_range:int=255, **kwargs):
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

        values = []
        in_fs, pred_fs = self.retrieve(input_path, pred_path)
        for in_f, pred_f in zip(in_fs, pred_fs):
            try:
                img1 = np.array(transform(Image.open(in_f).convert('RGB'))) * data_range
                img2 = np.array(transform(Image.open(pred_f).convert('RGB'))) * data_range
                values.append(psnr(img1, img2, data_range=data_range))
            except:
                continue

        return np.mean(values)

    @tag('ssim')
    def _ssim(self, input_path, pred_path, transform=None, data_range:int=255, **kwargs):
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

        values = []
        in_fs, pred_fs = self.retrieve(input_path, pred_path)
        for in_f, pred_f in zip(in_fs, pred_fs):
            try:
                img1 = transform(Image.open(in_f).convert('RGB')).unsqueeze(0) * data_range
                img2 = transform(Image.open(pred_f).convert('RGB')).unsqueeze(0) * data_range
                values.append(ssim(img1, img2).item())
            except:
                continue

        return np.mean(values)

    @tag('fid')
    def _fid(self, pred_path, label_path, **kwargs):
        return fid_score.calculate_fid_given_paths([str(pred_path), str(label_path)], 50, 'cuda', 2048).item()
    
    @tag('lpips')
    def _lpips(self, input_path, pred_path, transform=None, **kwargs):
        lpips_fn = lpips.LPIPS(net='vgg').to('cuda').eval()
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

        values = []
        in_fs, pred_fs = self.retrieve(input_path, pred_path)
        for in_f, pred_f in zip(in_fs, pred_fs):
            try:
                img1 = transform(Image.open(in_f).convert('RGB')).to('cuda')
                img2 = transform(Image.open(pred_f).convert('RGB')).to('cuda')
                values.append(lpips_fn(img1, img2).item())
            except:
                continue

        return np.mean(values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=Path)
    parser.add_argument('--path2', type=Path)
    parser.add_argument('--save_file', type=Path)
    parser.add_argument('--metric', type=str, nargs='+')
    parser.add_argument('--prompt', type=str)
    args = parser.parse_args()

    metric = Metric(args.metric)
    output = metric(args.path1, args.path2, prompt=args.prompt)

    args.save_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_file, "w") as f:
        f.write("Metrics Log\n")
        for m, o in zip(args.metric, output):
            print(f'{m}: {o}')
            f.write(f"{m}: {o}\n")
