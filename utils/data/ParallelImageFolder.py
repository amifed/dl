import torch
import torchvision
import os


from typing import Any, Callable, cast, Dict, List, Optional, Tuple

torchvision.datasets.ImageFolder


class ParallelImageFolder(torchvision.datasets.ImageFolder):
    def __init__(
            self, parallel_root, **kwargs
    ):
        super(ParallelImageFolder, self).__init__(**kwargs)
        self.parallel_root = parallel_root
        self.imgs = self.samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)

        _, parallel_name = os.path.split(path)
        parallel_path, parallel_sample = None, None
        if self.parallel_root is not None:
            parallel_path = os.path.join(os.path.join(
                self.parallel_root, self.classes[target], parallel_name))
            parallel_sample = self.loader(parallel_path)

        if self.transform is not None:
            sample = self.transform(sample)
            if parallel_sample is not None:
                parallel_sample = self.transform(parallel_sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, parallel_sample if parallel_sample is not None else []
