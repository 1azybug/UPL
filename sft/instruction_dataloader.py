from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch

class CompressDataset(IterableDataset):
    def __init__(self, examples, batch_size):
        super(CompressDataset).__init__()
        self.examples = examples
        self.batch_size = batch_size


    def __iter__(self):
        batch = {key: [] for key in self.examples[0].keys()}
        count = 0  # 初始化计数器

        for example in self.examples:
            for key, value in example.items():
                batch[key].append(value)
            
            count += 1  # 更新计数器

            # 当计数器达到 batch_size 时 yield 批次
            if count == self.batch_size:
                # yield {key: torch.stack(value) for key, value in batch.items()}
                # only for batch_size == 1 
                assert self.batch_size==1
                yield {key: torch.stack(value) if value[0] is not None else None for key, value in batch.items()}
                batch = {key: [] for key in batch}  # 重置批次
                count = 0  # 重置计数器
                
                
def get_dataset(task_type, examples, batch_size):
    if task_type == "Compress":
        return CompressDataset(examples, batch_size)
    
    raise Exception("Don't exist [{task_type}] task.")