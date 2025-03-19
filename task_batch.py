from torch.utils.data.sampler import Sampler
from itertools import cycle, islice
import random
import pandas as pd

class CustomBatchSampler(Sampler):
    r"""Yield a mini-batch of indices. 

    Args:
        data: Dataset for building sampling logic.
        batch_size: Size of mini-batch.
    """

    def __init__(self, examples, batch_size):
        # print(type(examples.ds))
        # print(examples.ds.columns)
        # build data for sampling here
        self.batch_size = batch_size
        self.data = {}
        # self.data = pd.DataFrame()
        # self.main_data = {}
        # get these two from config
        self.second_task_weight = 0.25
        self.task_pairs = { 'QA_de': 'translation_en_de',
                            'QA_ja': 'translation_en_ja',
                            'QA_tr': 'translation_en_tr',
                            'QA_fr': 'QA',
                            'QA_nl': 'QA',
                            'QA_zh': 'QA'}
        # self.main_tasks = ['QA_fr', 'QA_nl', 'QA_tr'] 
        # self.main_tasks = set().union(*(d.keys() for d in self.task_pairs))

        for item in examples.ds['task'].unique().tolist():
            # doible-check whether dataset indicies are the same as dataframe indicies - looks like that is the case
            inds = examples.ds.index[examples.ds['task'] == item].tolist()
            self.data[item] = inds
            # print(item)
            # size = str(item["task"])
            
            # if size in self.data:
            #     self.data[size].append(i)
            # else:
            #     self.data[size] = [i]
            
            # if size in self.task_pairs.keys():
            #     if size in self.main_data:
            #         self.main_data[size].append(i)
            #     else:
            #         self.main_data[size] = [i]

        
        # print(self.data)

        self.mixed_data = self.make_mixed_data()
        # print(self.mixed_data)

        self.total = 0
        # for size, indexes in self.data.items():
        for size, indexes in self.mixed_data.items():
            # print("All indices")
            # print(len(indexes))
            self.total += len(indexes) // self.batch_size

        
        # print(self.data)

    def make_mixed_data(self):
        mixed_data = {}
        # for later - make sure this is done iteratively - and randomly or can be seeded
        no_of_main_samples = self.batch_size - int(self.second_task_weight * self.batch_size)
        # print(no_of_main_samples)

        for main_task in self.task_pairs.keys():
            if main_task in self.data.keys():
                new_task = main_task + '_' + self.task_pairs[main_task]
                i_count = 0
                # other_task_counter = 0
                for i, item in enumerate(self.data[main_task]):
                    # print(item)
                    # size = str(item["task"])

                    if i_count == self.batch_size:
                        i_count = 0
                    
                    if new_task in mixed_data:
                        mixed_data[new_task].append(item)
                        i_count += 1
                    else:
                        mixed_data[new_task] = [item]
                        i_count += 1
                    
                    if i_count >= no_of_main_samples:
                        # mixed_data[new_task].append(list(islice(cycle(self.data[self.task_pairs[main_task]]), other_task_counter, other_task_counter + 1))[0])
                        mixed_data[new_task].append(random.choice(self.data[self.task_pairs[main_task]]))
                        # other_task_counter += 1
                        i_count += 1

            # print(mixed_data.keys())

        return mixed_data




        
        
    def __iter__(self):
        batch = []
        for size, indexes in self.mixed_data.items():
            count = len(indexes)
            # print(indexes)
            # doesn't quite do it for every single batch - check again later
            for i, idx in enumerate(indexes):
                # print('i')
                # print(idx)
                # print(size)
                batch.append(idx)

                if i == count - 1 and len(batch) < self.batch_size:
                    batch = []

                if len(batch) == self.batch_size:
                    yield batch
                    # print(batch)
                    batch = []
        
    def __len__(self):
        return self.total