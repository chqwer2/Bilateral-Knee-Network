
import pandas as pd
import numpy as np
try:
    from datasets.dataloaders import NewDataset, NewLoader, new_split_data, read_img
except:
    from .datasets.dataloaders import NewDataset, NewLoader, new_split_data, read_img



def get_datasets(train_path, test_path1, test_path2, bs=16, num_workers=8):
    
    data = pd.read_csv(train_path)
    test_data1 = pd.read_csv(test_path1)
    test_data2 = pd.read_csv(test_path2)
    
    train_dl, val_dl = NewLoader(data=data, 
                                batch_size=bs, 
                                num_workers=num_workers, 
                                is_sampler=True, 
                                is_test=False)
    
    test_dl1 = NewLoader(data=test_data1, 
                        is_sampler=False, 
                        is_test=True)
    
    test_dl2 = NewLoader(data=test_data2, 
                        is_sampler=False, 
                        is_test=True)
    
    test_dls = [test_dl1, test_dl2]
    
    return train_dl, val_dl, test_dls