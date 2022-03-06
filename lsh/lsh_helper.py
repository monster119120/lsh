import random

from lshashing import LSHRandom
import numpy as np
import torch


def lsh_sampling(model, data_loader, select_num, device):
    to_label_list = []
    all_preds = []
    all_indices = []
    model.to(device)
    model.eval()
    
    for i, (images, _) in enumerate(data_loader):
        images = images.to(device)
        
        with torch.no_grad():
            preds = model(images)
        all_preds.extend(preds.detach().cpu().numpy())
        all_indices.append(i)

    all_preds = torch.stack(all_preds)
    all_preds = all_preds.view(-1)

    all_indices = np.asarray(all_indices)
    
    lsh_results = LSHRandom(all_preds, hash_len=6, num_tables=2)
    
    table = lsh_results.tables[0].hash_table
    
    while True:
        for k in table:
            if len(to_label_list) >= select_num:
                return all_indices[to_label_list]
            if len(table[k]) > 0:
                to_label_list.append(table[k].pop(random.randrange(len(table[k]))))