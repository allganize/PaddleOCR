import random

with open('../data/train_data/simple_table/gt.txt') as f:
    lines = f.readlines()
    
random.shuffle(lines)

split_index = 1000

train_list = lines[:split_index]
val_list = lines[split_index:]

with open('../data/train_data/simple_table/train.txt','w',encoding='utf-8') as f:
    f.writelines(train_list)

with open('../data/train_data/simple_table/val.txt','w',encoding='utf-8') as f:
    f.writelines(val_list)