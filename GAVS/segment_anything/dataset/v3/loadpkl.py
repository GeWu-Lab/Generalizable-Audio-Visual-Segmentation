import pandas as pd
import pickle

num_shot = 3
split = 'train'
# split = 'test'

# 读取.pkl文件
with open(f'./v3_{num_shot}_shot/{split}.pkl', 'rb') as file:
    data = pickle.load(file)

# print(data)

new_data = []
for d in data:
    # print(d)
    new_dict = {
        'vid': d['vid'].rsplit('_', 2)[0],
        'uid': d['vid'],
        's_min': '',
        's_sec': '',
        'a_obj': '',
        'split': 'v3_1_shot_train',
        'label': d['label'],
    }
    print(new_dict)
    # input()
    new_data.append(new_dict)

# 将字典列表转换为DataFrame
df = pd.DataFrame(new_data)

# 将DataFrame保存为CSV文件
df.to_csv(f'./v3_{num_shot}_shot/{split}.csv', index=False)

print("CSV文件已生成.")
