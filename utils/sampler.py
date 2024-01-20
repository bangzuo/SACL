import numpy as np
import random

class UniformSampler():
    def __init__(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def sample_negative(self, users, n_items, train_user_dict, sample_num):
        row = sample_num
        col = users.shape[0]#为每个用户都采样

        samples_array = np.zeros((col, row), dtype=np.int64)
        for user_i, user in enumerate(users):
            pos_items = train_user_dict[user]#正样本是在这个里面键为user，查到的项目id

            for i in range(0, row): #row是采样数目
                while True:
                    neg = random.randint(0,n_items-1)#在项目范围之内随机采样一个，然后保证不是在正样本之中
                    if neg not in pos_items:
                        break
                samples_array[user_i, i] = neg
        return samples_array
        #返回所有用户采样的负样本
