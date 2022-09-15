import torchvision
import numpy as np
import random

class SubLoader(torchvision.datasets.CIFAR10):
    def __init__(self, *args, include_list=[], num_data=50,**kwargs):
        super(SubLoader, self).__init__(*args, **kwargs)

        if include_list == []:
            return

        data=[]
        targets=[]
        for i in include_list:

            exclude_list=np.delete(np.arange(10),i)
            labels = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

            data.extend(self.data[mask][:num_data])
            targets.extend(labels[mask].tolist()[:num_data])

        self.data=data
        self.targets=targets

        '''

        #i=0
        exclude_list=[1,2,3,4,5,6,7,8,9,10]
        labels = np.array(self.targets)
        exclude = np.array(exclude_list).reshape(1, -1)
        mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

        self.data_1 = self.data[mask][:num_data]
        self.targets_1 = labels[mask].tolist()[:num_data]
        
        self.data=self.data_0 + self.data_1
        self.targets=self.targets_0 + self.targets_1
        '''