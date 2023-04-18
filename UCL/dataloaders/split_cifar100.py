import os
import numpy as np
import torch
from torchvision import datasets,transforms
from sklearn.utils import shuffle

def get(seed=0, pc_valid=0.10):
    data={}
    taskcla=[]
    size=[3, 32, 32]

    if not os.path.isdir('../data/binary_split_cifar100/'):
        os.makedirs('../data/binary_split_cifar100')

        mean=[x / 255 for x in [125.3,123.0,113.9]]
        std=[x / 255 for x in [63.0,62.1,66.7]]
        
        # CIFAR100
        dat = {}
        
        dat['train']=datasets.CIFAR100('../data/',train=True,download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100('../data/',train=False,download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for n in range(10):
            data[n] = {}
            data[n]['name'] = 'cifar100'
            data[n]['ncla'] = 10
            data[n]['train'] = {'X': [],'y': []}
            data[n]['test'] = {'X': [],'y': []}

        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            for image,target in loader:
                task_idx = target.numpy()[0] // 10
                data[task_idx][s]['X'].append(image)
                data[task_idx][s]['y'].append(target.numpy()[0] % 10)

        # "Unify" and save
        for t in range(10):
            for s in ['train','test']:
                data[t][s]['X']=torch.stack(data[t][s]['X']).view(-1,size[0],size[1],size[2])
                data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
                torch.save(data[t][s]['X'], os.path.join(os.path.expanduser('../data/binary_split_cifar100'),
                                                         'data'+str(t+1)+s+'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser('../data/binary_split_cifar100'),
                                                         'data'+str(t+1)+s+'y.bin'))
    
    # Load binary files
    data={}
    data[0] = dict.fromkeys(['name','ncla','train','test'])
    ids=list(shuffle(np.arange(10),random_state=seed)+1)
    print('Task order =',ids)
    for i in range(10):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'X':[],'y':[]}
            data[i][s]['X']=torch.load(os.path.join(os.path.expanduser('../data/binary_split_cifar100'),
                                                    'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('../data/binary_split_cifar100'),
                                                    'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name']='cifar100-'+str(ids[i-1])
            
    # Validation
    for t in range(10):
        r=np.arange(data[t]['train']['X'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['X']=data[t]['train']['X'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['X']=data[t]['train']['X'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    n=0
    for t in range(10):
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size