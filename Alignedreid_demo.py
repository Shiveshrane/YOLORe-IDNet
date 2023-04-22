import numpy as np
import torch
from util.FeatureExtractor import FeatureExtractor
from torchvision import transforms
from IPython import embed
import models
from scipy.spatial.distance import cosine, euclidean
from  util.utils import *
from sklearn.preprocessing import normalize
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

def pool2d(tensor, type= 'max', batching=False):
    tensor = tensor.cuda() # size (1,2048, 8, 4)
    sz = tensor.size()
    if type == 'max':
        maxpool = torch.nn.MaxPool2d(kernel_size=(sz[2] // 8, sz[3]))
        maxpool = maxpool.cuda()
        x = maxpool(tensor) # size (1,2048,8) #first dim is batch...
    if type == 'mean':
        x = torch.nn.functional.mean_pool2d(tensor, kernel_size=(sz[2]//8, sz[3]) )

    # x_cpu = x.cpu()
    # x_numpy = x_cpu.data.numpy()
    # res = [np.transpose(data,(2,1,0))[0] for data in x_numpy]
    # res = np.array(res)
    res = [data.permute(2,1,0)[0] for data in x]
    return res

class Aligned_Reid_class:
    def __init__(self):
        self.res = []
        self.exact_list = ['7']
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        self.use_gpu = torch.cuda.is_available()
        if torch.cuda.is_available():
            self.map_location = lambda storage, loc: storage.cuda()
        else:
            self.map_location = 'cpu'

        self.model = models.init_model(name='resnet50', num_classes=751, loss={'softmax', 'metric'}, use_gpu=self.use_gpu,
                                       aligned=True)
        self.checkpoint = torch.load("./log/checkpoint_ep300.pth.tar", map_location=self.map_location, encoding='latin1')
        self.model.load_state_dict(self.checkpoint['state_dict'])  # loads module parameter
        self.myexactor = FeatureExtractor(self.model, self.exact_list)
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model.eval()
        if self.use_gpu:
            self.model = self.model.cuda()

    def compute_distance(self,path,feature_1):
        tick = time.time()

        img_path2 = path

        #img2 = read_image(img_path2)
        img2 = path

        t = time.time()
        img2 = img_to_tensor(img2, self.img_transform)
        print(time.time()-t, 'transform take seconds')

        if self.use_gpu:
            img2 = img2.cuda()
        if feature_1 is not None:
            a1 = feature_1
        temp = a1

        start = time.time()
        f2 = self.myexactor(img2)
        print(time.time()-start,'extraction took')

        p = time.time()
        a2  = pool2d(f2[0], type='max')[0]

        a2  = normalize(a2)
        print(time.time()-p,'pooling time')

        dist = np.zeros((8, 8))
        for i in range(8):
            temp_feat1 = a1[i]
            for j in range(8):
                temp_feat2 = a2[j]
                dist[i][j] = euclidean(temp_feat1, temp_feat2)

        res = self.res
        self.res = []
        tok = time.time()
        print('Re-id time per object from reid module: ', round(tok-tick,4))
        return show_alignedreid(dist),res,temp

    def torch_normalizer(self, tensor):
        normalized_tensor = tensor / tensor.norm(dim=1, keepdim=True)
        return normalized_tensor

    def extract_suspect_features(self, suspect_data):
        suspect_features_tensor = []
        for img in suspect_data:
            img = read_image(img)
            img = img_to_tensor(img, self.img_transform)
            if self.use_gpu:
                img = img.cuda()
                feat = self.myexactor(img)
                pooled_feature = pool2d(feat[0], type='max')[0]
                # a1 = normalize(pooled_feature)
                a1 = self.torch_normalizer(pooled_feature)
                suspect_features_tensor.append(a1)
        return suspect_features_tensor

    def inference(self, persons, suspect_features = None):
        s = time.time()
        transformed_imgs = [img_to_tensor(img, self.img_transform ).cuda() for img in persons]
        torch.cuda.synchronize()
        w = time.time()
        print('transform time',w-s)

        batch_tensor = torch.cat(transformed_imgs, dim=0)
        s = time.time()
        features = self.myexactor(batch_tensor)
        torch.cuda.synchronize()
        e = time.time()
        print('extration: ',e-s)
        res = pool2d(features[0],type='max')
        return res

    def inference2(self, persons, suspect_features=None):
        # start = time.time()
        transformed_imgs = [img_to_tensor(img, self.img_transform).cuda() for img in persons]
        batch_tensor = torch.cat(transformed_imgs, dim=0)
        features = self.myexactor(batch_tensor)
        res = pool2d(features[0], type='max')
        # end = time.time()
        # print('Intra Reid took..', end - start)
        return res
