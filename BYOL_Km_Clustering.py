import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from byol_pytorch import BYOL
import pytorch_lightning as pl
import glob
import random
import wandb
import math
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
################################ wandb.login()

# test model, a resnet 50
gnet = models.googlenet(pretrained=True)     #pretrained = false even also resulted 0.2 silhoutte and loss to upto 0.49



config = {
         #"key1": "value1",
         #"key2": "value2"

         # constants
         "Random_SEED": 2022,
         "BATCH_SIZE" : 96,
         "EPOCHS"     : 25,
         "LR"         : 8e-3,
         "NUM_GPUS"   : 2,
         "IMAGE_SIZE" : 512, # change from 256 to 51
         "NUM_WORKERS" : 2,  #multiprocessing.cpu_count
         "n_label" : 4,
         "NUM_clust" : 8,
         "exp_name" : "24thApr_1533"
         }


#config_path = "/workspace/configuration/"
#create a json file containing the hyper parameters 
config_json = os.path.join("/workspace/configuration/", (str(config["exp_name"])+".json"))
print(config_json)
with open(config_json, 'w') as f:
    json.dump(config, f)



print(f'The configuration file for the experiment named {config["exp_name"]} has been saved.')
# loading the constants from config file

Random_SEED = config["Random_SEED"]
BATCH_SIZE = config["BATCH_SIZE"]
EPOCHS     = config["EPOCHS"]
LR         = config["LR"]
NUM_GPUS   = config["NUM_GPUS"]
IMAGE_SIZE = config["IMAGE_SIZE"]   # change from 256 to 512
IMAGE_EXTS = ['.jpg', '.png', '.jpeg','.tif']

NUM_WORKERS = config["NUM_WORKERS"] #multiprocessing.cpu_count
n_label = config["n_label"]
NUM_Clust = config["NUM_clust"]
exp_name = config["exp_name"]


img_paths = glob.glob('/workspace/Data/solo_train_copy/*/*.tif')
print(f'Number of images: {len(img_paths)}')
result_path = "/workspace/byol_results/"
model_path = "/workspace/model/"+"byol_"+str(exp_name)+".pth"
print(model_path)



random.seed(Random_SEED)


# # resnet50 = models.resnet50(pretrained=True)
parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--image_folder', type=str, required = True,
                       help='path to your folder of images for self-supervised learning')

args = parser.parse_args()


img_paths = glob.glob('/workspace/Data/solo_train_copy/*/*.tif')
print(f'Number of images: {len(img_paths)}')

print(f'Number of images: {len(img_paths)}')

# pytorch lightning module

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)
        # wandb.log({'Learner':self.learner})

    def forward(self, images):
        # print(f"learning parameters is {self.learner(images)}")
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        # print(f"Loss value is {loss}")
        # wandb.log({'Loss': loss})
        return {'loss': loss}



    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()


# WANDB metric
# ##############wandb.init(project="byol_bach", name="byol_run", config={"learning_rate": 0.0003})
# wandb.log({"Loss": loss, "Learner": self.learner})


# images dataset


# def expand_greyscale(t):
#     return t.expand(3, -1, -1)

class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []
        self.image_size = image_size

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

     ######################################################   print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        
        ### print(img.size)
        
        #n = ranint (min(img.shape), image_size)
        # print(f'minimum size of the image {min(img.size)} and image size {self.image_size}')
        random_img_size = random.randint(self.image_size, min(img.size))
        ###print(random_img_size)
        off_set = math.floor(random_img_size/2)  

        #loc = #generate random location of int in the range ()
        random_img_loc_x = random.randint(off_set, img.size[0]-off_set)
        random_img_loc_y = random.randint(off_set, img.size[1]-off_set)

        # Crop the image
        cropped_image = img.crop((random_img_loc_x, random_img_loc_y,random_img_loc_x + (random_img_size/2), random_img_loc_y + (random_img_size/2)))


        #####################################################print(self.transform(cropped_image).size)
# 
        cropped_image = cropped_image.convert('RGB')
        return self.transform(cropped_image)
    
# main
    

if __name__ == '__main__':
    ds = ImagesDataset(args.image_folder, IMAGE_SIZE)
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    print("The data has been loaded to train the model by BYOL method")

    model = SelfSupervisedLearner(
        gnet,
        image_size = IMAGE_SIZE,
        # hidden_layer = 'avgpool',
        projection_size = 256,  # change 512 from 256
        projection_hidden_size = 4096,
        moving_average_decay = 0.99
    )
    print(f"The model is ready.\n")
    
    trainer = pl.Trainer(
        devices = 1,
        accelerator="gpu",
        max_epochs = EPOCHS,
        accumulate_grad_batches = 1,
        sync_batchnorm = True
    )


    trainer.fit(model, train_loader)
    print("Traing has been completed.")
    
    img_features = []
    ds_c = ImagesDataset(args.image_folder, IMAGE_SIZE)
    img_loader = DataLoader(ds_c, batch_size=32, num_workers=2, shuffle=False)
    
    
    # torch.set_grad_enabled(False)
    gnet.eval()
    for img in img_loader:
        #print(img.size())
        output = resnet(img)
        img_feature = output.detach()
        img_feature =torch.squeeze(img_feature)
        img_feature = img_feature.numpy()

        # Normalize the data over the batch
        image_feature_norm = (img_feature-np.min(img_feature))/ (np.max(img_feature)-np.min(img_feature))

        #########print(f'The length of the img_features l
        ######## print(f'The shape of the output from the model is: {img_feature.size()}') # 32 512
        # img_feature = np.concatenate(img_feature, axis=0)
        # print(f'The shape of the output after normalisation: {image_feature_norm.shape}') # 32 512

        for idx in range(image_feature_norm.shape[0]):
            img_features.append(image_feature_norm[idx,:])
    print(len(img_features))

    

    
        # Normalize the data over the batch
        image_feature_norm = (img_feature-np.min(img_feature))/ (np.max(img_feature)-np.min(img_feature))

        #########print(f'The length of the img_features l
        ######## print(f'The shape of the output from the model is: {img_feature.size()}') # 32 512
        # img_feature = np.concatenate(img_feature, axis=0)
        # print(f'The shape of the output after normalisation: {image_feature_norm.shape}') # 32 512

        for idx in range(image_feature_norm.shape[0]):
            img_features.append(image_feature_norm[idx,:])
 
    
    print(len(img_features))

    silhouette_list = []
    wcss_list = []
    labels_list = []
    centroids_list = []

    for NUM_Clust in range(9):

        # Choose the number of clusters
        num_clusters = NUM_Clust+2
        print(f"The number of cluster is: {NUM_Clust}")


        # Run K-means algorithm
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(img_features)

        # Evaluate the clusters
        wcss = kmeans.inertia_
        wcss_list.append(wcss)
        print(f"Within-cluster sum of squares is {wcss} for {num_clusters} clusters")

        # Visualize the clusters
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        labels_list.append(labels)
        centroids_list.append(centroids)
        #print(f'label is {labels}')
        #print(f'centroid is {centroids}')  

        # Evaluate the algorithm using the silhouette score
        silhouette_avg = silhouette_score(img_features, labels)
        silhouette_list.append(silhouette_avg)
        print(f"The average silhouette score is {silhouette_avg} for {num_clusters} clusters")

    #save centroids and labels
    result_path = "/workspace/byol_results/"

    np.save(os.path.join(result_path+"_"+(exp_name+"_centroids_list")), centroids_list)
    np.save(os.path.join(result_path+"_"+(exp_name+"_labels_list")), labels_list)
    np.save(os.path.join(result_path+"_"+(exp_name+"_wcss_list")), wcss_list)
    np.save(os.path.join(result_path+"_"+(exp_name+"_silhouette_list")), silhouette_list)



torch.save(exp_name, PATH='/workspace/')


# ##### To Do ###########

##we need extract features using the trained network by byol

##k means clustering k = no of label (keep it a argument)

##then check the purity score or get the voting by the label available
