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
resnet = models.resnet18(pretrained=False)


# Freeze all layers except the last one
# for param in resnet.parameters():
#     param.requires_grad = False

# # Get the number of input features of the last layer
# num_ftrs = resnet.fc.in_features

# # Create a new fully connected layer with 256 output features
# new_fc = torch.nn.Linear(num_ftrs, 256)

# # Replace the last layer of the ResNet18 model with the new fully connected layer
# resnet.fc = new_fc

# # Print the modified ResNet18 model
# print(resnet)
# # exit()
# # Load the ResNet-50 model without the top layer
# # resnet50 = models.resnet50(pretrained=True)
parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--image_folder', type=str, required = True,
                       help='path to your folder of images for self-supervised learning')

args = parser.parse_args()

# constants

BATCH_SIZE = 192
EPOCHS     = 50 
LR         = 5e-5
NUM_GPUS   = 2
IMAGE_SIZE = 512 # change from 256 to 512
IMAGE_EXTS = ['.jpg', '.png', '.jpeg','.tif']
NUM_WORKERS = 2  #multiprocessing.cpu_count
n_label = 4
NUM_Clust = 8

img_paths = glob.glob('/workspace/Data/solo_train_copy/*/*.tif')
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


def expand_greyscale(t):
    return t.expand(3, -1, -1)

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
            transforms.Lambda(expand_greyscale)
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
        resnet,
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
    
        #################################################### # loop over the epochs
        ##################for epoch in range(EPOCHS):
        # loop over the batches in the epoch
        ###################for step, (inputs, targets) in enumerate(train_loader):
            # train the model on the batch
            # ...

            # log the training loss at the end of each epoch
            #####################if step == len(train_loader) - 1:
                # train_loss = loss  # compute the training loss
               ############################## # wandb.log({"train_loss": loss, "epoch": epoch})

        # evaluate the model on the validation set
        # ...

        # log the validation loss and accuracy at the end of each epoch
        # wandb.log({"val_loss": val_loss, "val_acc": val_acc, "epoch": epoch})

    ############################################################################
    
    print("Traing has been completed.")
    #img_list = glob.glob()
    img_features = []
    ds_c = ImagesDataset(args.image_folder, IMAGE_SIZE)
    img_loader = DataLoader(ds_c, batch_size=32, num_workers=2, shuffle=False)
    # torch.set_grad_enabled(False)
    resnet.eval()

    for img in img_loader:
      ################################################  print(img.size())  #32 3 512 512
        output = resnet(img)
       ###################################### print(f'print o/p image size {output.size()}')  # 32 512 1  1
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
    # Choose the number of clusters
    num_clusters = NUM_Clust
    print(f"The number of cluster is: {NUM_Clust}")
    
    
    # Run K-means algorithm
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(img_features)
    
    # Evaluate the clusters
    wcss = kmeans.inertia_
    print("Within-cluster sum of squares:", wcss)
    
    # Visualize the clusters
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    print(f'label is {labels}')
    print(f'centroid is {centroids}')  
    
    # # Print the cluster labels for each image
    # for i, label in enumerate(kmeans.labels_):
    #     print(f"{img_paths[i]}: {label}")
    
    
    #     #get cluster label and centroids
    #     labels = kmeans.labels_
    #     centroids = kmeans.cluster_centers_
    
    # Evaluate the algorithm using the silhouette score
    silhouette_avg = silhouette_score(img_features, labels)
    print("The average silhouette score is :", silhouette_avg)

    


# #     train_acc = 0
# #     acc_list = []
     
# #     # training
# #     for epoch in range(1, EPOCHS):
# #         print(f"Epoch: {EPOCHS} in progress...")
# #     #     # train models
# #         # train_acc= resnet.train(train_loader, EPOCHS)
# #         print(f'train_accuracy is : {train_acc} and train_loss is {train_loss}')
# # # with wandb.init().run:
# #   # Train model
# #   wandb.save('model.pt')

# ##### To Do ###########

##we need extract features using the trained network by byol

##k means clustering k = no of label (keep it a argument)

##then check the purity score or get the voting by the label available


