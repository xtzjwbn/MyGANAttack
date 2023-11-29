import numpy as np
import torch


from models.models import netClassificationMLP,AutoEncoder,AdvGenerator,AdvDiscriminator

from pipeline.Pipeline import Pipeline

pipeline = Pipeline("../Data/German.json", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

random_seed = 222
pipeline.SplitTrainAndTest(0.1, 222)

#############################################
######### Classifier Phase  (cf)#############
#############################################
cf_lr = 2e-4
cf_loss = torch.nn.CrossEntropyLoss()
cf_step_size = 100
cf_step_gamma = 0.5
pipeline.SetClassifier(netClassificationMLP,learning_rate = cf_lr, loss = cf_loss, step_size = cf_step_size, gamma = cf_step_gamma)

cf_batch_size = 64
cf_epochs = 400
pipeline.ClassifierFit(batch_size = cf_batch_size, epochs = cf_epochs)
#############################################
#############################################
#############################################



#############################################
######### Encoder Phase  (ec) ###############
#############################################
ec_lr = 5e-3
ec_dim = 64
ec_loss = torch.nn.MSELoss()
ec_step_size = 400
ec_step_gamma = 0.6
pipeline.SetEncoder(AutoEncoder, encoder_dim = ec_dim, learning_rate = ec_lr, loss = ec_loss, step_size = ec_step_size, gamma = ec_step_gamma)

ec_batch_size = 128
ec_epochs = 2000
pipeline.EncoderFit(batch_size = ec_batch_size, epochs = ec_epochs)
#############################################
#############################################
#############################################



#############################################
############# GAN Phase  (gan) ##############
#############################################
lrG = 5e-5
lrD = 5e-5
pipeline.SetGAN(AdvGenerator,lrG,AdvDiscriminator,lrD)

K = 1
eps = 1
alpha_adv = 10
alpha_norm = 1
batch_size = 100
epochs = 400
pipeline.GANFit(K = K, eps = eps, alpha_adv = alpha_adv, alpha_norm = alpha_norm, batch_size = batch_size, epochs = epochs)
#############################################
#############################################
#############################################



#############################################
######### Comparison Phase  (cp) ############
#############################################

comparison = pipeline.GetComparison()
comparison.AddAttackModel("FGSM",eps=0.1)
comparison.AddAttackModel("FGSM",eps=0.3)
comparison.AddAttackModel("Deepfool")
comparison.AddAttackModel("Greedy",K=2)
comparison.AddAttackModel("MyGAN",K=2)
pipeline.ShowComparison()

#############################################
#############################################
#############################################

pipeline.Serialize("../models/German")

#############################################
######### Data Augmentation Phase  (da) #####
#############################################

pipeline.DefenseComparison()

#############################################
#############################################
#############################################