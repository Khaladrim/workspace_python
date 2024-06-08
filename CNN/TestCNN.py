import torch
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np
import os
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset
from datasets import load_from_disk
from tqdm import tqdm

seed = 2147483647
torch.manual_seed(seed)

# Load the dataset
dataset = load_dataset('AlvaroVasquezAI/Animal_Image_Classification_Dataset')

# Création des datasets d'entraînement, de validation et de test
def generate_subset(dataset=dataset):
    # Définir les indices pour chaque tier du dataset
    indices_tier1 = list(range(0, 1000))
    indices_tier2 = list(range(1000, 2000))
    indices_tier3 = list(range(2000, 3000))

    # Vérifiez si le dataset filtré existe déjà sur le disque
    # try:
    #     dataset_cat = load_from_disk("CNN/dataset_cat")
    #     print("Dataset cat loaded")
    # except FileNotFoundError:
    #     # Si le dataset filtré n'existe pas encore, filtrez le dataset et sauvegardez-le
    #     dataset_cat = dataset['train'].select(indices_tier1)
    #     dataset_cat.save_to_disk("CNN/dataset_cat")
        
    # try:
    #     dataset_dog = load_from_disk("CNN/dataset_dog")
    #     print("Dataset dog loaded")
    # except FileNotFoundError:
    #     dataset_dog = dataset['train'].select(indices_tier2)
    #     dataset_dog.save_to_disk("CNN/dataset_dog")

    # try:
    #     dataset_snake = load_from_disk("CNN/dataset_snake")
    #     print("Dataset snake loaded")
    # except FileNotFoundError:
    #     dataset_snake = dataset['train'].select(indices_tier3)
    #     dataset_snake.save_to_disk("CNN/dataset_snake")

    # Randomisation des datasets
    dataset = dataset.shuffle(seed=42)

    # Création des datasets d'entraînement, de validation et de test
    indice_train = list(range(0, 2400))
    indice_val = list(range(2400, 2700))
    indice_test = list(range(2700, 3000))

    dataset_train = dataset['train'].select(indice_train)
    dataset_val = dataset['train'].select(indice_val)
    dataset_test = dataset['train'].select(indice_test)

    return dataset_train, dataset_val, dataset_test

# Définir la transformation pour redimensionner les données
# A appeler pour les images non issues du dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Fonction pour afficher l'image
def show_image(tensor_image):
    # Convertir le tenseur en numpy array et transposer les dimensions
    np_image = tensor_image.numpy().transpose((1, 2, 0)) #(High, Width, Channel)
    
    # Image non-normalisée
    plt.imshow(np_image)
    plt.axis('off')  # Optionnel, pour enlever les axes
    plt.show()

def _traiter_Image_From_Dataset(dataset):
    image_torch_list = []
    for i in range(len(dataset)):
        image = dataset[i]['image']
        # Convertion de l'image en tensor
        image_np= np.array(image)
        image_torch = torch.from_numpy(image_np)
        # Convertion en tensor de float et ajout d'une dimension
        image_torch = image_torch.float().unsqueeze(0)
        # Permutation des axes pour avoir le bon format
        image_torch = image_torch.permute(0, 3, 1, 2)
        image_torch_list.append(image_torch)
    image_torched = torch.stack(image_torch_list)

    return image_torched

def traiter_Image_From_Dataset(dataset, path):
    if os.path.exists(path):
        image_torched = torch.load(path)
        print("Image torched loaded")
    else:
        image_torched = _traiter_Image_From_Dataset(dataset)
        torch.save(image_torched, path)
        print("Image torched saved")

    return image_torched

def traiter_Image_From_JPG(image):
    # Redimensionnement de l'image et conversion en tensor
    image_torch = transform(image)

    # Convertion en tensor de float et ajout d'une dimension
    image_torch = image_torch.float().unsqueeze(0)

    # Permutation des axes pour avoir le bon format
    image_torch = image_torch.permute(0, 1, 2, 3)
    show_image(image_torch[0])

    return image_torch


# Create a ConvNet
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(12675, 100) # adapter pour supprimer les nombres magiques
        self.fc2 = nn.Linear(100, 3)
        self._initialize_weights() # todo: résultats initaux très mauvais avec cette initialisation


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        #print("debug :: forward :: out.size: ", out.size())
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


#print("Nombre de paramètres du modèle: ", ConvNet().count_parameters())
# Initialisation du modèle, de la perte et de l'optimiseur
model = ConvNet()

# Entraînement du modèle
def train(isTrain = 0):
    dataset_train, dataset_val, dataset_test = generate_subset()
    # "CNN/image_torched" -> path to save/load the torched image from train dataset
    # "CNN/image_val_torch" -> path to save/load the torched image from validation dataset
    for param in model.parameters():
        param.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    lossi = []
    running_lossi = []
    stepi = []
    step = 0
    if isTrain == 0:
        inputs = traiter_Image_From_Dataset(dataset_train, "CNN/image_torched")
        data = dataset_train
        model.train()
        num_epochs = 1
        print("MODE :: Entrainement\n")

    elif isTrain == 1:
        inputs = traiter_Image_From_Dataset(dataset_val, "CNN/image_val_torch")
        data = dataset_val
        num_epochs = 1
        model.load_state_dict(torch.load("CNN/model.pth"))
        model.eval()
        print("MODE :: Validation\n")

    elif isTrain == 2:
        inputs = traiter_Image_From_Dataset(dataset_test, "CNN/image_test_torch")
        data = dataset_test
        num_epochs = 1
        model.load_state_dict(torch.load("CNN/model_v2.pth"))
        model.eval()
        print("MODE :: Test\n")

    for epoch in range(num_epochs):
        running_loss = 0.0
        if epoch == 5 and isTrain == 0:
            optimizer = optim.Adam(model.parameters(), lr=0.01)
        for i in tqdm(range(len(data))):
            # Remettre les gradients à zéro
            if isTrain == 0:
                optimizer.zero_grad()

            # Propagation avant
            outputs = model(inputs[i])
            label = torch.tensor([data[i]['label']])
            loss = criterion(outputs, label)
            lossi.append(loss.log10().item())
            stepi.append(step)
            # Propagation arrière
            if isTrain == 0:
                loss.backward()
                optimizer.step()
            step += 1
            # Impression des statistiques de perte
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data):.4f}")
        running_lossi.append(running_loss / len(data))
    torch.save(model.state_dict(), "CNN/model.pth")
    #print("lossi: ", lossi)
    plt.plot(stepi,lossi)
    plt.title("Loss")
    plt.show()
    # plt.plot(range(6), running_lossi)
    # plt.title("Running Loss")
    # plt.show()
    

train(isTrain=1)

@torch.no_grad()
def predict(model):
    # Prediction
    model.eval()
    # Ouvrir l'image
    ImageTest = Image.open("CNN/chien_test.jpg")

    # Traiter l'image
    image_torch = traiter_Image_From_JPG(ImageTest)


    #show_image(image_torch[0])
    # Predire
    output = model(image_torch)
    #print("outpout: ", output.argmax(dim=1).item())
    print("outpout: ", output)



    # Prédiction
    output = model(image_torch)
    output_max = output.argmax(dim=1)
    print("outpout: ", output_max.item())

# model.load_state_dict(torch.load("CNN/model_v2.pth"))
# predict(model)

# x = dataset_cat[999]['image']

# # Supposons que 'x' est votre image
# # Si votre image est un tensor PyTorch, vous devez la convertir en un tableau numpy
# if isinstance(x, Image.Image):
#     x = np.array(x)
# # Si votre image a 3 canaux, vous devez réorganiser les axes pour que les canaux soient le dernier axe
# if x.shape[0] == 3:
#     x = np.transpose(x, (1, 2, 0))

# plt.imshow(x)
# plt.show()