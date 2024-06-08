import torch
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np
import os
import torch.optim as optim
from tqdm import tqdm
from datasets import load_dataset

# Fixer la graine pour la reproductibilité
seed = 2147483647
torch.manual_seed(seed)

# Charger le dataset
dataset = load_dataset('AlvaroVasquezAI/Animal_Image_Classification_Dataset')

# Créer des sous-ensembles d'entraînement, de validation et de test
def generate_subset(dataset=dataset):
    # Randomisation des datasets
    dataset = dataset.shuffle(seed=42)

    # Indices pour chaque ensemble
    indice_train = list(range(0, 2400))
    indice_val = list(range(2400, 2700))
    indice_test = list(range(2700, 3000))

    # Sélectionner les sous-ensembles
    dataset_train = dataset['train'].select(indice_train)
    dataset_val = dataset['train'].select(indice_val)
    dataset_test = dataset['train'].select(indice_test)

    return dataset_train, dataset_val, dataset_test

# Définir les transformations pour redimensionner les images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Fonction pour afficher une image
def show_image(tensor_image):
    np_image = tensor_image.numpy().transpose((1, 2, 0))
    plt.imshow(np_image)
    plt.axis('off')
    plt.show()

# Fonction pour traiter les images à partir du dataset
def _traiter_Image_From_Dataset(dataset):
    image_torch_list = []
    for i in range(len(dataset)):
        image = dataset[i]['image']
        image_np = np.array(image)
        image_torch = torch.from_numpy(image_np).float().unsqueeze(0)
        image_torch = image_torch.permute(0, 3, 1, 2)
        image_torch_list.append(image_torch)
    return torch.stack(image_torch_list)

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
    image_torch = transform(image).float().unsqueeze(0).permute(0, 1, 2, 3)
    show_image(image_torch[0])
    return image_torch

# Définition du modèle ConvNet
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(12675, 100)  # adapter pour supprimer les nombres magiques
        self.fc2 = nn.Linear(100, 3)
        self._initialize_weights()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
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

# Initialisation du modèle, de la perte et de l'optimiseur
model = ConvNet()

# Fonction d'entraînement du modèle
def train(isTrain=0):
    dataset_train, dataset_val, dataset_test = generate_subset()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    lossi, running_lossi, stepi = [], [], []
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
        model.load_state_dict(torch.load("CNN/model_v2.pth"))
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
            if isTrain == 0:
                optimizer.zero_grad()

            outputs = model(inputs[i])
            print(outputs)
            label = torch.tensor([data[i]['label']])
            loss = criterion(outputs, label)
            lossi.append(loss.log10().item())
            stepi.append(step)

            if isTrain == 0:
                loss.backward()
                optimizer.step()
            step += 1

            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data):.4f}")
        running_lossi.append(running_loss / len(data))

    torch.save(model.state_dict(), "CNN/model_DEBUG.pth")
    plt.plot(stepi, lossi)
    plt.title("Loss")
    plt.show()

train(isTrain=1)

# Fonction de prédiction
@torch.no_grad()
def predict(model):
    model.eval()
    ImageTest = Image.open("CNN/chien_test.jpg")
    image_torch = traiter_Image_From_JPG(ImageTest)

    output = model(image_torch)
    output_max = output.argmax(dim=1)
    print("outpout: ", output_max.item())

# Charger le modèle et prédire
# model.load_state_dict(torch.load("CNN/model_v2.pth"))
# predict(model)
