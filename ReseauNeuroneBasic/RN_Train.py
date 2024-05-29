import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as SGD

import matplotlib.pyplot as plt
import seaborn as sns

# Paramètres globaux
# Fixer les graines pour la reproductibilité
seed = 42
torch.manual_seed(seed)
# Fixer les paramètres d'entraînement
trainable = False
nb_epoch = 100000

# Class d'entrainement du réseau de neurones
class BasicRN_train(nn.Module):
    # Initialisation des poids et des biais
    def __init__(self):
        super(BasicRN_train, self).__init__()
        self.hidden_layer = nn.Linear(1, 2)
        self.output_layer = nn.Linear(2, 1)
        self.additional_bias = nn.Parameter(torch.zeros(2))  # Biais additionnel

    # Fonction forward
    def forward(self, x):
        x = self.hidden_layer(x)
        x = x + self.additional_bias
        x = F.tanh(x)
        x = self.output_layer(x)
        return x
# Fonction pour afficher les paramètres du réseau
def print_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name}: {param.data}')
   
net = BasicRN_train()
print(net)

def trainModel():
    if(trainable == False):
        return
    
    # Définition d'un exemple de fonction de perte et d'un optimiseur
    criterion = nn.MSELoss()  # Erreur quadratique moyenne
    optimizer = SGD.SGD(net.parameters(), lr=0.5)  # Descente de gradient stochastique

    # Exemple de données d'entrée et de sortie
    inputs = torch.tensor([[0.], [0.3], [0.5], [0.8], [1.]])
    targets = torch.tensor([[0.], [0.], [1.0], [0.], [0.0]])

    # Boucle d'entraînement
    for epoch in range(nb_epoch):  # Nombre d'époques
        # Remise à zéro des gradients
        optimizer.zero_grad()
        # Passage en avant
        outputs = net(inputs)
        # Calcul de la perte
        loss = criterion(outputs, targets)
        # Passage en arrière (backpropagation)
        loss.backward()
        # Mise à jour des poids
        optimizer.step()
        
        if (epoch+1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{nb_epoch}], Loss: {loss.item():.4f}')
            #print_parameters(net)
        if (loss <= 0.0001):
            print(f'BREAK :: Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')
            saveModel()
            break
def saveModel():
    # Sauvegarder le modèle
    torch.save(net.state_dict(), 'modele_RN_01_Dose.pth')

trainModel()
