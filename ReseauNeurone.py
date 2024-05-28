import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as SGD

import matplotlib.pyplot as plt
import seaborn as sns
   

# Class d'entrainement du réseau de neurones
class BasicRN_train(nn.Module):
    # Initialisation des poids et des biais
    def __init__(self):
        super().__init__()
        #Voir video "The StatQuest Introduction to Pytoch" pour plus d'informations
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        # Mise à 0 du biais final et activation du gradient (true) 
        # pour l'optimiser lors de l'entrainement
        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True)
    
    
    # Fonction forward
    def forward(self, input):
        # Couche 1 haute
        input_to_top_relu = input * self.w00 + self.b00
        # Fonction d'activation
        top_relu_output = F.relu(input_to_top_relu)
        # Couche 2 haute
        scaled_top_relu_output = top_relu_output * self.w01

        # Couche 1 basse
        input_to_bottom_relu = input * self.w10 + self.b10
        # Fonction d'activation
        bottom_relu_output = F.relu(input_to_bottom_relu)
        # Couche 2 basse
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        # Couche finale: somme des deux couches hautes et basses + biais fianl
        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias
        # Fonction d'activation finale
        output = F.relu(input_to_final_relu)
        
        return output
    
def train(model, input, labels):
    # Création d'un objet optimiseur pour utiliser la descente de gradient stochastique
    #lr donne le pas. Si pas assez grand, l'entrainement sera lent. Si trop grand, on peut ne pas converger
    optimizer = SGD.SGD(model.parameters(), lr=0.1)
    print("Biais final avant entrainement: " + str(model.b00) + "\n")
    # Nombre d'epoch (combien de fois on passe sur l'ensemble des données d'entrainement)
    num_epoch = 100
    for epoch in range(num_epoch):
        total_loss = 0
        for iteration in range(len(input)):
            input_i = input[iteration]
            label_i = labels[iteration]
        
            output_i = model(input_i)
            # Calcul de la perte entre la sortie du modèle et la valeur attendue
            loss = (output_i - label_i)**2
            # Calcul de la dérivée de la perte par rapport à ce qu'on veut optimiser (le biais final)
            # loss.backward accumule les dérivées contenues dans output_i venant du modèle qui garde la trace des dérivées
            loss.backward()

            total_loss += float(loss)
        # Si la perte est assez faible, on arrête l'entrainement
        if(total_loss < 0.0001):
            print("Epoch: " + str(epoch) + " Loss: " + str(total_loss) + "\n")
            break
        # On continue l'optimisation sinon
        optimizer.step()
        # On remet les gradients à 0 pour ne pas les accumuler
        optimizer.zero_grad()
        #print("Epoch: " + str(epoch) + " Biais final: " + str(model.final_bias) + "\n")
    print("Biais final après entrainement: " + str(model.b00) + "\n")

# Création de 11 données d'entrée entre 0 et 1 inclus
input_doses = torch.linspace(start=0, end=1, steps=11)
print("Données d'entrée:")
print(input_doses)
print("##################################")
# Creation des données d'entrainement
input = torch.tensor([0., 0.5, 1])
labels = torch.tensor([0., 1, 0.])

# Création du modèle
print("Création du modèle")
#Utiliser le modèle ci-dessous pour voir l'impact du biais non optimisé
model_train = BasicRN_train()
print("##################################")
# Entrainement du modèle
print("Entrainement du modèle")
train(model_train, input, labels)
print("##################################")
# Utilisation du modèle
output_values = model_train(input_doses)



#Plot des valeurs de sortie
sns.set_theme(style='whitegrid')
sns.lineplot(x=input_doses, 
             y=output_values.detach(), #detach() pour enlever les gradients, utile surtout avec le modele BasicRN_train
             color='green', 
             linewidth=2,
             label='Valeurs de sortie')
plt.title("Sortie du modèle")
plt.xlabel("Efficacité du médicament")
plt.ylabel("Dose de médicament")
plt.show()