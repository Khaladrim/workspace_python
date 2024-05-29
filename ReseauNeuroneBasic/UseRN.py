import torch
from RN_Train import BasicRN_train
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le modèle pré-entraîné
net = BasicRN_train()
net.load_state_dict(torch.load('modele_RN_01_Dose.pth'))
#net.eval()  # Mettre le modèle en mode évaluation

# Création de 11 données d'entrée entre 0 et 1 inclus
input_doses = torch.linspace(start=0, end=1., steps=11).unsqueeze(1)
output_values = net(input_doses).detach()

#Plot des valeurs de sortie
sns.set_theme(style='whitegrid')
sns.lineplot(x=input_doses.squeeze().numpy(), 
             y=output_values.squeeze().numpy(), #detach() pour enlever les gradients, utile surtout avec le modele BasicRN_train
             color='green', 
             linewidth=2,
             label='Valeurs de sortie')
plt.title("Sortie du modèle")
plt.xlabel("Efficacité du médicament")
plt.ylabel("Dose de médicament")
plt.show()