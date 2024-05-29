import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Voir: https://lightning.ai/lightning-ai/studios/statquest-long-short-term-memory-lstm-with-pytorch-lightning?view=public&section=all&tab=files&layout=column&path=cloudspaces%2F01henpavmdtqndyk17xpzjdbj6&y=3&x=0

# Lightning pour coder plus rapidement LSTM
import lightning as L
# Pour les parametre a utiliser dans lightning
from torch.utils.data import DataLoader, TensorDataset

# Paramètres globaux
# Fixer les graines pour la reproductibilité
seed = 42
torch.manual_seed(seed)
compagnieA = [0., 0.5, 0.25, 1.]
compagnieB = [1., 0.5, 0.25, 1.]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMbyHand(L.LightningModule):
    def __init__(self):
        super().__init__()
        # On initialise les poids via la loi normale, les biais à 0
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)
        ## These are the Weights and Biases in the first stage, which determines what percentage
        ## of the long-term memory the LSTM unit will remember.
        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        ## These are the Weights and Biases in the second stage, which determins the new
        ## potential long-term memory and what percentage will be remembered.
        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
        ## These are the Weights and Biases in the third stage, which determines the
        ## new short-term memory and what percentage will be sent to the output.
        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

    # Fontion mathématique du LSTM
    def lstm_unit(self, input_value, long_memory, short_memory):
        # Premier étage du LSTM: determiner le pourcentage de la mémoire à long terme à se souvenir
        long_remember_percent = torch.sigmoid(short_memory * self.wlr1 + 
                                              (input_value * self.wlr2) + 
                                              self.blr1)
        # Deuxième étage du LSTM: determine une nouvelle mémoire à long terme potentielle 
        # et le pourcentage à rajouter à la mémoire long-terme actuelle
        potential_remember_percent = torch.sigmoid(short_memory * self.wpr1 + 
                                                   (input_value * self.wpr2) + 
                                                   self.bpr1)
        potential_memory = torch.tanh(short_memory * self.wp1 + 
                                      (input_value * self.wp2) + 
                                      self.bp1)
        # Après les 2 étages, on met à jour la mémoire à long terme
        update_long_memory = ((long_memory * long_remember_percent) + (potential_memory * potential_remember_percent))

        # Troisième étage du LSTM: determine la nouvelle mémoire à court terme 
        # et le pourcentage à se souvenir et envoyer en sortie
        output_percent = torch.sigmoid(short_memory * self.wo1 + 
                                      (input_value * self.wo2) + 
                                      self.bo1)
        update_short_memory = torch.tanh(update_long_memory) * output_percent

        return ([update_long_memory, update_short_memory])


    # Fonction forward
    def forward(self, input):
        # Initialisation des mémoires à long et court terme
        long_memory = 0
        short_memory = 0
        # Initalisation des données à mémoriser
        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]

        # Envoyer de la données day1 et des mémoires initiales dans lstm_unit
        [long_memory, short_memory] = self.lstm_unit(day1, long_memory, short_memory)
        # Et ainsi de suite pour les jours suivants
        [long_memory, short_memory] = self.lstm_unit(day2, long_memory, short_memory)
        [long_memory, short_memory] = self.lstm_unit(day3, long_memory, short_memory)
        [long_memory, short_memory] = self.lstm_unit(day4, long_memory, short_memory)

        return short_memory
    
    # Fonction configurant Adam optimizer
    def configure_optimizers(self):
        return Adam(self.parameters())

    # Fonction calculant les perte et enregistrant les résultats
    # batch: données de l'une des compagnies
    # batch_idx: index du batch
    def training_step(self, batch, batch_idx):
        # Utilisation de forward pour obtenir les prédictions
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        # Calcul de la perte
        loss = (output_i - label_i)**2

        # On enregistre les résultats dans un log
        # Herité de lightning: créer un fichier lightning_logs
        self.log("train_loss", loss)

        # Log des prédictions selon la compagnie
        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)
        
        return loss
    
model = LSTMbyHand().to(device) 
print("\nComparaison de la valeur à observer et la valeur prédite")
print("Compagnie A: observer = 0, Prédite =",
      round(model(torch.tensor(compagnieA)).detach().item(), 2))
print("Compagnie B: observer = 1, Prédite =",
      round(model(torch.tensor(compagnieB)).detach().item(), 2))
# On peut voir que la prédiction est correcte pour la compagnie A, 
# mais incorrecte pour la compagnie B
# Il faut entrainer le modele

inputs = torch.tensor([compagnieA, compagnieB])
labels = torch.tensor([0., 1.]) # les valeurs à prédire pour chaque compagnie

# Combinaison des données et des labels dans un TensorDataset
dataset = TensorDataset(inputs, labels)
# Création d'un DataLoader pour itérer sur les données
# Dataset permet un acces facile à des batch de données
# Permet de shuffle (mélanger) l'accès aux données
# Permet de facile utiliser une portion de la base de données pour l'entrainement
dataloader = DataLoader(dataset) #, batch_size=1, shuffle=True)

# Création de l'entraiement du modèle
trainer = L.Trainer(max_epochs=5000)
#trainer.fit(model, train_dataloaders=dataloader)
print("#############################################")
print("\nComparaison de la valeur à observer et la valeur prédite après entrainement 01")
print("Compagnie A: observer = 0, Prédite =",
      round(model(torch.tensor(compagnieA)).detach().item(), 2))
print("Compagnie B: observer = 1, Prédite =",
      round(model(torch.tensor(compagnieB)).detach().item(), 2))


# Resultat meilleur pour la B mais mauvais pour la A
# On regarde les logs tensorboard via la commande:
# tensorboard --logdir=path/to/log
# ATTENTION A BIEN AVOIR LES DROIT D'ACCES AUX FICHIERS
# En analysant les graphs:
# - total_loss n'a pas convergé, donc il peut encore diminuer
# - out_0 monte à 0.5 puis diminue, il pourrait donc converger vers 0 comme voulu
# - out_1 monte à 0.6 mais ne semble pas encore converger, donc il peut continuer de monter vers 1
# => On peut continuer l'entrainement en augmentant le nombre d'epochs