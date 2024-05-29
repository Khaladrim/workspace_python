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

class LightningLSTM(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=1)
    def forward(self, input):
        #transposé ligne vers colonne
        input_trans = input.view(len(input), 1)
        lstm_out, temp = self.lstm(input_trans)

        # LSTM a en mémoire toutes les sorties des jour
        # On ne s'interesse qu'à la derniere, donc lstm_out[-1]
        prediction = lstm_out[-1]
        return prediction
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)
    
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
    
model = LightningLSTM()

print("\nComparaison de la valeur à observer et la valeur prédite")
print("Compagnie A: observer = 0, Prédite =",
      round(model(torch.tensor(compagnieA)).detach().item(), 2))
print("Compagnie B: observer = 1, Prédite =",
      round(model(torch.tensor(compagnieB)).detach().item(), 2))

inputs = torch.tensor([compagnieA, compagnieB])
labels = torch.tensor([0., 1.]) # les valeurs à prédire pour chaque compagnie

# Combinaison des données et des labels dans un TensorDataset
dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset) #, batch_size=1, shuffle=True)

# Comme on a mis un pas plus grand (0.1) on reduit le nombre d'epoch
# Par defaut, log_every_n_steps=50, on le met à 2 pour voir plus de logs
trainer = L.Trainer(max_epochs=300, log_every_n_steps=2)

trainer.fit(model, train_dataloaders=dataloader)
print("#############################################")
print("\nComparaison de la valeur à observer et la valeur prédite après entrainement 01")
print("Compagnie A: observer = 0, Prédite =",
      round(model(torch.tensor(compagnieA)).detach().item(), 2))
print("Compagnie B: observer = 1, Prédite =",
      round(model(torch.tensor(compagnieB)).detach().item(), 2))
