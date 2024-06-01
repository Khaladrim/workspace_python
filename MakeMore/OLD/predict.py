import torch
import torch.nn.functional as F

# todo: faire une v2 avec une class du modele pour ne pas avoir à réécrire les fonctions à chaque fois
# Recréer les variables et la structure du modèle
block_size = 5
C = torch.randn((27,10))
mLine = C.shape[1] * block_size
W1 = torch.randn((mLine, 200))
b1 = torch.randn(200)
W2 = torch.randn((200, 27))
b2 = torch.randn(27)
parameters = [C, W1, b1, W2, b2]

# Charger les paramètres du modèle
parameters = torch.load('char_rnn_model.pth')

# Fixer les graines pour la reproductibilité
seed = 2147483647
torch.manual_seed(seed)

# Définir les dictionnaires
chars = list("abcdefghijklmnopqrstuvwxyz.")
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# Fonction de forward pass
def forward(X):
    emb = C[X] # embedder les inputs
    h = torch.tanh(emb.view(-1, mLine) @ W1 + b1) # hidden layer + fonction d'activation
    logits = h @ W2 + b2 # logits
    return logits

# Fonction pour prédire le prochain caractère
def predict(context, n_chars=1):
    context_idx = [stoi[c] for c in context] # Convertir le contexte en indices
    context_idx = torch.tensor(context_idx).unsqueeze(0) # Ajouter dimension batch
    predictions = []
    for _ in range(n_chars):
        logits = forward(context_idx)
        probs = F.softmax(logits, dim=1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        next_char = itos[next_idx]
        predictions.append(next_char)
        context_idx = torch.cat((context_idx[:,1:], torch.tensor([[next_idx]])), dim=1) # Mettre à jour le contexte
    return ''.join(predictions)

# Exemple d'utilisation
context = 'eliza' # Contexte de 3 caractères
next_char = predict(context, n_chars=3)
print(f"Next character prediction for '{context}': {next_char}")
