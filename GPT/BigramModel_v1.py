import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import matplotlib.pyplot as plt # pour farie des figures

from tqdm import tqdm

# Voir: https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=10&ab_channel=AndrejKarpathy

# Entrainer un modele sur du Shakespeare

# Lecture du texte
with open('GPT/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
#print("longueur du texte = ", len(text))
# print(text[:1000])

#Extraire tous les charactères uniques
chars = sorted(list(set(text)))
vocab_size = len(chars)
#print('Vocab size:', vocab_size)
#print(''.join(chars))

# Création d'un mapping entre les charactères et les entiers (tokenizer)
# C'est une manière de le faire, comme le sub-words vocabulaire (tiktoken utilisé par openAI)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s] # convertir un mot en une liste d'entiers
decode = lambda x: ''.join([itos[i] for i in x]) # convertir une liste d'entiers en un mot

#print(encode('hello'))
#print(decode(encode('hello')))

# # Exemple avec tiktoken, communément utilisé
# import tiktoken
# # On recupere la config de GPT_2 (peut prendre du temps à s'executer)
# enc = tiktoken.get_encoding('gpt2')
# print(enc.n_vocab)
# print(enc.encode("hii there"))
# print(enc.decode(enc.encode("hii there")))

#Tokenization du texte de Shakespear
data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:1000])

# Partager le dataset pour l'entrainement et la validation
n = int(0.9 * len(data)) # 90% utilisé pour l'entrainement
train_data, val_data = data[:n], data[n:]

# Pour entrainer le Transformer, on ne va pas utiliser tout le texte, mais des morceaux de texte (chunks)
# Utiliser tout le texte en une fois demanderait une trop grande computabilité
# Et aussi pour que le transformer apprenne à prédire le mot suivant en fonction d'un contexte plus ou moins long

# block_size = 8
# # Rappel sur le fonctionnement de la prédiction selon le contexte
# x = train_data[:block_size]
# y = train_data[1:block_size + 1] # décalé de 1 par rapport à x car on veut prédire le mot suivant
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"Contexte: {context} -----> Target: {target}")

torch.manual_seed(1337)
batch_size = 4 # combien de contextes independants on va traiter en même temps
block_size = 8 # longueur max du contexte

def get_batch(split):
    # Genérer un petit batch de data de l'entrée x et de la cible y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # choisir un index aléatoire
    x = torch.stack([data[i:i+block_size] for i in ix]) # les contextes
    y = torch.stack([data[i+1:i+1+block_size] for i in ix]) # les cibles

    return x, y
xb, yb = get_batch('train')
# print('input:')
# print(xb)
# print('target:')
# print(yb)
# print('-------')

# for b in range(batch_size): # batch dimension
#     for t in range(block_size): # time dimension
#         context = xb[b, :t + 1]
#         target = yb[b, t]
#         print(f"Contexte: {context} -----> Target: {target}")

# Implementation d'un reseau neuronal simple, bigramme
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # chaque token lu directement de logits pour prédire le suivant de la table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # Embedding est ici une table de taille (vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        logits = self.token_embedding_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            # La fonction cross_entropy prend en entrée un tenseur de taille (B, C)
            # Il faut reshape logits
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            # idem pour targets (B, T) -> (B*T)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx est un array (B, T) d'index du context présent
        for _ in range(max_new_tokens):
            # Prediction
            logits, loss = self(idx) # self tout seul revient à appeler la méthode forward
            # Focus sur le dernier token
            logits = logits[:, -1, :] # Devient (B, C)
            # Softmax
            probs = F.softmax(logits, dim=-1) #(B, C)
            # Echantillonner un nouveau token
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Ajouter le nouveau token à la fin du context
            idx = torch.cat([idx, idx_next], dim=-1) # (B, T+1)
        return idx
    
m = BigramLanguageModel(vocab_size)
out, loss = m(xb, yb)
# print(out.shape) # Sans le reshape (B, T, C) soit (batch_size, block_size, vocab_size)
#                 # Aves le reshapt (B*T, C) soit (batch_size*block_size, vocab_size
print("Loss before trainning: ", loss.item())
idx = torch.zeros((1, 1), dtype=torch.long) # (B, T) = (1, 1), pour démarrer le modele
# print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))
# [0] car generate fonctionne sur un batch de contextes, mais ici on en a qu'un seul

# Entrainement

optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

batch_size = 32 # Plus grand que 4 précédement
for steps in tqdm(range(10000), position=0, leave=True):
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print("Loss after trainning: ", loss.item())

#print(decode(m.generate(idx, max_new_tokens=1000)[0].tolist()))
