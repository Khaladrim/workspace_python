import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import matplotlib.pyplot as plt # pour farie des figures

from tqdm import tqdm

# Voir: https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=10&ab_channel=AndrejKarpathy

# Entrainer un modele sur du Shakespeare

batch_size = 32
block_size = 8
max_iters = 1
eval_interval = 10000
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
#---------------

torch.manual_seed(1337)

# Lecture du texte
with open('GPT/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#Extraire tous les charactères uniques
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Création d'un mapping entre les charactères et les entiers (tokenizer)
# C'est une manière de le faire, comme le sub-words vocabulaire (tiktoken utilisé par openAI)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s] # convertir un mot en une liste d'entiers
decode = lambda x: ''.join([itos[i] for i in x]) # convertir une liste d'entiers en un mot

# # Exemple avec tiktoken, communément utilisé
# import tiktoken
# # On recupere la config de GPT_2 (peut prendre du temps à s'executer)
# enc = tiktoken.get_encoding('gpt2')
# print(enc.n_vocab)
# print(enc.encode("hii there"))
# print(enc.decode(enc.encode("hii there")))

#Tokenization du texte de Shakespear
data = torch.tensor(encode(text), dtype=torch.long)

# Partager le dataset pour l'entrainement et la validation
n = int(0.9 * len(data)) # 90% utilisé pour l'entrainement
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    # Genérer un petit batch de data de l'entrée x et de la cible y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # choisir un index aléatoire
    x = torch.stack([data[i:i+block_size] for i in ix]) # les contextes
    y = torch.stack([data[i+1:i+1+block_size] for i in ix]) # les cibles
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    # Passage en mode evaluation
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # Passage en mode train
    model.train()
    return out

# Implementation d'un reseau neuronal simple, bigramme
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # chaque token lu directement de logits pour prédire le suivant de la table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # Embedding est ici une table de taille (vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd) # Pour chaque position dans le contexte, on retourne un vecteur de taille n_embd
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        # On ajoute les embeddings positionnels
        # Chaque token de 0 à T-1 est associé à un vecteur de taille n_embd
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device)) # (T, C)
        # x n'est pas seulement l'identité du token, mais aussi sa position dans le contexte
        x = tok_emb + pos_emb # (B, T, C)
        print(x.shape)
        logits = self.lm_head(x) # (B, T, vocab_size)
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
    
model = BigramLanguageModel()

m = model.to(device)

# Entrainement
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

# for iter in range(max_iters):
    
#     # Evaluer la perte sur le train/val set
#     if iter % eval_interval == 0:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
    
#     xb, yb = get_batch('train')

#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
    

# Generation depuis le model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=9)[0].tolist()))

