import torch
import torch.nn.functional as F

import random
import matplotlib.pyplot as plt # pour farie des figures

from tqdm import tqdm

# Fixer les graines pour la reproductibilité
seed = 2147483647
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Voir https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4
# Lire dans tous les mots
words = open('MakeMore/names.txt', 'r').read().splitlines()
wordsSize = len(words)
vocab_size = len(set(''.join(words))) + 1 # 27 charactères possibles + 1 pour le padding

# Créer un dictionnaire pour les mots et mapping depuis/vers les entiers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

#print(itos)
block_size = 8 # longueur du contexte: combien de mot on prend pour prédire le suivant (padding avec des points si block_size > len(mot))

# Création du dataset
def build_dataset(words):
    X, Y = [], [] # X = Inputs, Y = labels pour chaque X
    for w in words: # mis à chaque pour le moment (efficiency)
        context = [0] * block_size
        # On bloque sur les char de chaque mot
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            #print(''.join([itos[i] for i in context]), '----->', itos[ix])
            context = context[1:] + [ix] # décaler le contexte 
            # pour chaque mot, on print les 3 char de context et le char suivant

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y

# On melange les mots pour avoir un dataset plus varié
random.seed(42)
random.shuffle(words)

n1 = int(0.8 * wordsSize)
n2 = int(0.9 * wordsSize)

Xtr,Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# Print le contexte et la valeur cible correspondant
# for x,y in zip(Xtr[:20], Ytr[:20]):
#   print(''.join(itos[ix.item()] for ix in x), '-->', itos[y.item()])
#----------------------------------------------
class Linear:
  
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
    self.bias = torch.zeros(fan_out) if bias else None
  
  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])

#----------------------------------------------
class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):
    # calculate the forward pass
    if self.training:
        # La boucle if suivante permet de corriger le calcul de la moyenne et de la variance
        # En faisant les calcul sur la dimension 0 et 1, on obtient une moyenne et une variance par caractère
        if x.ndim == 2:
            dim = 0
        elif x.ndim == 3:
            dim = (0,1) # on execute sur la dimension 0 et 1
        xmean = x.mean(dim, keepdim=True) # batch mean
        xvar = x.var(dim, keepdim=True) # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]

#----------------------------------------------
class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []
#----------------------------------------------
class Embedding: # existe déjà dans pytorch mais avec plein de parametre
  
  def __init__(self, num_embeddings, embedding_dim):
    self.weight = torch.randn((num_embeddings, embedding_dim))
    
  def __call__(self, IX):
    self.out = self.weight[IX]
    return self.out
  
  def parameters(self):
    return [self.weight]

#----------------------------------------------
class FlattenConsecutive: # Flatten existe sur torch, mais notre version ici diverge un peu
    # Pour ameliorer l'efficacité du modele, au lieu de merger tous les caractères en une couche
    # On va les merger/convoluer 2 à 2 et a chaque couche successives.
    def __init__(self, n):
       self.n = n # combien de caracteres consecutifs on merge
    def __call__(self, x):
        B, T, C = x.shape # B = batch, T = binome de caractere, C = vecteur d'un caratere
        x = x.view(B, T // self.n, C*self.n)
        # Si T // self.n est de dimension 1, on squeeze pour retourner (B, C*self.n) au lieu de (B, 1, C*self.n)
        if x.shape[1] == 1:
           x = x.squeeze(1)
        self.out = x
        return self.out
    
    def parameters(self):
        return []
#----------------------------------------------
# Au lieu d'avoir une liste de couche, 
# on peut avoir une classe qui contient les couches 
# et qui les appelle successivement
class Sequential: # Existe aussi via Pytorch
  
  def __init__(self, layers):
    self.layers = layers
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    self.out = x
    return self.out
  
  def parameters(self):
    # get parameters of all layers and stretch them out into one list
    return [p for layer in self.layers for p in layer.parameters()]
#----------------------------------------------

n_embd = 24 # the dimensionality of the character embedding vectors
n_hidden = 128 # de 300 à 68 pour garder le mettre nombre de parametre qui sont bcp plus efficace avec l'architecture de conv par binome
g = torch.Generator().manual_seed(42) # for reproducibility

# Layers contient les couches successives du réseau de neurones
charConv = block_size // 4
model = Sequential ([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(charConv), Linear(n_embd * charConv, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(charConv), Linear(n_hidden * charConv, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(charConv), Linear(n_hidden * charConv, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),    
    Linear(n_hidden, vocab_size),
])

with torch.no_grad():
  # last layer: make less confident
  model.layers[-1].weight *= 0.1

parameters = model.parameters()
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True

# Optimisation
max_steps = 50000
batch_size = 32
lossi = []

for i in tqdm(range(max_steps), position=0, leave=True):
  
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
  
    # forward pass
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb) # loss function
  
    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
  
    # update
    lr = 0.1 if i < max_steps // 2 else 0.01 # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    # if i % 10000 == 0: # print every once in a while
    #     print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())

# plt.plot(lossi)
# plt.show()
# Pour éliminer le bruit et avoir un meilleur plot
plt.plot(torch.tensor(lossi).view(-1, 100).mean(1))
plt.show()

# Layer mis en mode évaluation
for layer in model.layers:
  layer.training = False

@torch.no_grad()
def split_loss(split):
    x,y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    } [split]

    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(f'{split} loss: {loss.item()}')

split_loss('train')
split_loss('val')
# Si les deux valeurs sont proches, c'est qu'on ne souffre pas d'overfitting
# On peut approfondir le réseau
# sample from the model
for _ in range(20):
    
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
        # forward pass the neural net
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        ix = torch.multinomial(probs, num_samples=1).item()
        # shift the context window and track the samples
        context = context[1:] + [ix]
        out.append(ix)
        # if we sample the special '.' token, break
        if ix == 0:
            break
    
    #print(''.join(itos[i] for i in out)) # decode and print the generated word

# Sauvegarde du modèle
torch.save(parameters, 'MakeMore/MLP_Partie_03/char_wawenet.pth')