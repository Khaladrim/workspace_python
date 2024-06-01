import torch
import torch.nn.functional as F


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

g = torch.Generator().manual_seed(42) # for reproducibility

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


def load(self, path):
    self.parameters = torch.load(path)
    self.C, self.W1, self.b1, self.W2, self.b2 = self.parameters

def predict(context, n_chars=1):
    context_idx = [stoi[c] for c in context]
    context_idx = torch.tensor(context_idx).unsqueeze(0)
    context = [context_idx]
    predictions = []
    with torch.no_grad():  # On désactive le calcul des gradients pour la prédiction
        for _ in range(n_chars):
            logits = model(context_idx)
            probs = F.softmax(logits, dim=1)
            next_idx = torch.multinomial(probs, num_samples=1)
            next_char = itos[next_idx.item()]
            predictions.append(next_char)
            context_idx = torch.cat((context_idx[:, 1:], next_idx), dim=1)
            if next_idx.item() == 0:
               break
    return ''.join(predictions)
    
def predict_V2():
    g = torch.Generator().manual_seed(2147483647 + 10)

    for _ in range(100):
        out = []
        context = [0] * block_size # initialize with all ...
        while True:

            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break

        print(''.join(itos[i] for i in out))

if __name__ == "__main__":
    # Instancier le modèle
    block_size = 8
    n_embd = 24 # the dimensionality of the character embedding vectors
    n_hidden = 128 # de 300 à 68 pour garder le mettre nombre de parametre qui sont bcp plus efficace avec l'architecture de conv par binome
    charConv = block_size // 4
    model = Sequential ([
        Embedding(vocab_size, n_embd),
        FlattenConsecutive(charConv), Linear(n_embd * charConv, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(charConv), Linear(n_hidden * charConv, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(charConv), Linear(n_hidden * charConv, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),    
        Linear(n_hidden, vocab_size),
    ])
    # Charger le modèle
    checkpoints = torch.load('MakeMore/char_wawenet.pth')
    parameters = model.parameters()
    for param, saved_param in zip(parameters, checkpoints):
        param.data = saved_param.data
    
    for layer in model.layers:
       layer.training = False
    # Prédire le prochain caractère
    # context = '....flor'
    # next_char = predict(context, n_chars=3)
    # print(f"Next character prediction for '{context}': {next_char}")

    # sample from the model
    print("Sample from the model:")
    predict_V2()

