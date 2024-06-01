import torch
import torch.nn.functional as F

import random
import matplotlib.pyplot as plt # pour farie des figures

from tqdm import tqdm

# Version chergeant à améliorer les conditions initiales du modèle
# En printant loss, on voit la première itération à 27, et la 2e à 4
# L'initialisation du modèle n'est pas bonne
# Notre erreur vient d'une initialisation actuellement alétoire des paramètres, avec une dissymétrie 
# dans les éléments matricielles, ce qui gaspille les premières itérations de l'entrainement

# 2e point important: la fonction d'activation et la backpropagation
# Avec tanh, les valeurs de h prés activation (qui sont gaussienne) sont ramenées entre [-1; 1] et sature sur ces bords
# Ceci pose au probleme pour la backpropagation, car pour aller à travers tanh, on multiplie le gradient par (1 - t**2), or si t vaut 1 ou -1, on efface le gradient
# Conséquence: les neuronnes concernés n'apprennent pas
#Pour quantifier l'impact:
    # plt.figure(figsize=(20, 10))
    # plt.imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest')
    # plt.title('Saturated neurons (in white)')
    # plt.show()
    # plt.hist(h.view(-1).tolist(), 50)
    # plt.title('Distribution of hidden layer values')
    # plt.show()
    # plt.hist(hpreact.view(-1).tolist(), 50)
    # plt.title('Distribution of hidden layer pre-activation values')
    # plt.show()
# Les carrées blancs sont les valeurs saturées
# Meme probleme avec Sigmoid et ReLU qui satures à -1, 1 et 0 respectivement
# Il faut donc manipuler W1 et b1 pour éviter la saturation




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
block_size = 3 # longueur du contexte: combien de mot on prend pour prédire le suivant (padding avec des points si block_size > len(mot))

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


# Initialisation des paramètres
# On normalise en multipliannt tous les parametres par une valeur proche de 0
# On peut faire au fur et a mesure en lançant l'entrainement sur 1 itération et vérifiant que la perte initiale diminue
# Ne pas mettre exactement 0 pour les weight, sinon le réseau ne pourra pas apprendre

g = torch.Generator().manual_seed(seed)
# embedding lookup table
# On utilise un vecter à 10D pour tirer partie du sur-dimensionnement du réseau à nb_neuronnes neuronnes
# On peut le faire à la main au début, mais on peut aussi se référer à la doc toch.nn.init:
# std = gain / sqrt(fan_mode). Pour tanh, gain = 5/3 et fan_mode = 1ere dimension
embDim = 10
C = torch.randn((vocab_size,embDim), generator=g) 
# Construction du Hidden Layer
# Weights
mLine = C.shape[1] * block_size
nb_neuronnes = 200
W1 = torch.randn((mLine, nb_neuronnes), generator=g) * ((5 / 3) / (mLine ** 0.5)) # manuellement: 0.2
# Biases
# Du fait d'utiliser la batch normalisation, les biais ne sont pas nécessaires
# Car supprimer par la moyenne dans le calcul de normalisation
#b1 = torch.randn(nb_neuronnes, generator=g) * 0.01

# Création du layer final
W2 = torch.randn((nb_neuronnes, vocab_size), generator=g) * 0.01 # 27 car 27 charactères possibles
b2 = torch.randn(vocab_size, generator=g) * 0

bngain = torch.ones(1, nb_neuronnes) # gamma
bnbias = torch.zeros(1, nb_neuronnes) # beta

# Init a une gaussienne de moyenne 0 et variance 1
bnmean_running = torch.zeros(1, nb_neuronnes)
bnstd_running = torch.ones(1, nb_neuronnes)

parameters = [C, W1, W2, b2, bngain, bnbias] 
sum_param = sum(p.numel() for p in parameters) # Nombre de paramètres du réseau
print("Nb parametre: " + str(sum_param)) # 3481 pour 100 neuronnes

# Entrainement
for p in parameters:
    p.requires_grad = True # On active la descente de gradient pour les paramètres

# Quand on a de grandes tailles de données, les calcules sont longs
# On peut alors faire des mini-batch pour optimiser les calculs, mais perte de qualité sur le gradient

lre = torch.linspace(-3, 0, 1000)
lrs = 10 ** lre

lri = []  # learning rate used
lossi = [] # loss corresponding to lri
stepi =[]

def forward(X, training=True):
    emb = C[X] # embedder les inputs selon mini-batch
    hpreact = emb.view(-1, mLine) @ W1 # + b1; inutilse car batch normalisation (supprimé par bmean ensuite)
    # Batch normalisation 
    #----------------------------------------
    # Chaque neuronne sera normalisé selon la gaussienne aux autres neuronnes du batch
    # A voir group/layer normalisation qui sont plus récent et moins chiant
    bnmean = hpreact.mean(dim=1, keepdim=True)
    bnstd = hpreact.std(dim=1, keepdim=True)
    hpreact = (hpreact - bnmean) / bnstd
    # Mais on veut une belle gaussienne que pour l'initialisation. Le modèle ne doit pas être forcé à apprendre une gaussienne
    # On ajoute donc un paramètre gamma (gain) et beta (biais) pour que le modèle puisse fluctuer autour de la gaussienne
    # On peut ajouter un epsilon à bnstd pour éviter la division par 0
    hpreact = hpreact * bngain + bnbias

    if(training):
        with torch.no_grad():
            # Moyenne mobile de la normalisation
            bnmean_running * 0.999 + 0.001 * bnmean
            bnstd_running * 0.999 + 0.001 * bnstd
    
    # Non linéarité
    #----------------------------------------
    h = torch.tanh(hpreact).to(device) # hidden layer + fonction activation 

    logits = h @ W2 + b2 # logits
    return logits, bnmean, bnstd
    
def train(X, Y):
    epochs = 20000 
    miniBatchSize = 32   
    for i in range(epochs):
        # mini-batch
        ix = torch.randint(0, X.shape[0], (miniBatchSize,)) 
        # Forward pass
        logits, bnmeani, bnstdi = forward(X[ix], True)
        loss = F.cross_entropy(logits, Y[ix]) # loss
        # Backward pass
        for p in parameters:
            p.grad = None # reset les gradients
        loss.backward() 
        # Update
        # lr = lrs[i]
        lr = 10 ** -1 if i < epochs // 2 else 10 ** -2 # learning rate optimal
        for p in parameters:
            p.data += -lr * p.grad # learning rate de 0.1
            # 46min de la vidéo pour plus de détails sur le learning rate
        # Track stats
        stepi.append(i)
        lossi.append(loss.log10().item())
        if i % 10000 == 0:
            print(f'{i:7d}/{epochs}: {loss.item():4f}')
        
    print(loss.item()) # afficher la loss

@torch.no_grad()
def split_loss(split):
    x,y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    } [split]
    emb = C[x]
    embcat = emb.view(-1, mLine)
    hpreact = embcat @ W1
    hpreact = (hpreact - bnmean_running) / bnstd_running
    hpreact = hpreact * bngain + bnbias
    h = torch.tanh(hpreact).to(device)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(f'{split} loss: {loss.item()}')



# print("Loss after training:")
train(Xtr, Ytr)
# plt.plot(stepi, lossi)
# plt.show()

# # On peut maintenant tester le modèle sur le dev set
# print("Loss of train set:")
# logits = forward(Xtr)
# loss = F.cross_entropy(logits, Ytr)
# print(loss.item())

# # On peut maintenant tester le modèle sur le dev set
# print("Loss of dev set:")
# logits = forward(Xdev)
# loss = F.cross_entropy(logits, Ydev)
# print(loss.item())

# print("Loss of test set:")
# logits = forward(Xte)
# loss = F.cross_entropy(logits, Yte)
# print(loss.item())

# Sauvegarde du modèle
torch.save(parameters, 'MakeMore/char_rnn_model_Revisited.pth')

# Calibration du batch norm a la fin de l'entrainement
# Les valeurs bnmean_running et beanstd_running sont en gros equivalente à bnmean et bnstd
# On n'a pas besoin de faire un deuxieme etage juste pour ça, mais c'est un exemple ci-dessous
# with torch.no_grad():
#     # Envoie du training set
#     emb = C[Xtr]
#     hpreact = emb.view(-1, mLine) @ W1 + b1
#     #Calcul de la moyenne et variance
#     bnmean = hpreact.mean(dim=1, keepdim=True)
#     bnstd = hpreact.std(dim=1, keepdim=True)

split_loss('train')
split_loss('val')
