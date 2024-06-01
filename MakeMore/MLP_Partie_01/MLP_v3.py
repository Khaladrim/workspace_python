import torch
import torch.nn.functional as F

import random
import matplotlib.pyplot as plt # pour farie des figures

from tqdm import tqdm
# Version splittant le dataset en training, validation et test
# Aller à la ligne 149

# Fixer les graines pour la reproductibilité
seed = 2147483647
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Voir https://www.youtube.com/watch?v=TCH_1BHY58I&ab_channel=AndrejKarpathy
# Lire dans tous les mots
words = open('MakeMore/names.txt', 'r').read().splitlines()
wordsSize = len(words)

# Créer un dictionnaire pour les mots et mapping depuis/vers les entiers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

#print(itos)
block_size = 5 # longueur du contexte: combien de mot on prend pour prédire le suivant (padding avec des points si block_size > len(mot))

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

# Training split, dev/validatio, split, test split
# -> 80%,           10%,                10%
# Trainnig set: pour entrainer le modèle
# Dev/Validation set: pour ajuster les hyperparamètres (taille du réseau, de l'embedding etc)
# Test set: pour tester le modèle sur des données inconnues et avoir la performance finale

Xtr,Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
# embedding lookup table
# On utilise un vecter à 10D pour tirer partie du sur-dimensionnement du réseau à nb_neuronnes neuronnes
embDim = 50
C = torch.randn((27,embDim)) 
# Construction du Hidden Layer
# Weights
mLine = C.shape[1] * block_size
nb_neuronnes = 300
W1 = torch.randn((mLine, nb_neuronnes)) # 30 = 3 * 10 de emb (10D embeddings et on a 3 charactères);
# Biases
b1 = torch.randn(nb_neuronnes) 

# Création du layer final
W2 = torch.randn((nb_neuronnes, 27)) # 27 car 27 charactères possibles
b2 = torch.randn(27) 

parameters = [C, W1, b1, W2, b2] 
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

def forward(X):
    emb = C[X] # embedder les inputs selon mini-batch
    # Le vecteur redevient un (32, 3, 2) au lieu de sa taille complete
    h = torch.tanh(emb.view(-1, mLine) @ W1 + b1).to(device) # hidden layer + fonction activation
    logits = h @ W2 + b2 # logits

    return logits
    
def train(X, Y):
    epochs = 100000 
    miniBatchSize = 3000   
    for i in tqdm(range(epochs), desc = 'Training Epochs'):
        # mini-batch
        ix = torch.randint(0, X.shape[0], (miniBatchSize,)) 
        # Forward pass
        logits = forward(X[ix])
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

    print(loss.item()) # afficher la loss


print("Loss after training:")
train(Xtr, Ytr)
# On plot la perte à chaque itération
# "L'épaisseur" dans la courbe est du à l'utilisation de mini-batch créant une sortes de bruit
# A mesure que les paramètres augmentent, le bruit devient de plus en plus génant
# on peut vouloir augmenter la taille des mini-batch en conséquence
plt.plot(stepi, lossi)
plt.show()

# On peut maintenant tester le modèle sur le dev set
print("Loss of train set:")
logits = forward(Xtr)
loss = F.cross_entropy(logits, Ytr)
print(loss.item())

# On peut maintenant tester le modèle sur le dev set
print("Loss of dev set:")
logits = forward(Xdev)
loss = F.cross_entropy(logits, Ydev)
print(loss.item())

print("Loss of test set:")
logits = forward(Xte)
loss = F.cross_entropy(logits, Yte)
print(loss.item())
# On peut tester le modeles le dataset dev et observer la performance
# Si on test sur le dataset d'entrainement, on peut déterminer si on overfit (le modèle apprend par coeur)
# Il faut alors redimensionner le réseau en changer le nombre de ses paramètres (l'augmenter)
# Si la perte ne s'ameliore pas trop, on peut redimensionner l'embedding qui est en 2D au début, 
# Et est peut être trop peu pour notre réseau maintenant sur-dimensionné

# Plot l'embedding
# Sur un plan en 2D, on visualise les 27 charactères
# plt.figure(figsize=(10,10))
# Abscisses et ordonnées (issus du vecteur 2D de C)
# plt.scatter(C[:,0].data, C[:,1].data, s=200)
# for i in range(C.shape[0]):
    # On ecrit le charactere sur le point correspondant
#    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color="white")
# plt.grid('minor')
# plt.show()

# On peut observer que le modèle a clusteriser les lettres en les regroupant par similarité

# Sauvegarde du modèle
torch.save(parameters, 'char_rnn_model.pth')
