import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # pour farie des figures

# Fixer les graines pour la reproductibilité
seed = 2147483647
torch.manual_seed(seed)

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

# Création du dataset
block_size = 3 # longueur du contexte: combien de mot on prend pour prédire le suivant (padding avec des points si block_size > len(mot))
X, Y = [], [] # X = Inputs, Y = labels pour chaque X
for w in words [:5]: # mis à chaque pour le moment (efficiency)
    context = [0] * block_size
    # On bloque sur les char de chaque mot
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        #print(''.join([itos[i] for i in context]), '----->', itos[ix])
        context = context[1:] + [ix] # décaler le contexte 
        # pour chaque mot, on print les 3 char de context et le char suivant
# embedding lookup table
C = torch.randn((27,2)) # 27 lignes et 2 colonnes. 
#Les 27 lignes correspondent aux 27 charactères possibles, et chaque ligne contient un vecteur de taille 2
#print(C)

# Pour embedder, on peut soit faire C(X), ou utiliser one_hot encoding, ça revient au meme
# F.one_hot(torch.tensor(5), num_classes=27).float() @ C # 5 est l'index du charactère à encoder pour l'exemple
# On utilise .float pour caster en float car one_hot est un int64, et C en float
# On aura un vecteur de taille 27 avec un 1 à l'index 5, le reste à 0
# print(F.one_hot(torch.tensor(5), num_classes=27))

# On utilise C[x] car c'est plus rapide.

emb = C[torch.tensor(X)] # embedder les inputs
#emb.size = 32, 3, 2 (32 exemples, 3 charactères par batch, 2 dimensions pour chaque charactère)
# Construction du Hidden Layer
# Weights
W1 = torch.randn((6, 100)) # 6 = 3 * 2 de emb (2D embeddings et on a 3 charactères); 100 = nb de neuronnes, à notre choix
# Biases
b1 = torch.randn(100) # 100 car 100 neuronnes

# Ensuite, on ne peut pas juste faire emb @ W1 + b1; car emb est un tensor 3D et W1 est en 2D
# Il va falloit concaténer pour que emb passe de (32, 3, 2) à (32, 6) pour pouvoir faire le produit matriciel
# emb_cat = torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]], 1) # dim=1 pour concaténer sur la 2ème dimension
# On donne une liste de tensor à concaténer
# emb[:, 0, :] = 32, 2, soit les veteurs 2D des 32 exemples du premier mot, etc
# print(emb_cat.shape)

# Mais, si on change le block_size (de 3 à 5 par exemple) on devra changer le code ci-dessus
# et ajouter emb[:, 3, :] et emb[:, 4, :] à la liste
# On utilise alors torch.unbind pour défaire le tensor en liste de tensor.
# La fonction retourne directement emb[:, 0, :], emb[:, 1, :], emb[:, 2, :] etc

# torch.cat(torch.unbind(emb, 1), 1).shape

# Voir à 25-26min de la vidéo pour plus de détails, mais on peut utiliser directement view en y definissant les dimensiosn voulues
# Pour ne pas hardcoder de taille, on peut utiliser -1 pour dire "tout le reste"
# Ou emb.shape[0]
h = emb.view(-1, 6) @ W1 + b1 # On a maintenant un tensor de taille 32, 100
#print(h.shape) # (32, 100)
# Cette façon est plus efficiante que de concatener car il n'y a pas de copie de mémoire

# Création du layer final
W2 = torch.randn((100, 27)) # 100 car 100 neuronnes en entrée, 27 car 27 charactères possibles
b2 = torch.randn(27) 

parameters = [C, W1, b1, W2, b2] 
sum_param = sum(p.numel() for p in parameters) # Nombre de paramètres du réseau
# print(sum_param) # 3481

logits = h @ W2 + b2 # logits = sorties du réseau
#print(logits.shape) # (32, 27)

# counts = logits.exp() # softmax
# prob = counts / counts.sum(1, keepdims=True) # normalisation pour avoir des probabilités
# print(probs[0].sum()) # doit être égal à 1 car somme des probabilités

# Maintenant, on doit traiter Y et prédire le charactère suivant
# On va boucler sur les probabilités et associer la proba au bon charactère, comme définit par Y (labels)

#torch.arrange(32) # tensor([0, 1, 2 ... 31])

#prob[torch.arange(32), Y] # On prend les probabilités associées aux labels Y, et avec les poitds/biais initiaux, on a des probabilités aléatoires (on espère 1 après entrainement)

# loss = -prob[torch.arange(32), Y].log().mean() # Cross Entropy Loss
# Fonction torch calculant la cross entropy loss plus efficacement
loss = F.cross_entropy(logits, torch.tensor(Y))

# Entrainement
for p in parameters:
    p.requires_grad = True # On active la descente de gradient pour les paramètres

for _ in range(1000):
    # Forward pass
    emb = C[torch.tensor(X)] # embedder les inputs
    h = emb.view(-1, 6) @ W1 + b1 # hidden layer
    logits = h @ W2 + b2 # logits
    loss = F.cross_entropy(logits, torch.tensor(Y)) # loss
    # Backward pass
    for p in parameters:
        p.grad = None # reset les gradients
    loss.backward() 
    #update
    for p in parameters:
        p.data += -0.1 * p.grad # learning rate de 0.1

print(loss.item()) # afficher la loss
# On arrive à avoir une faible loss car on overfit sur 32 exemples en ayant 3481 paramètres
# On n'obtient pas une losse proche de 0 car dans les exemples, le premier cas (...) doit prédire une lettre différente selon le nom
