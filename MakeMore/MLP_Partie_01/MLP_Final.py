import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

from tqdm import tqdm # barre de progression

class CharRNN:
    def __init__(self, block_size=5, embedding_dim=50, hidden_dim=300):
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Créer un dictionnaire pour les mots et mapping depuis/vers les entiers
        self.chars = list("abcdefghijklmnopqrstuvwxyz.")
        self.stoi = {s:i+1 for i,s in enumerate(self.chars)}
        self.stoi['.'] = 0
        self.itos = {i:s for s,i in self.stoi.items()}

        # Initialisation des paramètres
        self.C = torch.randn((27, embedding_dim))
        self.W1 = torch.randn((embedding_dim * block_size, hidden_dim))
        self.b1 = torch.randn(hidden_dim)
        self.W2 = torch.randn((hidden_dim, 27))
        self.b2 = torch.randn(27)
        
        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        for p in self.parameters:
            p.requires_grad = True
        sum_param = sum(p.numel() for p in self.parameters) # Nombre de paramètres du réseau
        print("Nb parametre: " + str(sum_param)) # 3481 pour 100 neuronnes

    def build_dataset(self, words):
        X, Y = [], []
        for w in words:
            context = [0] * self.block_size
            for ch in w + '.':
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y

    def forward(self, X):
        emb = self.C[X]
        h = torch.tanh(emb.view(-1, self.embedding_dim * self.block_size) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits

    def train(self, X, Y, epochs=100000, lr=0.1, lr_decay=0.01):
        stepi = []
        lossi = []
        miniBatchSize = 3000
        for i in tqdm(range(epochs), desc = 'Training Epochs'):
            ix = torch.randint(0, X.shape[0], (miniBatchSize,))
            logits = self.forward(X[ix])
            loss = F.cross_entropy(logits, Y[ix])
            for p in self.parameters:
                p.grad = None
            loss.backward()
            for p in self.parameters:
                p.data += -lr * p.grad
            if i >= epochs // 2:
                lr *= lr_decay
            stepi.append(i)
            lossi.append(loss.log10().item())
        
        plt.plot(stepi, lossi)
        plt.show()
        print(loss.item())

    def save(self, path):
        torch.save(self.parameters, path)

    def load(self, path):
        self.parameters = torch.load(path)
        self.C, self.W1, self.b1, self.W2, self.b2 = self.parameters

    def predict(self, context, n_chars=1):
        context_idx = [self.stoi[c] for c in context]
        context_idx = torch.tensor(context_idx).unsqueeze(0)
        predictions = []
        for _ in range(n_chars):
            logits = self.forward(context_idx)
            probs = F.softmax(logits, dim=1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = self.itos[next_idx]
            predictions.append(next_char)
            context_idx = torch.cat((context_idx[:,1:], torch.tensor([[next_idx]])), dim=1)
        return ''.join(predictions)

# Exemple d'utilisation
if __name__ == "__main__":
    # Fixer les graines pour la reproductibilité
    seed = 2147483647
    torch.manual_seed(seed)

    # Lire les mots
    words = open('MakeMore/names.txt', 'r').read().splitlines()
    random.seed(42)
    random.shuffle(words)

    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    # Instancier le modèle
    model = CharRNN()

    # Construire les datasets
    Xtr, Ytr = model.build_dataset(words[:n1])
    Xdev, Ydev = model.build_dataset(words[n1:n2])
    Xte, Yte = model.build_dataset(words[n2:])

    # Entraîner le modèle
    model.train(Xtr, Ytr)

    # Sauvegarder le modèle
    model.save('char_rnn_model_version_Final.pth')

    # Tester la prédiction
    context = 'eliz'
    next_char = model.predict(context, n_chars=5)
    print(f"Next character prediction for '{context}': {next_char}")
