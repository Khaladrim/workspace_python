import torch
import torch.nn.functional as F

class CharRNN:
    def __init__(self, block_size=5, embedding_dim=50, hidden_dim=300):
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.chars = list("abcdefghijklmnopqrstuvwxyz.")
        self.stoi = {s:i+1 for i,s in enumerate(self.chars)}
        self.stoi['.'] = 0
        self.itos = {i:s for s,i in self.stoi.items()}

        self.C = torch.randn((27, embedding_dim))
        self.W1 = torch.randn((embedding_dim * block_size, hidden_dim))
        self.b1 = torch.randn(hidden_dim)
        self.W2 = torch.randn((hidden_dim, 27))
        self.b2 = torch.randn(27)
        
        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]

    def forward(self, X):
        emb = self.C[X]
        h = torch.tanh(emb.view(-1, self.embedding_dim * self.block_size) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits

    def load(self, path):
        self.parameters = torch.load(path)
        self.C, self.W1, self.b1, self.W2, self.b2 = self.parameters

    def predict(self, context, n_chars=1):
        context_idx = [self.stoi[c] for c in context]
        context_idx = torch.tensor(context_idx).unsqueeze(0)
        predictions = []
        with torch.no_grad():  # On désactive le calcul des gradients pour la prédiction
            for _ in range(n_chars):
                logits = self.forward(context_idx)
                probs = F.softmax(logits, dim=1)
                next_idx = torch.multinomial(probs, num_samples=1).item()
                next_char = self.itos[next_idx]
                predictions.append(next_char)
                context_idx = torch.cat((context_idx[:,1:], torch.tensor([[next_idx]])), dim=1)
        return ''.join(predictions)

if __name__ == "__main__":
    # Instancier le modèle
    model = CharRNN()

    # Charger le modèle
    model.load('MakeMore/char_rnn_model_version_Final.pth')

    # Prédire le prochain caractère
    context = 'alexa'
    next_char = model.predict(context, n_chars=10)
    print(f"Next character prediction for '{context}': {next_char}")
