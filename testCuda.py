import torch

# Vérifiez si CUDA est disponible
cuda_available = torch.cuda.is_available()

if cuda_available:
    print("CUDA est disponible. Vous pouvez utiliser le GPU.")
else:
    print("CUDA n'est pas disponible. Vous utilisez le CPU.")

# Afficher le nom du GPU s'il est disponible
if cuda_available:
    print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")

# Créer un tenseur
x = torch.tensor([1.0, 2.0, 3.0])

# Déplacer le tenseur vers le GPU s'il est disponible
if cuda_available:
    x = x.to('cuda')

# Effectuer une opération simple sur le tenseur
y = x * 2

# Déplacer le tenseur vers le CPU
if cuda_available:
    y = y.to('cpu')

# Afficher le résultat
print(f"Tenseur d'origine : {x}")
print(f"Résultat après opération : {y}")
