import torch
import torch.nn as nn

class AccidentSeverityModel(nn.Module):
    """
    Modèle de réseau de neurones pour la prédiction de la gravité des accidents.
    """
    def __init__(self, input_size, num_classes):
        """
        Initialise le modèle.
        
        Args:
            input_size (int): Taille de l'entrée du modèle.
            num_classes (int): Nombre de classes de sortie.
        """
        super(AccidentSeverityModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """
        Définit le passage avant du modèle.
        
        Args:
            x (torch.Tensor): Tensor d'entrée.
        
        Returns:
            torch.Tensor: Tensor de sortie.
        """
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

