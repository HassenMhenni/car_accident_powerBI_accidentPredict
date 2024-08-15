import os
import torch
from sklearn.metrics import classification_report, confusion_matrix

def save_results(accuracy, y_true, y_pred, model_path):
    """
    Sauvegarde les résultats dans un fichier texte.
    
    Args:
        accuracy (float): Précision globale du modèle.
        y_true (torch.Tensor): Vraies étiquettes.
        y_pred (torch.Tensor): Prédictions du modèle.
        model_path (str): Chemin où sauvegarder les résultats.
    """
    results_path = os.path.join(model_path, 'results.txt')
    with open(results_path, 'w') as f:
        f.write(f'Précision globale: {accuracy*100:.2f}%\n\n')
        f.write('Rapport de classification:\n')
        f.write(classification_report(y_true.numpy(), y_pred.numpy(), 
                                      target_names=['Classe 1', 'Classe 2', 'Classe 3', 'Classe 4']))
        f.write('\nMatrice de confusion:\n')
        f.write(str(confusion_matrix(y_true.numpy(), y_pred.numpy())))

def save_model(model, model_path):
    """
    Sauvegarde le modèle entraîné.
    
    Args:
        model (nn.Module): Modèle à sauvegarder.
        model_path (str): Chemin où sauvegarder le modèle.
    """
    model_file = os.path.join(model_path, 'model.pth')
    torch.save(model.state_dict(), model_file)

