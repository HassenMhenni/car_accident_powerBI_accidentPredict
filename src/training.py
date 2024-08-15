import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from src.model import AccidentSeverityModel

def train_and_eval(X_train, y_train, X_test, y_test, num_epochs=100, patience=5):
    """
    Entraîne et évalue le modèle avec early stopping.
    
    Args:
        X_train (pd.DataFrame): Données d'entraînement.
        y_train (pd.Series): Étiquettes d'entraînement.
        X_test (pd.DataFrame): Données de test.
        y_test (pd.Series): Étiquettes de test.
        num_epochs (int): Nombre maximum d'époques.
        patience (int): Nombre d'époques à attendre avant l'arrêt précoce.
    
    Returns:
        tuple: accuracy, predicted, y_test_tensor, model
    """
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    
    input_size = X_train_tensor.shape[1]
    num_classes = len(y_train.unique())
    model = AccidentSeverityModel(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = np.inf
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)

        print(f'Epoch {epoch+1}/{num_epochs} - Train loss: {loss.item():.4f} - Val loss: {val_loss.item():.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping on epoch {epoch + 1}')
                break
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    
    return accuracy, predicted, y_test_tensor, model

def cross_val(X, y, n_splits=5):
    """
    Effectue une validation croisée.
    
    Args:
        X (pd.DataFrame): Données d'entrée.
        y (pd.Series): Étiquettes.
        n_splits (int): Nombre de plis pour la validation croisée.
    
    Returns:
        list: Liste des précisions pour chaque pli.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f'--------Training fold {fold}--------')
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        accuracy, _, _, _ = train_and_eval(X_train_fold, y_train_fold, X_test_fold, y_test_fold)
        fold_accuracies.append(accuracy)
        
        print(f'Fold {fold}: Accuracy = {accuracy*100:.2f}%')

    return fold_accuracies

def train_final_model(X, y):
    """
    Entraîne le modèle final sur l'ensemble des données.
    
    Args:
        X (pd.DataFrame): Données d'entrée.
        y (pd.Series): Étiquettes.
    
    Returns:
        tuple: accuracy, predicted, true_labels, model
    """
    print('--------Training global data--------')
    return train_and_eval(X, y, X, y)
