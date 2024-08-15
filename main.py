import os
import warnings
from src.data_preparation import load_data, process_data
from src.training import cross_val, train_final_model
from src.utils import save_results, save_model

# Ignorer les avertissements
warnings.filterwarnings('ignore')

def main():
    # Définir le chemin du répertoire
    directory_path = os.path.dirname(os.path.abspath(__file__))
    
    # Charger et préparer les données
    print("Load and preprocess data...")
    data = load_data(directory_path)
    X, y = process_data(data)
    
    # Cross val
    print("Cross Validation...")
    fold_accuracies = cross_val(X, y)
    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    print(f'Average accuracy on all folds: {avg_accuracy*100:.2f}%')
    
    # Entraînement du modèle final
    print("Train final model...")
    accuracy, predictions, true_labels, final_model = train_final_model(X, y)
    print(f'Global accuracy after training: {accuracy*100:.2f}%')
    
    # Sauvegarder les résultats et le modèle
    model_path = os.path.join(directory_path, 'model')
    os.makedirs(model_path, exist_ok=True)
    
    print("Save results...")
    save_results(accuracy, true_labels, predictions, model_path)
    
    print("Save model...")
    save_model(final_model, model_path)
    
    print("Finish!")

if __name__ == "__main__":
    main()