try:
    # Relative import (when run as module: python3 -m cnn.main)
    from .cnnModel import train_model, predict_image
    from .CSVCreator import create_csv
except ImportError:
    # Absolute import (when run as script: python3 main.py)
    from cnnModel import train_model, predict_image
    from CSVCreator import create_csv
import os

def main():
    # Path to dataset CSV
    csv_name = 'dataset.csv'
    
    # Get dataset path from environment variable (for Docker) or use project Dataset folder
    dataset_path = os.environ.get('DATASET_PATH')
    
    if not dataset_path:
        # Get absolute path to Dataset folder in project root
        # main.py is in src/cnn/, so we go up 2 levels to project root
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_file_dir))
        dataset_path = os.path.join(project_root, 'Dataset', 'train')
        
        # Use absolute path
        dataset_path = os.path.abspath(dataset_path)
    
    print(f"Dataset path: {dataset_path}")
    
    # Verify dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}\n"
                              f"Please ensure Dataset/train folder exists in project root.")
    
    model_name = 'ai_detection_model.h5'
    # Create model directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    
    # Create CSV
    if not os.path.exists(os.path.join(dataset_path, csv_name)):
        print("CSV file not found.")
        print("Creating CSV from dataset...")
        create_csv(data_path=dataset_path, output_csv_name=csv_name) 
    
    print("Starting CNN training for AI-generated image detection...")
    
    # Train the model with optimized settings
    model, history = train_model(
        dataset_path=dataset_path,
        epochs=20,  # Adjust as needed
        batch_size=32,
        validation_split=0.2,
        use_cache=True,  # Enable caching for faster subsequent epochs
        cache_in_memory=False,  # Use disk cache (set True for small datasets)
        use_mixed_precision=True,  # Enable mixed precision for faster GPU training
        enable_augmentation=False,  # Set True to enable data augmentation
        model_save_path=model_path  # Save best model during training
    )
    
    # Save the trained model
    model.save(model_path)
    print("Model saved as {model_file}")
    
    # Print final accuracy
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"Final Training Accuracy: {final_accuracy:.4f}")
    print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")

if __name__ == "__main__":
    print("Starting CNN main...")
    main()