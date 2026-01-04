from cnnModel import train_model, predict_image
import os

def main():
    # Path to dataset CSV
    dataset_name = 'dataset.csv'
    dataset_path = r"E:\omg bruhhhhhh\DatasetFixed\train" # CHANGE THIS TO YOUR DATASET PATH
    model_name = 'ai_detection_model.h5'
    model_path = os.path.join(r"./src/cnn/model/", model_name)
    
    print("Starting CNN training for AI-generated image detection...")
    
    # Train the model
    model, history = train_model(
        dataset_path=dataset_path,
        epochs=20,  # Adjust as needed
        batch_size=32,
        validation_split=0.2
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