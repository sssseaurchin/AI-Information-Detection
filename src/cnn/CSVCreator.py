import os
import csv


# Adjust as needed
def create_csv(dataset_folder:str, output_csv:str="dataset.csv") -> None:
    """Creates a CSV file from a folder with categorical subfolders containing images. Works on same folder level.

    Args:
        dataset_folder (str): Folder containing subfolders of image categories.
        output_csv (str, optional): Output CSV folder name. Defaults to "dataset.csv".
    """
    
    # Paths
    curr = os.path.dirname(os.path.abspath(__file__))
    dataset_folder = os.path.join(curr, dataset_folder)
    output_csv = os.path.join(dataset_folder, output_csv)

    # Get image categories from subfolders
    categories = [d for d in os.listdir(dataset_folder) 
                if os.path.isdir(os.path.join(dataset_folder, d))]

    print(f"Found categories: {categories}")

    # Create CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'category'])
        
        for category in categories:
            category_path = os.path.join(dataset_folder, category)
            for image in os.listdir(category_path):
                if image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_path = os.path.join(category_path, image)
                    writer.writerow([image_path, category])

    print(f"CSV created: {output_csv}")