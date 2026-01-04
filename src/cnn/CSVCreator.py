import os
import csv


# Adjust as needed
def create_csv(data_path:str, output_csv_name:str="dataset.csv") -> None:
    """Creates a CSV file from a folder with categorical subfolders containing images. Works on same folder level.

    Args:
        data_path (str): Path to the dataset folder.
        output_csv_name (str, optional): Output CSV file name. Defaults to "dataset.csv".
    """
    
    # Paths
    csv_path = os.path.join(data_path, output_csv_name)

    # Get image categories from subfolders
    categories = [d for d in os.listdir(data_path) 
                if os.path.isdir(os.path.join(data_path, d))]

    print(f"Found categories: {categories}")

    # Create CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'category'])
        
        for category in categories:
            category_path = os.path.join(data_path, category)
            if category!= "_unsorted":
                for image_name in os.listdir(category_path):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        writer.writerow([image_name, category])
                else:
                    pass # Skip _unsorted folder        

    print(f"CSV created: {output_csv_name} at {csv_path}")
    
create_csv(r"E:\omg bruhhhhhh\DatasetFixed\train")