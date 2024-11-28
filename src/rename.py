import os

def rename_images_in_folder(folder_path, base_name="Cancer", extension="png"):
    """
    Renames all images in the specified folder to the format:
    base_name1.extension, base_name2.extension, ..., base_nameN.extension
    
    Args:
        folder_path (str): Path to the folder containing images.
        base_name (str): Base name for the renamed files. Default is "Cancer".
        extension (str): File extension for renamed files. Default is "png".
        
    Returns:
        None
    """
    # List of supported image extensions
    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

    # Ensure the extension starts with a dot
    extension = f".{extension.lstrip('.')}"
    
    # Create a counter for file naming
    counter = 1

    # Iterate through all files in the folder
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        
        # Skip non-files or unsupported extensions
        if not os.path.isfile(file_path):
            continue
        
        if not filename.lower().endswith(supported_extensions):
            print(f"Skipping unsupported file: {filename}")
            continue

        # New filename
        new_name = f"{base_name}{counter}{extension}"
        new_file_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(file_path, new_file_path)
        print(f"Renamed: {filename} -> {new_name}")
        
        counter += 1

    print(f"Renamed {counter - 1} files in folder: {folder_path}")

# Example usage
folder_path = "./Data/ProcessedCancer"  # Replace with your folder path
rename_images_in_folder(folder_path)
