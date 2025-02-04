import os, shutil

def generate_folder(path):
    folder_path = path


    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' has been removed successfully.")
    except FileNotFoundError:
        print(f"Folder '{folder_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied: Unable to delete '{folder_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    os.mkdir(folder_path)


def clear_folder(folder : str):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))