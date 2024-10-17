import os

def get_path_to_venv():
    # Construct the path to the .venv folder (join with the current directory up to the root of the project)
    venv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.venv'))
    return venv_path

def get_path_to_data():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data folder not found at: {data_path}")
    return data_path