from datetime import datetime
import pickle


def get_time():
    """Find current time in human-readable format"""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time


def curr_time():
    """Print current time"""
    print("Current Time: ", get_time(), flush=True)


def load_pickle_file(filename):
    """
    Load an unstructured pickle file and return the data

    Args:
        filename (str): The path to the pickle file

    Returns:
        data (object): The data stored in the pickle file
    """
    fr = open(filename, 'rb')
    data = pickle.load(fr)
    fr.close()
    return data
