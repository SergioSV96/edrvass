import os, os.path
import errno


def mkdir_p(path):
    '''Create a directory and don't error if it already exists.

    Parameters
    ----------
    path : str
        The path to the directory you want to create.
    '''
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def safe_open_w(path):
    '''Open file and create its parent directry if it does not exist.

    Parameters
    ----------
    path : str
        The path to the file you want to open.
    '''
    mkdir_p(os.path.dirname(path))
    return open(path, 'w')