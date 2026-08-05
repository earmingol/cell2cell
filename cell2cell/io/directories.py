# -*- coding: utf-8 -*-

import os

from natsort import natsorted


def create_directory(pathname):
    '''Creates a directory.

    Uses a path to create a directory. It creates
    all intermediate folders before creating the
    leaf folder.

    Parameters
    ----------
    pathname : str
        Full path of the folder to create.
    '''
    if not os.path.isdir(pathname):
        os.makedirs(pathname)
        print("{} was created successfully.".format(pathname))
    else:
        print("{} already exists.".format(pathname))


def get_files_from_directory(pathname, dir_in_filepath=False):
    '''Obtains a list of filenames in a folder.

    Parameters
    ----------
    pathname : str
        Full path of the folder to explore.

    dir_in_filepath : boolean, default=False
        Whether adding `pathname` to the filenames

    Returns
    -------
    filenames : list
        A list containing the names (strings) of the files
        in the folder, naturally sorted by filename.
    '''
    directory = os.fsencode(pathname)
    # Naturally sorted to avoid a filesystem-dependent order of the files
    files = natsorted([os.fsdecode(file) for file in os.listdir(directory)])
    filenames = [pathname + '/' + file if dir_in_filepath else file for file in files]
    return filenames
