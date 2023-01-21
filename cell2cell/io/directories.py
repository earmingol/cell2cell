# -*- coding: utf-8 -*-

from __future__ import absolute_import

import os


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
        in the folder.
    '''
    directory = os.fsencode(pathname)
    filenames = [pathname + '/' + os.fsdecode(file) if dir_in_filepath else os.fsdecode(file) for file in os.listdir(directory)]
    return filenames
