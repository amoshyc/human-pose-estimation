import pathlib
from mpii import MPIIDownloader

root_dir = pathlib.Path('./mpii')
dl = MPIIDownloader(root_dir)
dl.start()
