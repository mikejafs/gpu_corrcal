import os
from pathlib import Path

"""
- Note that __file__ contains the path to the file containing
    this code
- .resolve() is fairly important since it forces an absolute
    path, meaning certainty about which file is being loaded

"""

dir = Path(__file__).resolve().parent.parent
print(dir)


# we can drop into a parent adjacent file using the simple
# / "sub-folder-name" syntax:

dir = Path(__file__).resolve().parent.parent / "other_folder"

