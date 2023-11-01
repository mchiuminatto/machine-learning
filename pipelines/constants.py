import os
from pathlib import PurePath

ROOT_FOLDER_APP: str = os.path.dirname(os.path.abspath(__file__))
ROOT_FOLDER: str = str(PurePath(ROOT_FOLDER_APP).parent)
