import git
from pathlib import Path

PROJECT_HOME_DIR = Path(str(git.Repo(search_parent_directories=True).working_tree_dir))
