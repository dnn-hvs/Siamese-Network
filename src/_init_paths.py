import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add loss to PYTHONPATH
lib_path = osp.join(this_dir, 'loss')
add_path(lib_path)

# Add utils to PYTHONPATH
lib_path = osp.join(this_dir, 'utils')
add_path(lib_path)

# Add network to PYTHONPATH
lib_path = osp.join(this_dir, 'network')
add_path(lib_path)

# Add dataset to PYTHONPATH
lib_path = osp.join(this_dir, 'dataset')
add_path(lib_path)