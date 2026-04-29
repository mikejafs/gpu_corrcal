import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_path, "..", "fused_reduction"))
sys.path.insert(0, base_dir)

print(base_dir)

from warp_red_kern_r3 import *

