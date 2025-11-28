import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from track_crop import crop
from IR import IR

crop()
IR()