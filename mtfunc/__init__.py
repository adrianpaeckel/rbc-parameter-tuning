import os
dir_path=os.path.dirname(os.path.realpath(__file__))
print(dir_path,'  was appended to PYTHONPATH')
os.sys.path.append(dir_path)
# from .restclient import *
# from .helper import *
# from .datacqui import *
# from .evaluation import *
