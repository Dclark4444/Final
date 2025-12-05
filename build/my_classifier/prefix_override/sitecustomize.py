import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/clarkd26/intro_robo_ws/src/Final/install/my_classifier'
