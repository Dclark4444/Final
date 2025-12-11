import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/hanimn/intro_robo_ws/src/Final/src_defend_agent_attack/install/defendagentattack'
