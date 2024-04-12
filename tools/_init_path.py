import sys
sys.path.insert(0, '../')

import socket
hostname = socket.gethostname()

host = None

if hostname == 'zlin':
    host = 'local'

elif hostname == 'inin':
    host = 'remote'
else:
    host = 'local'