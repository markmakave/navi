# conect to 'markmakave.com:80' and send random data
import socket
import random
import time
import sys
import math

# create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# get local machine name
host = "markmakave.com"
port = 80

# connection to hostname on the port.
s.connect((host, port))

# Send wave data
while True:
    data = bytearray(10000)
    for i in range(0, 100):
        for j in range(0, 100):
            data[i * 100 + j] = int(math.sin(i / 10.0 + j / 10.0 + time.time() * 2) * 127 + 128)
    
    s.send(data)
