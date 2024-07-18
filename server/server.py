# server/server.py

import socket
import time
import config
import csv
from datetime import datetime

def log_packet(interface, sequence, arrival_time):
    with open(config.LOG_FILE_PATH, mode='a') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow([interface, sequence, arrival_time])

def server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((config.SERVER_IP, config.SERVER_PORT))
    
    packets = {}

    while True:
        data, addr = sock.recvfrom(2048)
        arrival_time = time.time()
        message = data.decode().strip()
        interface = addr[0]
        sequence = int(message.split()[-1])
        
        if interface not in packets:
            packets[interface] = {}
        
        if sequence not in packets[interface]:
            packets[interface][sequence] = arrival_time
        
        log_packet(interface, sequence, arrival_time)
        
        # Calculate latency and loss (simple implementation)
        if interface in packets and sequence in packets[interface]:
            latency = arrival_time - packets[interface][sequence]
            print(f"Latency for packet {sequence} from {interface}: {latency:.6f} seconds")

if __name__ == "__main__":
    # Create log file and write header if it doesn't exist
    with open(config.LOG_FILE_PATH, mode='w') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["Interface", "Sequence Number", "Arrival Time"])
    
    server()
