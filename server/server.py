# server/server.py

import socket
import time
import config
import csv

def log_packet(interface, sequence, arrival_time, latency):
    with open(config.LOG_FILE_PATH, mode='a') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow([interface, sequence, arrival_time, latency])

def server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((config.SERVER_IP, config.SERVER_PORT))
    
    packets = {}

    while True:
        data, addr = sock.recvfrom(2048)
        arrival_time = time.time()
        message = data.decode().strip()
        header, timestamp = message.split('|')
        parts = header.split()
        interface = addr[0]
        sequence = int(parts[-1])
        send_time = float(timestamp)
        
        if interface not in packets:
            packets[interface] = {}
        
        packets[interface][sequence] = arrival_time
        
        latency = arrival_time - send_time
        log_packet(interface, sequence, arrival_time, latency)
        
        print(f"Latency for packet {sequence} from {interface}: {latency:.6f} seconds")

if __name__ == "__main__":
    # Create log file and write header if it doesn't exist
    with open(config.LOG_FILE_PATH, mode='w') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["Interface", "Sequence Number", "Arrival Time", "Latency"])
    
    server()
