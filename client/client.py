# client/client.py

import socket
import time
import threading
import config

def generate_packet(sequence_number, interface_id):
    header = f"Packet from interface {interface_id} with sequence {sequence_number}".encode()
    padding = b' ' * (config.PACKET_SIZE - len(header))
    return header + padding

def send_packets(interface_ip, interface_id):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((interface_ip, 0))
    
    sequence_number = 0
    
    while True:
        packet = generate_packet(sequence_number, interface_id)
        sock.sendto(packet, (config.SERVER_IP, config.SERVER_PORT))
        
        sequence_number += 1
        time.sleep(config.PACKET_INTERVAL)

if __name__ == "__main__":
    interface1_thread = threading.Thread(target=send_packets, args=(config.INTERFACE1_IP, 1))
    interface2_thread = threading.Thread(target=send_packets, args=(config.INTERFACE2_IP, 2))
    
    interface1_thread.start()
    interface2_thread.start()
    
    interface1_thread.join()
    interface2_thread.join()
