import socket
import time
import threading
import config

def generate_packet(sequence_number, interface_id, packet_size):
    header = f"Packet from interface {interface_id} with sequence {sequence_number}".encode()
    timestamp = f"{time.time():.6f}".encode()  # 타임스탬프 추가
    padding = b' ' * (packet_size - len(header) - len(timestamp))
    return header + b'|' + timestamp + padding

def send_packets(interface_ip, interface_id, interface_name):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, 25, interface_name.encode())  # SO_BINDTODEVICE 옵션 설정
    sock.bind((interface_ip, 0))
    
    sequence_number = 0
    
    while True:
        remaining_bytes = config.BYTES_PER_FRAME
        while remaining_bytes > 0:
            packet_size = min(config.PACKET_SIZE, remaining_bytes)
            packet = generate_packet(sequence_number, interface_id, packet_size)
            sock.sendto(packet, (config.SERVER_IP, config.SERVER_PORT))
            remaining_bytes -= packet_size
            
            print(f"Interface {interface_id} ({interface_ip}) sent {packet_size} bytes packet with sequence {sequence_number}")
            
            sequence_number += 1
        time.sleep(1/config.FPS)

if __name__ == "__main__":
    interface1_thread = threading.Thread(target=send_packets, args=(config.INTERFACE1_IP, 1, config.INTERFACE1_NAME))
    interface2_thread = threading.Thread(target=send_packets, args=(config.INTERFACE2_IP, 2, config.INTERFACE2_NAME))
    
    interface1_thread.start()
    interface2_thread.start()
    
    interface1_thread.join()
    interface2_thread.join()
