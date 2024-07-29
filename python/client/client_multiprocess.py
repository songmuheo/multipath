import socket
import time
import multiprocessing
import config

def generate_packet(sequence_number, interface_id, packet_size):
    header = f"Packet from interface {interface_id} with sequence {sequence_number}".encode()
    timestamp = f"{time.time():.6f}".encode()  # 타임스탬프 추가
    padding = b' ' * (packet_size - len(header) - len(timestamp))
    return header + b'|' + timestamp + padding

def send_packets(interface_ip, interface_id, interface_name, start_event):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, 25, interface_name.encode())  # SO_BINDTODEVICE 옵션 설정
    sock.bind((interface_ip, 0))
    
    sequence_number = 0
    
    # Start event를 기다림
    start_event.wait()

    while sequence_number <= 50000:
        remaining_bytes = config.BYTES_PER_FRAME
        while remaining_bytes > 0:
            packet_size = min(config.PACKET_SIZE, remaining_bytes)
            packet = generate_packet(sequence_number, interface_id, packet_size)
            sock.sendto(packet, (config.SERVER_IP, config.SERVER_PORT))
            remaining_bytes -= packet_size
            
            print(f"Interface {interface_id} ({interface_ip}) sent {packet_size} bytes packet with sequence {sequence_number}")
            
            sequence_number += 1
            if sequence_number > 50000:
                break
        time.sleep(1/config.FPS)

if __name__ == "__main__":
    start_event = multiprocessing.Event()

    interface1_process = multiprocessing.Process(target=send_packets, args=(config.INTERFACE1_IP, 1, config.INTERFACE1_NAME, start_event))
    interface2_process = multiprocessing.Process(target=send_packets, args=(config.INTERFACE2_IP, 2, config.INTERFACE2_NAME, start_event))
    
    interface1_process.start()
    interface2_process.start()
    
    # 모든 프로세스가 준비되면 이벤트를 설정하여 동시에 시작하도록 함
    time.sleep(1)  # 모든 프로세스가 준비될 시간을 줌
    start_event.set()
    
    interface1_process.join()
    interface2_process.join()
