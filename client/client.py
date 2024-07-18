import socket
import time
import threading

# 서버 정보
SERVER_IP = "203.229.155.232"
SERVER_PORT = 12345

# 두 개의 Wi-Fi 인터페이스 IP 주소
INTERFACE_1_IP = "172.20.10.3"
INTERFACE_2_IP = "192.168.0.80"

# 패킷 크기와 전송 간격 설정 (H.264 트래픽 유사)
PACKET_SIZE = 1024  # 1KB
PACKET_INTERVAL = 0.033  # 약 30fps
DURATION = 10  # 패킷 전송 지속 시간 (초)

def send_packets(interface_ip, interface_name):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((interface_ip, 0))
    seq_num = 0
    start_time = time.time()
    while time.time() - start_time < DURATION:
        message = f"{interface_name}: Sequence {seq_num}".encode()
        sock.sendto(message, (SERVER_IP, SERVER_PORT))
        seq_num += 1
        time.sleep(PACKET_INTERVAL)
    sock.close()

if __name__ == "__main__":
    thread1 = threading.Thread(target=send_packets, args=(INTERFACE_1_IP, "Interface 1"))
    thread2 = threading.Thread(target=send_packets, args=(INTERFACE_2_IP, "Interface 2"))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
