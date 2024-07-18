import socket
import time
import threading

# UDP 서버 정보
SERVER_IP = "203.229.155.232"
SERVER_PORT = 12345

# 두 개의 Wi-Fi 인터페이스 IP 주소 (사설 IP 주소일 수 있음)
INTERFACE_1_IP = "172.20.10.3"
INTERFACE_2_IP = "192.168.0.80"

# 패킷 크기와 전송 간격 설정 (H.264 트래픽 유사)
PACKET_SIZE = 1024  # 1KB
PACKET_INTERVAL = 0.033  # 약 30fps

# UDP 소켓 생성
sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sock1.bind((INTERFACE_1_IP, 0))
sock2.bind((INTERFACE_2_IP, 0))

def send_packets(sock, interface_name):
    seq_num = 0
    while True:
        message = f"{interface_name}: Sequence {seq_num}".encode()
        sock.sendto(message, (SERVER_IP, SERVER_PORT))
        seq_num += 1
        time.sleep(PACKET_INTERVAL)

# 두 개의 스레드 생성 및 시작
thread1 = threading.Thread(target=send_packets, args=(sock1, "Interface 1"))
thread2 = threading.Thread(target=send_packets, args=(sock2, "Interface 2"))

thread1.start()
thread2.start()

thread1.join()
thread2.join()
