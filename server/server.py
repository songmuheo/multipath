import socket
import csv
import time
import threading

# 서버 설정
SERVER_IP = "0.0.0.0"
SERVER_PORT = 12345
BUFFER_SIZE = 1024

# CSV 파일 설정
CSV_FILE_PATH = "data/server_logs.csv"

# 서버 소켓 생성
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((SERVER_IP, SERVER_PORT))

# 데이터 기록을 위한 파일 오픈
csv_file = open(CSV_FILE_PATH, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Interface", "Sequence Number", "Latency", "Cycle"])

packet_log = {}
cycle_number = 0

def process_packet(data, addr):
    global cycle_number
    current_time = time.time()
    message = data.decode()
    interface, seq_num = message.split(": Sequence ")
    seq_num = int(seq_num)
    src_ip = addr[0]
    
    # 패킷 수신 시간 기록
    if (src_ip, seq_num) in packet_log:
        # 패킷이 다른 인터페이스로도 전송된 경우
        send_time, interface_name = packet_log.pop((src_ip, seq_num))
        latency = current_time - send_time
        csv_writer.writerow([current_time, interface_name, seq_num, latency, cycle_number])
        csv_writer.writerow([current_time, interface, seq_num, latency, cycle_number])
    else:
        # 첫 번째 인터페이스에서 패킷 수신
        packet_log[(src_ip, seq_num)] = (current_time, interface)

def reset_cycle():
    global cycle_number
    while True:
        time.sleep(12)  # 클라이언트의 전송 지속 시간보다 약간 긴 시간 대기
        cycle_number += 1

# 사이클 리셋 스레드 시작
threading.Thread(target=reset_cycle, daemon=True).start()

while True:
    data, addr = sock.recvfrom(BUFFER_SIZE)
    threading.Thread(target=process_packet, args=(data, addr)).start()
