# client/config.py

SERVER_IP = "203.229.155.232"  # 서버 IP 주소
SERVER_PORT = 12345  # 서버 포트
PACKET_SIZE = 1500  # 패킷 크기 (H.264 코덱의 일반적인 프레임 크기)
# PACKET_INTERVAL = 0.033  # 패킷 간 간격 (30fps 영상 데이터 전송 간격)
# INTERFACE1_IP = "10.16.130.64"  # 첫 번째 Wi-Fi 인터페이스 IP
INTERFACE1_IP = "172.20.10.3"  # 첫 번째 Wi-Fi 인터페이스 IP
INTERFACE1_NAME = "wlp1s0"
INTERFACE2_IP = "10.16.132.84"  # 두 번째 Wi-Fi 인터페이스 IP
INTERFACE2_NAME = "wlx588694fd23d6"

BITRATE = 2_000_000  # 2Mbps
FPS = 30
BYTES_PER_SECOND = BITRATE // 8
BYTES_PER_FRAME = BYTES_PER_SECOND // FPS