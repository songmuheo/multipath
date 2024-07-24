# Python codes
#### Master branch

Client에서 2개의 Interfaces를 통해 UDP 패킷을 전송.

Server에서는 각 Interface에서 받은 UDP 패킷에 대한 정보를 csv 파일로 로깅한다

#### Version_1.1

Client에서 2개의 Interfaces를 통해 UDP 패킷을 전송. 이 때, H.264에서 영상을 전송할 때 발생하는 traffic과 최대한 동일한 traffic을 발생 시킨다.

Server에서는 각 Interface에서 받은 UDP 패킷에 대한 정보를 csv 파일로 로깅한다
### Client
```
/~/Multipath/python/client 에서
python3 client.py
```

### Server
```
/~/Multipath/python/server 에서
python3 server.py
```