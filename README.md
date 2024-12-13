## client/
```
/~/multipath/client
mkdir build
cd build
cmake ..
make
./client
```

H.264를 이용하여 인코딩 및 전송
두 개의 path로 전송
두 개 path에 대한 interface 설정해줘야 함

config.h
server ip
interface 1
interface 2
설정해줘야 함
저장은 client/logs/{연월일}/ 폴더에 저장
해당 폴더에 송신한 frames, log 저장

## server/
```
/~/multipath/server
mkdir build
cd build
cmake ..
make
./server
```

Client가 송신한 frames 받아서 저장
저장은 server/logs/{연월일}/ 에 저장
수신한 bin files 저장은 현재 하지 않음
각 path (kt, lg)관련해서 따로 저장

## rl/

강화학습
