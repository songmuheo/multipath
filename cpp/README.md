# c++ codes
#### Master branch

Client에서 Realsense depth camera로부터 받은 영상 데이터를 두 개의 Interfaces에 duplicate하여 Server로 송신한다. 이 때 H.264 코덱을 이용하여 송신한다.

Server에서는 각 Interface에서 받은 영상을 Ip주소 혹은 Interface 별로 두 개의 영상을 서로 다른 open CV window에서 영상을 재생한다.
### Client
```
/~/Multipath/cpp/client 에서
mkdir build
cmake ..
make
./client
```

### Server
```
/~/Multipath/cpp/server 에서
mkdir build
cmake ..
make
./server
```