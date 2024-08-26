# c++ codes
#### Master branch

Client에서 Realsense depth camera로부터 받은 영상 데이터를 두 개의 Interfaces에 duplicate하여 Server로 송신한다. 이 때 H.264 코덱을 이용하여 송신한다.

Server에서는 각 Interface에서 받은 영상을 Ip주소 혹은 Interface 별로 두 개의 영상을 서로 다른 open CV window에서 영상을 재생한다.

#### Version_1.1

Client에서는 패킷에 Interface ID와 Sequence number를 추가하여 전송한다. 이 추가된 헤더와 함께 패킷을 전송

Server에서는 패킷을 수신할 때 헤더에서 Interface ID와 Sequence number를 추출한다. 또한 전역변수로 마지막으로 수신된 시퀀스 번호와 도착 시간을 지정한다. 패킷을 수신하고 도착 시간을 기록하여, 시퀀스 번호가 이전 패킷보다 크거나 같고, 도착 시간이 
더 빠를 경우에만 패킷을 처리한다. 

이를 통해 중복된 패킷 중에서 더 빨리 도착한 패킷을 "선택" 할 수 있다.
위를 통해 2개의 인터페이스에서 오는 영상 패킷을 이용하여 로스가 발생하지 않고, 더 빨리 도착한 패킷을 이용하여 비디오를 스트리밍한다.

#### Version_1.2

영상 전송 패킷에 Sequence 정보, Time stamp 등을 넣어서 성능을 식별하는 기능을 넣을 예정(Latency, Loss, Jitter ...)

코드의 오류가 있는 것들을 잡아볼 예정

매우 좋지 않은 Video quality를 높일 것 -> 현재까지의 낮은 화질은 문제가 있어보임?

두 path에서 오는 것을 먼저 오면, 뒤에건 버리고 앞에것만 catch 하는 이 쪽 부분을 더 보완해야할 것 같음...

#### Version2

Client - 

기존의 오류: realsense pipeline을 thread 실행 내에서 처리해서, thread마다 완전히 같은 frame을 capture하지 않을 수 있음. 따라서, 전송되는 패킷이 완전히 같은 패킷이 아닐 확률이 높고, 디코딩시 오류가 있을 확률이 높음.

thread로 작업을 나누기 이전에 pipeline과 frame capture를 완료하고, thread에서는 전송만 담당하도록 설정

즉, librealsense에서 video 받아오고, ffmpeg으로 H.264인코딩하고, 두 개의 인터페이스로 이 인코딩한 같은 패킷을 전송함. 이 때, 전송을 위한 프로토콜은 단순히 UDP를 사용하였음



Server -

각 포트에 들어온 (즉, 두 인터페이스에서 보낸) UDP 패킷을 단순히 디코딩하고, openCV window에 표시 및 저장한다. 두 패킷의 latency를 고려하여 하나의 영상으로 합치는 코드는 아직 작성하지 않았음.

#### Version3

목표: 
Streaming protocol을 이용한 구현, 두 인터페이스로부터 수신한 패킷을 latency를 고려하여 하나의 최적화된 영상으로 만들기. 

### Client
```
/~/Multipath/cpp/client 에서
mkdir build
cd build
cmake ..
make
./client
```

### Server
```
/~/Multipath/cpp/server 에서
mkdir build
cd build
cmake ..
make
./server
```