LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 PYTHONPATH=$(pwd)/build/codec_module python3 main.py
- RL train/eval

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 PYTHONPATH=$(pwd)/build/codec_module python3 get_ssim.py
- 전체를 i-frame으로 인코딩 및 디코딩 했을 때 frames의 ssim을 계산

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 PYTHONPATH=/home/songmu/multipath/rl/build/codec_module python3 evaluation_baseline.py
- baseline의 평가
    - LG, KT, Combine
    - GoP : 1, 10, 30
- evaluation dataset을 평가하도록 dataset 범위 설정 해줘야 함

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 PYTHONPATH=/home/songmu/multipath/rl/build/codec_module python3 evaluation_heuristic.py
- heuristic 알고리즘 평가