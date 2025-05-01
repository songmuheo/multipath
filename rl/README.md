LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 PYTHONPATH=$(pwd)/codec_module/build python3 main.py
- RL train/eval
- 관련 logging, model 저장은 rl/results/연월일/ 폴더에 저장
- config에 해당하는 폴더 명 등 위치해 있음

evaluation/evaluation_rl.ipynb
- 해당하는 학습 결과 분석
- 분석 결과는 /home/songmu/multipath/rl/results/{연월일}/logs/output' 폴더에 저장
- 분석할 학습 결과 폴더 지정해 줘야함
    - folder = {folder path}

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 PYTHONPATH=$(pwd)/build/codec_module python3 get_ssim.py
- 전체를 i-frame으로 인코딩 및 디코딩 했을 때 frames의 ssim을 계산

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 PYTHONPATH=$(pwd)/codec_module/build python3 evaluation/evaluation_baseline.py
- baseline의 평가
    - LG, KT, Combine
    - GoP : 1, 10, 30
- 수정 parameters
    - latency_threshold : loss의 기준이 되는 latency threshold
    - training_data_split : evaluation set을 해당 데이터 셋의 몇 %로 자를 것인지, 해당 퍼센테이지의 뒷 쪽 부분을 evaluation에 이용
- 결과 저장은 env_logs 폴더에 저장 됨 (server/logs/연월일/)

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 PYTHONPATH=/home/songmu/multipath/rl/build/codec_module python3 evaluation/evaluation_heuristic.py
- heuristic 알고리즘 평가


LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 PYTHONPATH=$(pwd)/codec_module/build python3 test_encoder_decoder.py

environment/env_delayed_feedback.py
- state update를 frame 단위로 delay 시켜줌
- delayed feedback 을 frame 단위로 적용