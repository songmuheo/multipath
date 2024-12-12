LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 PYTHONPATH=$(pwd)/build/codec_module python3 main.py

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 PYTHONPATH=$(pwd)/build/codec_module python3 evaluation_compare.py

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 PYTHONPATH=$(pwd)/build/codec_module python3 analyze.py