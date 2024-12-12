# utils/directory_manager.py

import os
from datetime import datetime

def create_timestamped_directory(base_dir='results'):
    """현재 연/월/일/시 기준의 폴더를 생성하고 경로를 반환합니다."""
    now = datetime.now()
    # 날짜와 시간을 기반으로 한 폴더 이름 생성 (예: 2024/11/01/14)
    folder_name = now.strftime('%Y_%m_%d_%H_%M')
    folder_path = os.path.join(base_dir, folder_name)
    
    # 메인 폴더 및 하위 폴더 생성
    os.makedirs(folder_path, exist_ok=True)
    logs_path = os.path.join(folder_path, 'logs')
    models_path = os.path.join(folder_path, 'models')
    
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    
    
    return folder_path