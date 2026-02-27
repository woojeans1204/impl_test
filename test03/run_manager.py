import os
import subprocess
import time
import multiprocessing
import json               
import requests           
from datetime import datetime
from dotenv import load_dotenv
import re
import yaml

os.environ['TZ'] = 'Asia/Seoul'
if hasattr(time, 'tzset'):
    time.tzset()
load_dotenv()
# ====================================================
# [설정] 경로 및 환경 설정
# ====================================================
BASE_DIR = os.getenv("BASE_DIR")
PROJECT_REL_PATH = "impl_test/test3"
PROJECT_FULL_PATH = os.path.join(BASE_DIR, PROJECT_REL_PATH)
CONFIGS_FULL_PATH = os.path.join(PROJECT_FULL_PATH, "configs")

CONFIG_DIR = "exp260202"
LOG_BASE_DIR = os.path.join(BASE_DIR, "logs")
EXP_LIST_FILE = "experiment_list.conf"
NUM_GPUS = int(os.getenv("NUM_GPUS", 1))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 1))
START_PORT = 29500

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
# ====================================================

def send_slack_msg(text):
    if "YOUR/WEBHOOK" in SLACK_WEBHOOK_URL:
        return
    payload = {"text": text, "username": "ExpManager", "icon_emoji": ":rocket:"}
    try:
        requests.post(SLACK_WEBHOOK_URL, json=payload)
    except Exception as e:
        print(f"❌ 슬랙 전송 실패: {e}")

# [수정] 로그 파일에서 에포크 정보를 추출하는 함수
def get_last_epoch(log_path):
    if not os.path.exists(log_path):
        return "로그 생성 중..."
    try:
        # 마지막 2000바이트 정도만 읽어서 최신 진행 상황 확인
        with open(log_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 2000)) # 끝부분 2KB만 읽기
            chunk = f.read().decode('utf-8', errors='ignore')

        # tqdm은 \r을 사용하여 한 줄을 덮어씁니다. 
        # 이를 줄바꿈으로 변경하여 가장 마지막에 찍힌 정보를 찾습니다.
        lines = chunk.replace('\r', '\n').split('\n')
        
        for line in reversed(lines):
            line = line.strip()
            # "Epoch"와 "%"가 동시에 포함된 줄을 찾습니다.
            if "Epoch" in line and "%" in line:
                # "Epoch 2: 30%|███" -> "Epoch 2: 30%"
                return line.split('|')[0].strip()
            elif "Epoch" in line:
                return line.strip()
    except Exception as e:
        print(f"로그 읽기 오류: {e}")
    
    return "진행 정보 업데이트 중..."

# [수정 1] 모니터링 함수: 현재/전체 에포크 파싱하여 표시
def status_monitor(finished_count, total_count, start_time, worker_status):
    while True:
        time.sleep(1800) # 30분마다
        
        current_done = finished_count.value
        elapsed = datetime.now() - start_time
        
        status_text = ""
        for wid, info in worker_status.items():
            log_line = get_last_epoch(info['log_path'])
            total_epochs = info.get('total_epochs', '?') # 저장된 총 에포크 가져오기
            
            # 로그에서 숫자만 추출 (예: "Epoch 101" -> 101)
            current_epoch = "?"
            match = re.search(r"Epoch\s+(\d+)", log_line)
            if match:
                current_epoch = match.group(1)

            # 1. 숫자형으로 변환 시도
            try:
                curr_val = int(current_epoch) if current_epoch != '?' else None
                total_val = int(total_epochs) if total_epochs != '?' else None
            except (ValueError, TypeError):
                curr_val, total_val = None, None

            # 2. 퍼센트 계산 (둘 다 숫자일 때만)
            progress_pct = ""
            if curr_val is not None and total_val is not None and total_val > 0:
                progress_pct = f" - {(curr_val-1) / total_val * 100:.1f}%"

            # 3. 최종 텍스트 조립
            status_text += f"• *Worker {wid}*: {info['config']} (Epoch: {current_epoch}/{total_epochs}{progress_pct})\n"

        msg = (
            f"⏰ *[Hourly Update] 실험 진행 보고*\n"
            f"- 진행률: {current_done} / {total_count} 완료\n"
            f"- 경과 시간: {elapsed}\n"
            f"- *현재 상세 상황*:\n{status_text if status_text else '• 대기 중'}"
        )
        send_slack_msg(msg)
        
        if current_done >= total_count:
            break

# [추가] 총 에포크 수 읽기 헬퍼 함수
def get_total_epochs(path):
    try:
        with open(path, 'r') as f:
            conf = yaml.safe_load(f)
        return conf.get('train').get('epochs') or '?'
    except:
        return '?'

def worker(worker_id, task_queue, finished_count, lock, worker_status):
    real_gpu_id = worker_id % NUM_GPUS 
    
    while True:
        try:
            config_file = task_queue.get_nowait()
        except multiprocessing.queues.Empty:
            if worker_id in worker_status: del worker_status[worker_id]
            break

        print(f"▶️ [GPU {real_gpu_id}] 시작: {config_file}") # 이 줄이 있는지 확인!
        log_name = os.path.basename(config_file)
        log_path = os.path.join(LOG_BASE_DIR, f"log_{log_name}.out")

        # [수정 2] 총 에포크 읽어서 공유 딕셔너리에 저장
        full_conf_path = os.path.join(CONFIGS_FULL_PATH, CONFIG_DIR, config_file)
        total_epochs = get_total_epochs(full_conf_path)
        
        worker_status[worker_id] = {
            'config': config_file, 
            'log_path': log_path, 
            'total_epochs': total_epochs # 여기 추가됨
        }

        cmd = (
            f"CUDA_VISIBLE_DEVICES={real_gpu_id} "
            f"accelerate launch --num_processes 1 "
            f"--main_process_port {START_PORT + worker_id} " 
            f"train.py --config {CONFIG_DIR}/{config_file}"
        )

        with open(log_path, "w") as log_file:
            process = subprocess.run(
                cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT, cwd=PROJECT_FULL_PATH 
            )

        send_slack_msg(f"✅ [GPU {real_gpu_id}] 실험 완료: `{config_file}`")

        with lock:
            finished_count.value += 1
        
        time.sleep(1)
        print(f"✅ [GPU {real_gpu_id}] 종료: {config_file}") # 종료 시 출력

def main():
    start_time = datetime.now() 
    send_slack_msg(f"🎬 [Cluster] 스케줄러 가동 시작\n- 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(LOG_BASE_DIR, exist_ok=True)
    
    tasks = []
    if os.path.exists(EXP_LIST_FILE):
        with open(EXP_LIST_FILE, 'r') as f:
            tasks = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    else:
        return

    total_tasks = len(tasks)
    task_queue = multiprocessing.Queue()
    for task in tasks: task_queue.put(task)

    # [수정] 프로세스 간 공유 가능한 매니저 객체 생성
    manager = multiprocessing.Manager()
    worker_status = manager.dict() # 워커 상태를 저장할 공유 딕셔너리
    finished_count = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()

    # 모니터링 프로세스 실행
    monitor_p = multiprocessing.Process(
        target=status_monitor, 
        args=(finished_count, total_tasks, start_time, worker_status)
    )
    monitor_p.daemon = True
    monitor_p.start()

    processes = []
    for worker_id in range(NUM_WORKERS):
        p = multiprocessing.Process(
            target=worker, 
            args=(worker_id, task_queue, finished_count, lock, worker_status)
        )
        p.start()
        processes.append(p)

    for p in processes: p.join()

    duration = datetime.now() - start_time
    send_slack_msg(f"🎉 모든 실험 종료! (총 소요: {duration})")

if __name__ == "__main__":
    main()