"""
서버 시작 스크립트 - FastAPI 서버와 접근성 분석 서비스를 함께 시작
"""
import os
import sys
import subprocess
import argparse
import threading
import time
import signal
import psutil
from config import FASTAPI_HOST, FASTAPI_PORT

def start_fastapi_server():
    """FastAPI 서버 시작"""
    print("FastAPI 서버 시작 중...")
    server_process = subprocess.Popen(
        [sys.executable, "api_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # 서버가 시작될 때까지 기다림
    time.sleep(2)
    
    if server_process.poll() is not None:
        print("FastAPI 서버 시작 실패!")
        stderr = server_process.stderr.read()
        print(f"오류: {stderr}")
        return None
    
    print(f"FastAPI 서버가 http://{FASTAPI_HOST}:{FASTAPI_PORT}에서 실행 중입니다.")
    return server_process

def monitor_output(process, name):
    """프로세스 출력 모니터링"""
    for line in process.stdout:
        print(f"[{name}] {line.strip()}")
    for line in process.stderr:
        print(f"[{name} 오류] {line.strip()}")

def shutdown_process(process):
    """프로세스와 모든 자식 프로세스 종료"""
    if process is None:
        return
    
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        
        for child in children:
            try:
                child.terminate()
            except:
                pass
        
        process.terminate()
        process.wait(timeout=5)
        
        # 여전히 실행 중인 프로세스 강제 종료
        for child in children:
            if child.is_running():
                try:
                    child.kill()
                except:
                    pass
        
        if parent.is_running():
            parent.kill()
            
    except Exception as e:
        print(f"프로세스 종료 중 오류 발생: {e}")

def main():
    parser = argparse.ArgumentParser(description="접근성 분석 서비스 시작")
    parser.add_argument("--api-only", action="store_true", help="FastAPI 서버만 시작")
    args = parser.parse_args()
    
    processes = []
    
    # FastAPI 서버 시작
    api_server = start_fastapi_server()
    if api_server:
        processes.append(api_server)
        threading.Thread(target=monitor_output, args=(api_server, "FastAPI"), daemon=True).start()
    else:
        print("FastAPI 서버를 시작할 수 없습니다. 종료합니다.")
        sys.exit(1)
    
    # 종료 신호 처리
    def signal_handler(sig, frame):
        print("\n종료 신호를 받았습니다. 서버를 종료합니다...")
        for process in processes:
            shutdown_process(process)
        print("모든 서버가 종료되었습니다.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("서버가 실행 중입니다. Ctrl+C를 눌러 종료하세요.")
        
        # 메인 스레드가 종료되지 않도록 유지
        while all(process.poll() is None for process in processes):
            time.sleep(1)
        
        # 어떤 프로세스가 종료되었는지 확인
        for i, process in enumerate(processes):
            if process.poll() is not None:
                name = "FastAPI" if i == 0 else "프로세스"
                print(f"{name}가 종료되었습니다. 반환 코드: {process.returncode}")
                stderr = process.stderr.read()
                if stderr:
                    print(f"오류: {stderr}")
        
    except KeyboardInterrupt:
        print("\n키보드 인터럽트를 받았습니다. 서버를 종료합니다...")
    finally:
        for process in processes:
            shutdown_process(process)
        print("모든 서버가 종료되었습니다.")

if __name__ == "__main__":
    main()