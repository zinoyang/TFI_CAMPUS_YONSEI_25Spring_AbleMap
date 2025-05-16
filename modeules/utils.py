"""
유틸리티 함수 모음
"""
import os
import json
import re
import time
import logging
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
try:
    from PIL.ExifTags import TAGS, GPSTAGS
except ImportError:
    TAGS, GPSTAGS = {}, {}

from config import REPORTS_DIR, OVERLAY_DIR

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("accessibility_analyzer.log", mode='a')
    ]
)
logger = logging.getLogger("AccessibilityAnalyzer")

def get_file_name(file_path):
    """
    파일 경로에서 확장자 없는 파일명 추출
    
    Args:
        file_path: 파일 경로
        
    Returns:
        str: 파일명 (확장자 제외)
    """
    return Path(file_path).stem

def generate_output_paths(image_path, base_dir=None):
    """
    입력 이미지 경로로부터 출력 파일 경로 생성
    
    Args:
        image_path: 이미지 파일 경로
        base_dir: 기본 출력 디렉토리 (None이면 기본값 사용)
        
    Returns:
        dict: 출력 파일 경로
    """
    file_name = get_file_name(image_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    overlay_dir = base_dir or OVERLAY_DIR
    report_dir = REPORTS_DIR if base_dir is None else base_dir
    
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    return {
        "overlay": os.path.join(overlay_dir, f"{file_name}_overlay_{timestamp}.png"),
        "report": os.path.join(report_dir, f"{file_name}_report_{timestamp}.json")
    }

def save_report(data, output_path):
    """
    분석 결과를 JSON 파일로 저장
    
    Args:
        data: 저장할 데이터
        output_path: 출력 파일 경로
        
    Returns:
        str: 저장된 파일 경로
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return output_path
    except Exception as e:
        logger.error(f"보고서 저장 오류: {str(e)}")
        return None

def extract_location_from_image(image_path):
    """
    이미지 메타데이터에서 위치 정보 추출 (가능한 경우)
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        dict: 위치 정보
    """
    location_info = {
        "latitude": None,
        "longitude": None,
        "address": None,
        "building_name": None,
        "siDoNm": None,
        "cggNm": None,
        "faclNm": None
    }
    
    try:
        # 이미지 메타데이터에서 GPS 정보 추출 시도
        image = Image.open(image_path)
        
        if hasattr(image, '_getexif') and image._getexif() is not None:
            exif_data = image._getexif()
            
            if exif_data:
                labeled_exif = {}
                for (key, val) in exif_data.items():
                    labeled_exif[TAGS.get(key, key)] = val
                
                # GPS 정보 추출
                if 'GPSInfo' in labeled_exif:
                    gps_info = {}
                    for key in labeled_exif['GPSInfo'].keys():
                        gps_info[GPSTAGS.get(key, key)] = labeled_exif['GPSInfo'][key]
                    
                    # 위도 계산
                    if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info:
                        lat_data = gps_info['GPSLatitude']
                        lat_ref = gps_info['GPSLatitudeRef']
                        lat = _convert_to_degrees(lat_data)
                        if lat_ref != 'N':
                            lat = -lat
                        location_info['latitude'] = lat
                    
                    # 경도 계산
                    if 'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:
                        lon_data = gps_info['GPSLongitude']
                        lon_ref = gps_info['GPSLongitudeRef']
                        lon = _convert_to_degrees(lon_data)
                        if lon_ref != 'E':
                            lon = -lon
                        location_info['longitude'] = lon
        
        # 파일명에서 위치 정보와 시설명 추출
        file_name = get_file_name(image_path)
        # 시도명_시군구명_도로명_번호 형식으로 되어있는지 확인
        parts = file_name.split('_')
        if len(parts) >= 3:
            # 공백 제거 및 값 할당
            location_info['siDoNm'] = parts[0].strip()  # 시도명
            location_info['cggNm'] = parts[1].strip()   # 시군구명
            # 도로명과 번지를 합쳐서 저장
            road_parts = parts[2:]
            location_info['roadNm'] = '_'.join(road_parts).strip()  # 도로명과 번지
            location_info['building_name'] = None  # building_name은 더 이상 사용하지 않음
        
    except Exception as e:
        logger.warning(f"위치 정보 추출 오류: {str(e)}")
    
    # 위치 정보가 없는 경우 None 반환
    if location_info['latitude'] is None and location_info['siDoNm'] is None:
        return None
    
    return location_info

def _convert_to_degrees(value):
    """
    GPS 좌표를 도(degree) 단위로 변환
    
    Args:
        value: (degree, minute, second) 형식의 GPS 좌표
        
    Returns:
        float: 도(degree) 단위 좌표
    """
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])
    return d + (m / 60.0) + (s / 3600.0)

def resize_image(image_path, max_size=1024):
    """
    이미지 크기 조정
    
    Args:
        image_path: 이미지 파일 경로
        max_size: 최대 크기 (픽셀)
        
    Returns:
        numpy.ndarray: 크기 조정된 이미지
    """
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    h, w = img.shape[:2]
    
    # 최대 크기 초과 시 리사이징
    if max(h, w) > max_size:
        ratio = max_size / max(h, w)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return img

def measure_execution_time(func):
    """
    함수 실행 시간 측정을 위한 데코레이터
    
    Args:
        func: 측정할 함수
    
    Returns:
        wrapper: 시간 측정 기능이 추가된 함수
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} 실행 시간: {execution_time:.2f}초")
        return result
    return wrapper

def create_directory_structure():
    """
    프로젝트에 필요한 디렉토리 구조 생성
    """
    # 모듈 디렉토리
    os.makedirs('modules', exist_ok=True)
    
    # 데이터 디렉토리
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/images', exist_ok=True)
    os.makedirs('data/results/overlays', exist_ok=True)
    os.makedirs('data/results/reports', exist_ok=True)
    
    # 캐시 디렉토리
    os.makedirs('cache', exist_ok=True)
    
    # 테스트 디렉토리
    os.makedirs('tests', exist_ok=True)
    
    logger.info("디렉토리 구조가 생성되었습니다.")

def validate_image(image_path):
    """
    이미지 파일 유효성 검사
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        bool: 유효한 이미지 파일 여부
    """
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception as e:
        logger.warning(f"이미지 유효성 검사 실패: {image_path} - {str(e)}")
        return False

def get_image_dimensions(image_path):
    """
    이미지 크기 정보 얻기
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        tuple: (너비, 높이) 또는 None (에러 시)
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        logger.warning(f"이미지 크기 정보 읽기 실패: {image_path} - {str(e)}")
        return None

def load_json_file(file_path):
    """
    JSON 파일 로드
    
    Args:
        file_path: JSON 파일 경로
        
    Returns:
        dict: 로드된 JSON 데이터 또는 None (에러 시)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"JSON 파일 로드 실패: {file_path} - {str(e)}")
        return None

def format_timestamp(timestamp=None):
    """
    타임스탬프 포맷팅
    
    Args:
        timestamp: 타임스탬프 (None이면 현재 시간)
        
    Returns:
        str: 포맷된 타임스탬프 문자열
    """
    if timestamp is None:
        timestamp = datetime.now()
    elif isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            timestamp = datetime.now()
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def clean_filename(filename):
    """
    파일명에서 유효하지 않은 문자 제거
    
    Args:
        filename: 원본 파일명
        
    Returns:
        str: 정리된 파일명
    """
    # 파일 시스템에서 사용할 수 없는 문자 제거
    invalid_chars = r'[<>:"/\\|?*]'
    clean_name = re.sub(invalid_chars, '_', filename)
    
    # 공백 및 마침표 정리
    clean_name = clean_name.strip().rstrip('.')
    
    # 파일명이 비어있는 경우 기본값 설정
    if not clean_name:
        clean_name = "untitled"
        
    return clean_name

def get_supported_image_extensions():
    """
    지원되는 이미지 확장자 목록 반환
    
    Returns:
        list: 지원되는 이미지 확장자 목록
    """
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

def is_image_file(file_path):
    """
    파일이 이미지인지 확인
    
    Args:
        file_path: 파일 경로
        
    Returns:
        bool: 이미지 파일 여부
    """
    ext = os.path.splitext(file_path)[1].lower()
    return ext in get_supported_image_extensions()

def get_image_files_in_directory(directory):
    """
    디렉토리에서 모든 이미지 파일 목록 반환
    
    Args:
        directory: 디렉토리 경로
        
    Returns:
        list: 이미지 파일 경로 목록
    """
    if not os.path.isdir(directory):
        logger.error(f"디렉토리를 찾을 수 없음: {directory}")
        return []
    
    image_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_image_file(file_path):
                image_files.append(file_path)
    
    return image_files