"""
외부 API와 통신하는 모듈 (FastAPI 통합 업데이트)
"""
import requests
import json
import time
from datetime import datetime
"""
from config import (
    ACCESSIBILITY_API_KEY, ACCESSIBILITY_API_ENDPOINT, 
    API_REQUEST_TIMEOUT, API_MAX_RETRIES,
    USE_FASTAPI, FASTAPI_ENDPOINT, FASTAPI_API_KEY
)
"""
from config import ( 
    API_REQUEST_TIMEOUT, API_MAX_RETRIES,
    USE_FASTAPI, FASTAPI_ENDPOINT, FASTAPI_API_KEY
)

class APIClient:
    def __init__(self, api_key=None, endpoint=None):
        """
        API 클라이언트 초기화
        
        Args:
            api_key: API 키 (None이면 설정에서 가져옴)
            endpoint: API 엔드포인트 (None이면 설정에서 가져옴)
        """
        # FastAPI 사용 여부에 따라 설정 결정
        if USE_FASTAPI:
            self.api_key = api_key or FASTAPI_API_KEY
            self.endpoint = endpoint or FASTAPI_ENDPOINT
            self.use_fastapi = True
        else:
            self.api_key = api_key or ACCESSIBILITY_API_KEY
            self.endpoint = endpoint or ACCESSIBILITY_API_ENDPOINT
            self.use_fastapi = False
    
    def send_accessibility_data(self, location_info, accessibility_info, facility_info=None, llm_analysis=None, image_path=None, overlay_path=None):
        """
        접근성 데이터를 API로 전송
        
        Args:
            location_info: 위치 정보
            accessibility_info: 접근성 분석 정보
            facility_info: 장애인편의시설 정보 (선택적)
            llm_analysis: LLM 분석 결과 (선택적)
            image_path: 원본 이미지 경로 (선택적)
            overlay_path: 오버레이 이미지 경로 (선택적)
            
        Returns:
            dict: API 응답
        """
        # API 요청 데이터 구성
        data = {
            "location": location_info,
            "accessibility": accessibility_info,
            "timestamp": datetime.now().isoformat()
        }
        
        # 선택적 정보 추가
        if facility_info:
            data["facility"] = facility_info
        
        if llm_analysis:
            data["ai_analysis"] = llm_analysis
            
        if image_path:
            data["image_path"] = image_path
            
        if overlay_path:
            data["overlay_path"] = overlay_path
        
        # API 요청 헤더
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 재시도 메커니즘을 적용한 API 요청
        return self._send_request_with_retry(self.endpoint, headers, data)
    
    def _send_request_with_retry(self, url, headers, data, max_retries=API_MAX_RETRIES, timeout=API_REQUEST_TIMEOUT):
        """
        재시도 메커니즘이 적용된 API 요청
        
        Args:
            url: 요청 URL
            headers: 요청 헤더
            data: 요청 데이터
            max_retries: 최대 재시도 횟수
            timeout: 타임아웃 시간(초)
            
        Returns:
            dict: API 응답
        """
        retries = 0
        
        while retries < max_retries:
            try:
                response = requests.post(url, headers=headers, json=data, timeout=timeout)
                
                # 성공 응답 처리
                if response.status_code == 200 or response.status_code == 201:
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        return {"status": "success", "message": "데이터 전송 성공 (JSON 응답 없음)"}
                
                # 오류 응답 처리
                if response.status_code == 429:  # Too Many Requests
                    retries += 1
                    wait_time = int(response.headers.get('Retry-After', 5))
                    print(f"API 요청 제한 초과, {wait_time}초 후 재시도 ({retries}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code >= 500:  # 서버 오류
                    retries += 1
                    wait_time = 2 ** retries  # 지수 백오프
                    print(f"서버 오류 발생 ({response.status_code}), {wait_time}초 후 재시도 ({retries}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                
                # 4xx 오류는 재시도 없이 바로 반환
                return {
                    "error": True,
                    "status_code": response.status_code,
                    "message": f"API 오류: {response.reason}",
                    "response": response.text
                }
                
            except requests.exceptions.Timeout:
                retries += 1
                wait_time = 2 ** retries
                print(f"API 요청 타임아웃, {wait_time}초 후 재시도 ({retries}/{max_retries})...")
                time.sleep(wait_time)
            
            except requests.exceptions.ConnectionError:
                retries += 1
                wait_time = 3 ** retries
                print(f"API 연결 오류, {wait_time}초 후 재시도 ({retries}/{max_retries})...")
                time.sleep(wait_time)
            
            except Exception as e:
                return {"error": True, "message": f"API 요청 중 오류 발생: {str(e)}"}
        
        # 최대 재시도 횟수 초과
        return {"error": True, "message": "최대 재시도 횟수를 초과하여 API 요청에 실패했습니다."}
    
    def test_connection(self):
        """
        API 연결 테스트
        
        Returns:
            bool: 연결 성공 여부
        """
        test_data = {
            "test": True,
            "timestamp": datetime.now().isoformat()
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        test_endpoint = f"{self.endpoint.rstrip('/')}/ping" if self.use_fastapi else f"{self.endpoint.rstrip('/')}/ping"
        
        try:
            response = requests.post(test_endpoint, headers=headers, json=test_data, timeout=5)
            return response.status_code == 200
        except:
            return False