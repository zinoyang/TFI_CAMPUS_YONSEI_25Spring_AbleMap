"""
LLM API와 통신하는 모듈 - 한국어 응답 버전
"""
import requests
import json
import base64
import time
import mimetypes
import io
from datetime import datetime
from PIL import Image
from config import LLM_API_KEY, API_MAX_RETRIES

# 타임아웃 값을 직접 정의
API_REQUEST_TIMEOUT = 120  # 120초로 설정

class LLMAnalyzer:
    def __init__(self, api_key=LLM_API_KEY):
        """
        LLM 분석기 초기화
        """
        self.api_key = api_key
        self.api_url = ""
        self.model = ""
    
    def create_prompt(self, accessibility_info, facility_info=None):
        """
        LLM에 전달할 프롬프트 생성

        Args:
            accessibility_info: 접근성 분석 정보
            facility_info: 장애인편의시설 정보 (선택적)

        Returns:
            str: 프롬프트 문자열
        """
        prompt = f"""
    다음은 건물 외부 접근성 분석 결과입니다:

    - 계단 존재 여부: {accessibility_info.get('has_stairs', False)}
    - 경사로 존재 여부: {accessibility_info.get('has_ramp', False)}
    - 입구 접근 가능 여부: {accessibility_info.get('entrance_accessible', True)}
    - 감지된 장애물: {', '.join(accessibility_info.get('obstacles', [])) if accessibility_info.get('obstacles') else '없음'}
    - 보도 존재 여부: {accessibility_info.get('has_sidewalk', False)}
    """

        # 세부 장애물 정보 포함
        if 'obstacle_details' in accessibility_info:
            prompt += "\n세부 장애물 정보:\n"
            for obj, details in accessibility_info['obstacle_details'].items():
                prompt += f"- {obj}: {json.dumps(details, ensure_ascii=False)}\n"

        # 외부 접근성 점수 추가
        if 'accessibility_score' in accessibility_info:
            prompt += f"\n건물 외부 접근성 점수 (external_accessibility_score): {accessibility_info['accessibility_score']}/10\n"
            prompt += """
    ※ 이 점수는 이미지 세그멘테이션 결과를 기반으로 자동 산정되었습니다.
    다음 요소들이 반영되어 있습니다:
    - 계단의 존재 여부 및 위치
    - 출입구(문)의 너비
    - 인도와 출입구 간 연결 거리
    - 계단 난간 유무

    LLM은 이 외부 점수를 참고하되, 내부 접근성(시설 정보 기반)을 중심으로 internal_accessibility_score를 산정해주세요.
    """

        # 공공데이터 기반 장애인편의시설 정보 포함
        if facility_info:
            prompt += "\n장애인편의시설 공공데이터 정보:\n"

            if facility_info.get("available", False):
                # 기본 정보
                if facility_info.get("basic_info"):
                    basic = facility_info["basic_info"]
                    prompt += f"- 시설명: {basic.get('faclNm', '정보 없음')}\n"
                    prompt += f"- 주소: {basic.get('lcMnad', '정보 없음')}\n"
                    prompt += f"- 설립일: {basic.get('estbDate', '정보 없음')}\n"

                # 기능 정보
                if facility_info.get("facility_features") and facility_info["facility_features"].get("evalInfo"):
                    prompt += "\n시설 기능:\n"
                    for feat in facility_info["facility_features"]["evalInfo"]:
                        prompt += f"- {feat}\n"

                # 접근성 세부 정보
                if facility_info.get("accessibility_details"):
                    details = facility_info["accessibility_details"]
                    if details.get("entrance"):
                        prompt += f"\n입구 접근성: {'접근 가능' if details['entrance'].get('accessible', False) else '제한됨'}\n"
                        prompt += "입구 특징: " + ", ".join(details["entrance"].get("features", [])) + "\n"
                    if details.get("parking"):
                        prompt += f"장애인 주차: {'있음' if details['parking'].get('available', False) else '없음'}\n"
                        prompt += "주차 특징: " + ", ".join(details["parking"].get("features", [])) + "\n"
                    if details.get("restroom"):
                        prompt += f"장애인 화장실: {'있음' if details['restroom'].get('available', False) else '없음'}\n"
                        prompt += "화장실 특징: " + ", ".join(details["restroom"].get("features", [])) + "\n"
                    if details.get("elevator"):
                        prompt += f"엘리베이터: {'있음' if details['elevator'].get('available', False) else '없음 또는 정보 없음'}\n"
            else:
                prompt += f"- {facility_info.get('message', '시설 정보를 찾을 수 없습니다.')}\n"

        # 내부 점수 산정 기준 설명 및 최종 점수 계산 안내
        prompt += """

    내부 접근성 점수 (internal_accessibility_score)는 아래 항목 기반으로 총 10점 만점으로 산정해주세요:

    [주출입구 관련 총 3점]
    - 주출입구 접근로: 1점
    - 주출입구 높이차이 제거: 1점
    - 주출입구(문): 1점

    [장애인 화장실 관련 총 2점]
    - 장애인사용가능화장실: 2점

    [엘리베이터 관련 총 2점]
    - 승강기: 2점

    [기타 항목 총 3점]
    - 장애인전용주차구역: 1점
    - 장애인사용가능객실: 1점
    - 유도 및 안내 설비: 1점

    또한 다음 규칙에 따라 최종 접근성 점수 (final_accessibility_score)를 계산해주세요:

    - 외부 접근성 점수 (external_accessibility_score): 40%
    - 내부 접근성 점수 (internal_accessibility_score): 60%
    - 계산 결과는 소수점 첫째 자리에서 반올림합니다 (예: 7.6 → 8)

    다음 JSON 형식으로 결과를 한국어로 제공해주세요:

    {
    "external_accessibility_score": 1-10,
    "internal_accessibility_score": 1-10,
    "final_accessibility_score": 1-10,
    "stairs_count": 추정 계단 수,
    "stairs_height": "추정 높이 설명",
    "alternative_route": true/false,
    "alternative_route_description": "설명",
    "recommendations": ["조언1", ...],
    "observations": ["관찰1", ...],
    "improvement_suggestions": ["제안1", ...]
    }
    """

        return prompt

    
    def analyze_image(self, image_path, overlay_path, accessibility_info, facility_info=None):
        """
        이미지와 접근성 정보를 LLM으로 분석
        
        Args:
            image_path: 원본 이미지 경로
            overlay_path: 오버레이 이미지 경로
            accessibility_info: 접근성 분석 정보
            facility_info: 장애인편의시설 정보 (선택적)
            
        Returns:
            dict: LLM 분석 결과
        """
        prompt = self.create_prompt(accessibility_info, facility_info)
        
        # 이미지 인코딩 (최적화 함수 사용)
        original_image_b64, original_mime = self.optimize_image_for_api(image_path)
        overlay_image_b64, overlay_mime = self.optimize_image_for_api(overlay_path)
        
        if not original_image_b64 or not overlay_image_b64:
            return {"error": "이미지 인코딩 실패"}
            
        # API 요청 준비
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": original_mime,
                                "data": original_image_b64
                            }
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": overlay_mime,
                                "data": overlay_image_b64
                            }
                        }
                    ]
                }
            ],
            "system": "당신은 한국의 장애인 접근성 평가 전문가입니다. 제시된 평가 기준에 따라 이미지와 데이터를 바탕으로 정확하고 객관적인 접근성 점수를 산정합니다. 모든 응답은 반드시 한국어로 제공해야 합니다."
        }
        
        try:
            # 재시도 메커니즘 적용
            retries = 0
            while retries < API_MAX_RETRIES:
                try:
                    print(f"API 요청 시도 중... (타임아웃: {API_REQUEST_TIMEOUT}초)")
                    start_time = time.time()
                    response = requests.post(self.api_url, headers=headers, json=data, timeout=API_REQUEST_TIMEOUT)
                    response.raise_for_status()
                    result = response.json()
                    end_time = time.time()
                    print(f"API 요청 완료: {end_time - start_time:.2f}초 소요")
                    return self._parse_llm_response(result["content"][0]["text"])
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    retries += 1
                    print(f"API 요청 실패 ({e}), 재시도 {retries}/{API_MAX_RETRIES}")
                    if retries == API_MAX_RETRIES:
                        return {"error": f"최대 재시도 횟수 초과: {str(e)}"}
                    # 재시도 간격 증가 (지수 백오프)
                    wait_time = 2 ** retries
                    print(f"{wait_time}초 후 재시도합니다...")
                    time.sleep(wait_time)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # 요청 한도 초과
                        retries += 1
                        wait_time = int(e.response.headers.get('Retry-After', 60))
                        print(f"API 요청 제한 초과, {wait_time}초 후 재시도 {retries}/{API_MAX_RETRIES}")
                        if retries == API_MAX_RETRIES:
                            return {"error": "API 요청 제한 초과"}
                        time.sleep(wait_time)
                    else:
                        print(f"API 응답 내용: {e.response.text}")  # 디버깅을 위해 응답 내용 출력
                        return {"error": f"HTTP 오류: {e.response.status_code} - {str(e)}"}
                except Exception as e:
                    print(f"예상치 못한 오류: {str(e)}")
                    return {"error": f"API 요청 중 오류 발생: {str(e)}"}
        except Exception as e:
            return {"error": f"분석 처리 중 오류: {str(e)}"}
    
    # 이미지 최적화 및 인코딩 함수는 원래 코드와 동일하게 유지
    def optimize_image_for_api(self, image_path, max_size=(1024, 1024)):
        """
        API 전송용으로 이미지 크기 최적화
        
        Args:
            image_path: 이미지 파일 경로
            max_size: 최대 이미지 크기 (가로, 세로)
            
        Returns:
            tuple: (base64로 인코딩된 이미지, MIME 타입)
        """
        try:
            # MIME 타입 감지
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith('image/'):
                mime_type = 'image/jpeg'  # 기본값으로 jpeg 사용
            
            # 이미지 로드 및 크기 최적화
            img = Image.open(image_path)
            img.thumbnail(max_size, Image.LANCZOS)
            
            # 메모리에 이미지 저장
            buffer = io.BytesIO()
            img_format = 'JPEG' if mime_type == 'image/jpeg' else 'PNG'
            img.save(buffer, format=img_format)
            buffer.seek(0)
            
            return base64.b64encode(buffer.read()).decode('utf-8'), mime_type
        except Exception as e:
            print(f"이미지 최적화 오류: {str(e)}")
            # 오류 시 기존 방식으로 인코딩
            return self.encode_image_to_base64(image_path)
    
    def encode_image_to_base64(self, image_path):
        """
        이미지를 base64로 인코딩하고 MIME 타입 반환
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            tuple: (base64로 인코딩된 이미지, MIME 타입)
        """
        try:
            # MIME 타입 감지
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith('image/'):
                mime_type = 'image/jpeg'  # 기본값으로 jpeg 사용
            
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8'), mime_type
        except Exception as e:
            print(f"이미지 인코딩 오류: {str(e)}")
            return None, None
    
    def _parse_llm_response(self, response_text):
        """
        LLM 응답에서 JSON 파싱
        
        Args:
            response_text: LLM 응답 텍스트
            
        Returns:
            dict: 파싱된 JSON 데이터
        """
        try:
            # JSON 형식 텍스트 추출
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # JSON을 찾을 수 없는 경우 텍스트 그대로 반환
                return {"text_response": response_text.strip()}
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 텍스트 그대로 반환
            return {"text_response": response_text.strip()}