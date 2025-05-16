"""
접근성 분석을 수행하는 모듈
"""
import numpy as np
from scipy.ndimage import binary_dilation
from config import CLASS_MAP, ACCESSIBILITY_THRESHOLD_DISTANCE

class AccessibilityAnalyzer:
    def __init__(self, class_map=CLASS_MAP):
        """
        접근성 분석기 초기화
        """
        self.class_map = class_map
        self.threshold_distance = ACCESSIBILITY_THRESHOLD_DISTANCE
    
    def analyze(self, seg_map):
        """
        세그멘테이션 맵에서 접근성 정보 분석
        
        Args:
            seg_map: 세그멘테이션 맵
            
        Returns:
            dict: 접근성 정보
        """
        # 기본 접근성 정보 초기화
        accessibility_info = {
            'has_stairs': False,
            'has_ramp': False,  # 현재 데이터셋에 ramp 클래스가 없으므로 항상 False
            'entrance_accessible': True,
            'obstacles': [],
            'obstacle_details': {}
        }
        
        # 계단 감지
        stairs_mask = seg_map == self.class_map['stairs']
        if np.any(stairs_mask):
            accessibility_info['has_stairs'] = True
            accessibility_info['obstacles'].append('stairs')
            
            # 계단 크기 분석 (픽셀 수로 상대적인 크기 추정)
            stairs_pixels = np.count_nonzero(stairs_mask)
            total_pixels = seg_map.size
            stairs_ratio = stairs_pixels / total_pixels
            accessibility_info['obstacle_details']['stairs'] = {
                'pixel_count': int(stairs_pixels),
                'ratio': float(stairs_ratio),
                'estimated_size': self._estimate_size(stairs_ratio)
            }
        
        # 문 감지 및 분석
        door_mask = seg_map == self.class_map['door']
        if np.any(door_mask):
            # 문 크기 분석
            door_pixels = np.count_nonzero(door_mask)
            total_pixels = seg_map.size
            door_ratio = door_pixels / total_pixels
            accessibility_info['has_door'] = True
            accessibility_info['obstacle_details']['door'] = {
                'pixel_count': int(door_pixels),
                'ratio': float(door_ratio),
                'estimated_width': self._estimate_door_width(door_ratio)
            }
            
            # 문과 계단의 관계 분석
            if accessibility_info['has_stairs']:
                min_distance = self._calculate_object_distance(stairs_mask, door_mask)
                accessibility_info['obstacle_details']['stairs_to_door_distance'] = float(min_distance)
                
                # 문 앞에 계단이 있는지 분석
                if min_distance < self.threshold_distance:
                    accessibility_info['entrance_accessible'] = False
                    accessibility_info['obstacles'].append('stairs_at_entrance')
        
        # 인도 감지
        sidewalk_mask = seg_map == self.class_map['sidewalk']
        if np.any(sidewalk_mask):
            accessibility_info['has_sidewalk'] = True
            
            # 인도와 입구의 관계 분석
            if accessibility_info.get('has_door', False):
                sidewalk_to_door = self._calculate_object_distance(sidewalk_mask, door_mask)
                accessibility_info['obstacle_details']['sidewalk_to_door_distance'] = float(sidewalk_to_door)
                
                # 인도에서 문까지 연결성 분석
                if sidewalk_to_door > self.threshold_distance:
                    accessibility_info['obstacles'].append('disconnected_sidewalk')
        
        # 건물 감지
        building_mask = seg_map == self.class_map['building']
        if np.any(building_mask):
            accessibility_info['has_building'] = True
            
            # 건물 크기 분석
            building_pixels = np.count_nonzero(building_mask)
            total_pixels = seg_map.size
            building_ratio = building_pixels / total_pixels
            accessibility_info['obstacle_details']['building'] = {
                'pixel_count': int(building_pixels),
                'ratio': float(building_ratio)
            }
        
        # 난간 감지 (계단용)
        railing_mask = seg_map == self.class_map['railing']
        if np.any(railing_mask):
            accessibility_info['has_railing'] = True
            
            # 난간이 계단 근처에 있는지 확인
            if accessibility_info['has_stairs']:
                railing_to_stairs = self._calculate_object_distance(railing_mask, stairs_mask)
                accessibility_info['obstacle_details']['railing_to_stairs_distance'] = float(railing_to_stairs)
                
                # 계단에 난간이 있으면 접근성 향상
                if railing_to_stairs < self.threshold_distance:
                    accessibility_info['has_stairs_railing'] = True
        
        # 종합적인 접근성 판단
        accessibility_score = self._calculate_accessibility_score(accessibility_info)
        accessibility_info['accessibility_score'] = accessibility_score
        
        return accessibility_info
    
    def _calculate_object_distance(self, mask1, mask2):
        """
        두 객체 마스크 간의 최소 거리 계산
        """
        # 두 마스크의 픽셀 좌표 얻기
        y1, x1 = np.where(mask1)
        y2, x2 = np.where(mask2)
        
        if len(y1) == 0 or len(y2) == 0:
            return float('inf')
        
        # 계산량을 줄이기 위해 각 마스크에서 최대 100개 픽셀만 샘플링
        max_samples = min(100, len(y1), len(y2))
        indices1 = np.random.choice(len(y1), max_samples) if len(y1) > max_samples else np.arange(len(y1))
        indices2 = np.random.choice(len(y2), max_samples) if len(y2) > max_samples else np.arange(len(y2))
        
        y1_sample, x1_sample = y1[indices1], x1[indices1]
        y2_sample, x2_sample = y2[indices2], x2[indices2]
        
        # 최소 거리 계산
        min_dist = float('inf')
        for i in range(len(y1_sample)):
            for j in range(len(y2_sample)):
                dist = np.sqrt((y1_sample[i] - y2_sample[j])**2 + (x1_sample[i] - x2_sample[j])**2)
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _estimate_size(self, ratio):
        """
        픽셀 비율에 따른 상대적 크기 추정
        """
        if ratio < 0.01:
            return "very small"
        elif ratio < 0.05:
            return "small"
        elif ratio < 0.15:
            return "medium"
        elif ratio < 0.3:
            return "large"
        else:
            return "very large"
    
    def _estimate_door_width(self, ratio):
        """
        문의 상대적 너비 추정 및 휠체어 통과 가능성 판단
        """
        if ratio < 0.01:
            return "narrow"
        elif ratio < 0.03:
            return "standard"
        else:
            return "wide"
    
    def _calculate_accessibility_score(self, accessibility_info):
        """
        종합적인 접근성 점수 계산 (0-10)
        """
        score = 10  # 기본 점수
        
        # 계단 관련
        if 'stairs_at_entrance' in accessibility_info['obstacles']:
            # 입구 바로 앞에 계단이 있으면 큰 감점
            score -= 5
            
            # 난간이 있으면 약간 점수 보상
            if accessibility_info.get('has_stairs_railing', False):
                score += 1
        elif accessibility_info['has_stairs']:
            # 계단이 있지만 입구에서 떨어져 있으면 작은 감점
            score -= 2
        
        # 인도 연결성
        if 'disconnected_sidewalk' in accessibility_info['obstacles']:
            score -= 2
        
        # 문 너비
        if accessibility_info.get('has_door', False):
            door_width = accessibility_info['obstacle_details']['door']['estimated_width']
            if door_width == "narrow":
                score -= 3
            elif door_width == "wide":
                score += 1
        
        # 최종 점수 범위 조정
        return max(1, min(10, score))
    
    def get_accessibility_explanation(self, accessibility_info):
        """
        접근성 점수에 대한 설명 생성
        
        Args:
            accessibility_info: 접근성 정보
            
        Returns:
            str: 접근성 설명
        """
        score = accessibility_info.get('accessibility_score', 0)
        
        if score >= 9:
            explanation = "매우 높은 접근성: 휠체어 사용자가 쉽게 접근 가능한 환경입니다."
        elif score >= 7:
            explanation = "좋은 접근성: 휠체어 사용자가 대부분 접근 가능하나 약간의 불편함이 있을 수 있습니다."
        elif score >= 5:
            explanation = "보통 접근성: 휠체어 사용자가 접근 가능하나 일부 도움이 필요할 수 있습니다."
        elif score >= 3:
            explanation = "낮은 접근성: 휠체어 사용자는 접근에 상당한 어려움이 있을 수 있습니다."
        else:
            explanation = "매우 낮은 접근성: 휠체어 사용자는 도움 없이 접근하기 어렵습니다."
        
        return explanation