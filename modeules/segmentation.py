"""
이미지 세그멘테이션을 수행하는 모듈
"""
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from config import SEGFORMER_MODEL, DEVICE, COLOR_MAP, CLASS_MAP


class SegmentationModel:
    def __init__(self, model_name=SEGFORMER_MODEL):
        """
        세그멘테이션 모델 초기화
        """
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval()
        self.model.to(DEVICE)
        self.device = DEVICE
        self.class_map = CLASS_MAP
        self.color_map = COLOR_MAP
    
    def process_image(self, image_path):
        """
        이미지를 로드하고 세그멘테이션 처리
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            tuple: (원본 이미지, numpy 이미지, 세그멘테이션 결과)
        """
        # 이미지 로드
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # 세그멘테이션 수행
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 결과 처리
        logits = outputs.logits
        seg_map = logits.argmax(dim=1)[0].cpu().numpy()
        
        return image, image_np, seg_map
    
    def process_image_from_array(self, image_np):
        """
        numpy 배열 이미지를 세그멘테이션 처리
        
        Args:
            image_np: numpy 배열 형식의 이미지
            
        Returns:
            tuple: (PIL 이미지, numpy 이미지, 세그멘테이션 결과)
        """
        # PIL 이미지로 변환
        if image_np.shape[2] == 3:  # RGB
            image = Image.fromarray(image_np)
        else:  # BGR -> RGB 변환 필요
            image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        
        # 세그멘테이션 수행
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 결과 처리
        logits = outputs.logits
        seg_map = logits.argmax(dim=1)[0].cpu().numpy()
        
        return image, image_np, seg_map
    
    def create_overlay(self, image_np, seg_map, alpha=0.5):
        """
        세그멘테이션 결과를 오버레이하여 시각화
        
        Args:
            image_np: numpy 형식의 이미지
            seg_map: 세그멘테이션 맵
            alpha: 오버레이 투명도 (0-1)
            
        Returns:
            numpy array: 오버레이된 이미지
        """
        # 클래스 ID별 색상 매핑
        color_map_img = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        
        for class_name, class_id in self.class_map.items():
            if class_name in self.color_map:
                color_map_img[seg_map == class_id] = self.color_map[class_name]
        
        # 원본 이미지 크기로 리사이즈
        h, w = image_np.shape[:2]
        color_map_resized = cv2.resize(color_map_img, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 오버레이 적용
        blended = cv2.addWeighted(image_np, 1 - alpha, color_map_resized, alpha, 0)
        
        return blended, color_map_resized
    
    def save_overlay(self, blended_image, output_path):
        """
        오버레이 이미지 저장
        
        Args:
            blended_image: 오버레이된 이미지
            output_path: 저장 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        return cv2.imwrite(output_path, cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR))
    
    def get_class_map(self):
        """
        클래스 매핑 정보 반환
        
        Returns:
            dict: 클래스 매핑 정보
        """
        return self.class_map