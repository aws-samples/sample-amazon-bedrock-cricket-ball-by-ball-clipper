"""
OCR Processor Module
Handles text detection and extraction from video frames using EasyOCR
Located in src/text_extract/ to emphasize its text extraction functionality
"""

import cv2
import easyocr
import numpy as np
import time
from typing import List, Dict, Tuple, Optional


class OCRProcessor:
    """Handles OCR processing with optimization strategies"""
    
    def __init__(self, languages: List[str] = ['en'], gpu: bool = True, ocr_region: dict = None):
        """
        Initialize OCR processor
        
        Args:
            languages: Languages for OCR recognition
            gpu: Whether to use GPU acceleration
            ocr_region: OCR region configuration dict with 'type', 'x', 'y', 'width', 'height'
                       type can be 'fraction' (0.0-1.0) or 'pixels' (absolute coordinates)
        """
        self.languages = languages
        self.gpu = gpu
        self.ocr_region = ocr_region
        
        print("Setting up EasyOCR...")
        if ocr_region:
            region_type = ocr_region.get('type', 'fraction')
            x, y, w, h = ocr_region['x'], ocr_region['y'], ocr_region['width'], ocr_region['height']
            print(f"🎯 OCR Region configured ({region_type}): x={x}, y={y}, width={w}, height={h}")
        else:
            print("🖼️  OCR will process full frame")
        
        import torch
        if torch.backends.mps.is_available() and gpu:
            print("Apple M3 Metal GPU detected - using MPS acceleration")
        
        self.reader = easyocr.Reader(languages, gpu=gpu)
        
        # Pre-warm OCR model
        print("Pre-warming OCR model...")
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _ = self.reader.readtext(dummy_image)
        
        # Optimization tracking
        self.bbox_match_skips = 0
        self.text_cache_skips = 0
        self.last_ocr_texts = None
        self.over_bounding_box = None
        
        print("OCR Processor initialized successfully!")
    
    def _crop_to_region(self, frame: np.ndarray) -> tuple:
        """
        Crop frame to specified OCR region
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (cropped_frame, crop_info) where crop_info contains offset coordinates
        """
        if not self.ocr_region:
            # No region specified, return original frame
            return frame, None
        
        height, width = frame.shape[:2]
        region_type = self.ocr_region.get('type', 'fraction')
        
        if region_type == 'fraction':
            # Convert fractions to pixel coordinates
            x = int(self.ocr_region['x'] * width)
            y = int(self.ocr_region['y'] * height)
            crop_width = int(self.ocr_region['width'] * width)
            crop_height = int(self.ocr_region['height'] * height)
        elif region_type == 'pixels':
            # Use pixel coordinates directly
            x = self.ocr_region['x']
            y = self.ocr_region['y']
            crop_width = self.ocr_region['width']
            crop_height = self.ocr_region['height']
            
            # Ensure coordinates are within frame bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            crop_width = min(crop_width, width - x)
            crop_height = min(crop_height, height - y)
        else:
            print(f"⚠️  Unknown region type: {region_type}, using full frame")
            return frame, None
        
        # Crop the frame
        x_end = x + crop_width
        y_end = y + crop_height
        cropped_frame = frame[y:y_end, x:x_end]
        
        # Store crop information for coordinate adjustment
        crop_info = {
            'x_offset': x,
            'y_offset': y,
            'crop_width': crop_width,
            'crop_height': crop_height,
            'original_width': width,
            'original_height': height
        }
        
        print(f"🔍 CROPPED TO REGION: {width}x{height} → {crop_width}x{crop_height}")
        print(f"   📍 Crop region: ({x}, {y}) to ({x_end}, {y_end})")
        coverage_w = (crop_width / width) * 100
        coverage_h = (crop_height / height) * 100
        print(f"   📊 Coverage: {coverage_w:.1f}% width, {coverage_h:.1f}% height")
        
        return cropped_frame, crop_info
    
    def _adjust_bbox_coordinates(self, bbox, crop_info: dict) -> list:
        """
        Adjust bounding box coordinates from cropped frame back to original frame coordinates
        
        Args:
            bbox: Bounding box coordinates from cropped frame
            crop_info: Crop information containing offsets
            
        Returns:
            Adjusted bounding box coordinates for original frame
        """
        if crop_info is None:
            return bbox
        
        adjusted_bbox = []
        for point in bbox:
            # Add crop offsets to get original frame coordinates
            adjusted_x = point[0] + crop_info['x_offset']
            adjusted_y = point[1] + crop_info['y_offset']
            adjusted_bbox.append([adjusted_x, adjusted_y])
        
        return adjusted_bbox
    
    def extract_text_from_frame(self, frame: np.ndarray, confidence_threshold: float = 0.4) -> List[Tuple]:
        """
        Extract text from a video frame using OCR
        
        Args:
            frame: Input video frame
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            List of (bbox, text, confidence) tuples
        """
        # 🔍 CROP TO SPECIFIED REGION if configured
        processing_frame, crop_info = self._crop_to_region(frame)
        
        if crop_info:
            print(f"🎯 REGION OCR: Processing specified region only")
        else:
            print(f"🖼️  FULL FRAME OCR: Processing entire frame")
        
        # Resize frame for better OCR performance
        height, width = processing_frame.shape[:2]
        resize_factor = min(1920/width, 1080/height, 1.0)
        
        if resize_factor < 1.0:
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            resized_frame = cv2.resize(processing_frame, (new_width, new_height))
        else:
            resized_frame = processing_frame
        
        # Convert BGR to RGB for EasyOCR
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Perform OCR
        ocr_results = self.reader.readtext(
            rgb_frame,
            width_ths=0.7,
            height_ths=0.7,
            paragraph=False,
            detail=1
        )
        
        # Filter by confidence and adjust coordinates
        filtered_results = []
        scale_factor = 1.0 / resize_factor if resize_factor < 1.0 else 1.0
        
        for bbox, text, confidence in ocr_results:
            if confidence > confidence_threshold:
                # Scale bbox back to processing frame size first
                if scale_factor != 1.0:
                    scaled_bbox = [[int(x * scale_factor), int(y * scale_factor)] for x, y in bbox]
                else:
                    scaled_bbox = bbox
                
                # Adjust coordinates back to original frame if we cropped
                final_bbox = self._adjust_bbox_coordinates(scaled_bbox, crop_info)
                
                filtered_results.append((final_bbox, text, confidence))
        
        if crop_info:
            print(f"🎯 REGION OCR COMPLETE: Found {len(filtered_results)} text regions")
        else:
            print(f"🖼️  FULL FRAME OCR COMPLETE: Found {len(filtered_results)} text regions")
        
        return filtered_results
    
    def create_text_bbox_mapping(self, ocr_results: List[Tuple]) -> Dict[str, Dict]:
        """
        Create a mapping of extracted text to their bounding box information
        
        Args:
            ocr_results: OCR results with bounding boxes
            
        Returns:
            Dictionary mapping text to bbox info
        """
        text_bbox_map = {}
        
        for bbox, text, confidence in ocr_results:
            # Calculate dimensions
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            
            text_bbox_map[text.strip()] = {
                'bbox': bbox,
                'dimensions': {
                    'width': int(width),
                    'height': int(height),
                    'area': int(width * height)
                },
                'position': {
                    'top_left': (int(min(x_coords)), int(min(y_coords))),
                    'bottom_right': (int(max(x_coords)), int(max(y_coords))),
                    'center': (int(sum(x_coords)/len(x_coords)), int(sum(y_coords)/len(y_coords)))
                },
                'confidence': confidence
            }
        
        return text_bbox_map
    
    def check_text_similarity_cache(self, detected_texts: List[str]) -> bool:
        """
        Check if current OCR texts are similar to previous successful extraction
        
        Args:
            detected_texts: Current frame's detected texts
            
        Returns:
            True if texts are similar (can skip processing), False otherwise
        """
        if self.last_ocr_texts and set(detected_texts) == set(self.last_ocr_texts):
            self.text_cache_skips += 1
            return True
        return False
    
    def update_text_cache(self, detected_texts: List[str]):
        """
        Update the text similarity cache
        
        Args:
            detected_texts: Texts to cache
        """
        self.last_ocr_texts = detected_texts.copy()
    
    def check_over_bounding_box_match(self, text_bbox_map: Dict[str, Dict], 
                                     timestamp: float, frame_number: int) -> Optional[str]:
        """
        Check if stored over bounding box matches any text in current frame
        
        Args:
            text_bbox_map: Current frame's text-bbox mapping
            timestamp: Current video timestamp
            frame_number: Current frame number
            
        Returns:
            Over text if bbox matches, None otherwise
        """
        if not self.over_bounding_box:
            return None
        
        stored_over_text = self.over_bounding_box['over_text']
        stored_bbox_info = self.over_bounding_box['bbox_info']
        stored_dimensions = stored_bbox_info['dimensions']
        
        # Look for the same over text in current frame
        for text, bbox_info in text_bbox_map.items():
            if text == stored_over_text:
                current_dimensions = bbox_info['dimensions']
                
                # Check if dimensions match (with tolerance)
                width_diff = abs(current_dimensions['width'] - stored_dimensions['width'])
                height_diff = abs(current_dimensions['height'] - stored_dimensions['height'])
                
                # Allow 10% tolerance for dimension matching
                width_tolerance = stored_dimensions['width'] * 0.1
                height_tolerance = stored_dimensions['height'] * 0.1
                
                if width_diff <= width_tolerance and height_diff <= height_tolerance:
                    print(f"🎯 OVER BOUNDING BOX MATCH FOUND - SKIPPING AGENT ANALYSIS")
                    print(f"   📝 Over Text: '{text}'")
                    print(f"   📏 Stored Dimensions: {stored_dimensions['width']}x{stored_dimensions['height']}")
                    print(f"   📏 Current Dimensions: {current_dimensions['width']}x{current_dimensions['height']}")
                    
                    # Update the stored bounding box with current frame info
                    self.over_bounding_box['bbox_info'] = bbox_info
                    self.over_bounding_box['frame_number'] = frame_number
                    self.over_bounding_box['timestamp'] = timestamp
                    
                    self.bbox_match_skips += 1
                    return stored_over_text
        
        return None
    
    def store_over_bounding_box(self, over_text: str, text_bbox_map: Dict[str, Dict], 
                               frame_number: int, timestamp: float):
        """
        Store over bounding box information for future optimization
        
        Args:
            over_text: The over text to store
            text_bbox_map: Text-bbox mapping
            frame_number: Current frame number
            timestamp: Current timestamp
        """
        if over_text in text_bbox_map:
            self.over_bounding_box = {
                'over_text': over_text,
                'bbox_info': text_bbox_map[over_text],
                'frame_number': frame_number,
                'timestamp': timestamp
            }
            
            print(f"🎯 OVER BOUNDING BOX INFORMATION STORED:")
            print(f"   📝 Over Text: '{over_text}'")
            print(f"   📏 Dimensions: {text_bbox_map[over_text]['dimensions']['width']}x{text_bbox_map[over_text]['dimensions']['height']}")
            print(f"   📍 Position: {text_bbox_map[over_text]['position']['center']}")
    
    def get_optimization_stats(self) -> Dict:
        """
        Get OCR optimization statistics
        
        Returns:
            Dictionary with optimization metrics
        """
        return {
            'bbox_match_skips': self.bbox_match_skips,
            'text_cache_skips': self.text_cache_skips,
            'total_skips': self.bbox_match_skips + self.text_cache_skips,
            'over_bounding_box_stored': self.over_bounding_box is not None
        }
    
    def annotate_frame(self, frame: np.ndarray, ocr_results: List[Tuple], 
                      cricket_relevant_texts: List[str] = None) -> np.ndarray:
        """
        Annotate frame with OCR results and bounding boxes
        
        Args:
            frame: Input frame
            ocr_results: OCR detection results
            cricket_relevant_texts: Texts identified as cricket-relevant
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # 🎯 DRAW OCR REGION BOUNDARY if configured
        if self.ocr_region:
            height, width = frame.shape[:2]
            region_type = self.ocr_region.get('type', 'fraction')
            
            if region_type == 'fraction':
                # Convert fractions to pixel coordinates
                x = int(self.ocr_region['x'] * width)
                y = int(self.ocr_region['y'] * height)
                region_width = int(self.ocr_region['width'] * width)
                region_height = int(self.ocr_region['height'] * height)
            else:  # pixels
                x = self.ocr_region['x']
                y = self.ocr_region['y']
                region_width = self.ocr_region['width']
                region_height = self.ocr_region['height']
            
            # Draw OCR region boundary
            region_color = (255, 0, 255)  # Magenta for OCR region
            cv2.rectangle(annotated_frame, (x, y), (x + region_width, y + region_height), region_color, 3)
            
            # Add OCR region label
            label_text = f"OCR REGION ({region_width}x{region_height})"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_y = max(y - 10, label_size[1] + 10)
            cv2.rectangle(annotated_frame, (x, label_y - label_size[1] - 5), 
                         (x + label_size[0] + 10, label_y + 5), region_color, -1)
            cv2.putText(annotated_frame, label_text, (x + 5, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 📝 DRAW TEXT BOUNDING BOXES
        for bbox, text, confidence in ocr_results:
            # Color based on cricket relevance
            if cricket_relevant_texts and text in cricket_relevant_texts:
                color = (0, 255, 0)  # Green for cricket-relevant text
                thickness = 3
                label_bg_color = (0, 200, 0)
            elif confidence > 0.8:
                color = (0, 255, 255)  # Yellow for high confidence
                thickness = 2
                label_bg_color = (0, 200, 200)
            else:
                color = (0, 165, 255)  # Orange for other text
                thickness = 1
                label_bg_color = (0, 130, 200)
            
            # Draw bounding box
            cv2.polylines(annotated_frame, [np.array(bbox)], True, color, thickness)
            
            # Add text label with confidence
            label_text = f"{text} ({confidence:.2f})"
            
            # Position label above the bounding box
            top_left = bbox[0]
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_y = max(int(top_left[1]) - 10, label_size[1] + 5)
            
            # Draw label background
            cv2.rectangle(annotated_frame, 
                         (int(top_left[0]), label_y - label_size[1] - 5),
                         (int(top_left[0]) + label_size[0] + 5, label_y + 5),
                         label_bg_color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label_text, 
                       (int(top_left[0]) + 2, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
