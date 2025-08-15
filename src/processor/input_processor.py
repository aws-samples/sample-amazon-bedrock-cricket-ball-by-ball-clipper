"""
Input Processor Module
Handles different types of video input sources (local files, network streams)
Located in src/processor/ to emphasize its processing functionality
"""

import re
import cv2
from pathlib import Path
from typing import Optional, Tuple, Dict


class InputProcessor:
    """Handles various video input sources and validation"""
    
    def __init__(self):
        """Initialize input processor"""
        print("Input Processor initialized successfully!")
    
    def is_network_stream(self, url: str) -> bool:
        """
        Check if the provided string is a network stream URL
        
        Args:
            url: URL string to check
            
        Returns:
            True if it's a network stream, False otherwise
        """
        network_patterns = [
            r'^udp://.*',           # UDP streams
            r'^tcp://.*',           # TCP streams  
            r'^rtmp://.*',          # RTMP streams
            r'^rtsp://.*',          # RTSP streams
            r'^http://.*\.m3u8.*',  # HLS streams
            r'^https://.*\.m3u8.*', # HLS streams (HTTPS)
            r'^rtp://.*',           # RTP streams
            r'^mms://.*',           # MMS streams
            r'^mmsh://.*',          # MMSH streams
            r'^mmst://.*',          # MMST streams
        ]
        
        for pattern in network_patterns:
            if re.match(pattern, url, re.IGNORECASE):
                return True
        return False
    
    def is_local_file(self, path: str) -> bool:
        """
        Check if the provided string is a local file path
        
        Args:
            path: File path to check
            
        Returns:
            True if it's a local file, False otherwise
        """
        return not self.is_network_stream(path)
    
    def validate_input(self, video_input: str) -> Tuple[bool, str, str]:
        """
        Validate video input and determine its type
        
        Args:
            video_input: Video input string
            
        Returns:
            Tuple of (is_valid, input_type, message)
        """
        if self.is_network_stream(video_input):
            return True, "network_stream", f"Network stream detected: {video_input}"
        
        elif self.is_local_file(video_input):
            if Path(video_input).exists():
                return True, "local_file", f"Local file found: {video_input}"
            else:
                return False, "local_file", f"Local file not found: {video_input}"
        
        else:
            return False, "unknown", f"Unknown input type: {video_input}"
    
    def prepare_video_source(self, video_input: str) -> Tuple[Optional[str], str]:
        """
        Prepare video source for processing
        
        Args:
            video_input: Original video input
            
        Returns:
            Tuple of (processed_video_path, input_type)
        """
        is_valid, input_type, message = self.validate_input(video_input)
        
        if not is_valid:
            print(f"❌ {message}")
            return None, input_type
        
        print(f"✅ {message}")
        
        if input_type == "network_stream":
            print(f"📡 Network stream detected: {video_input}")
            return video_input, input_type
        
        elif input_type == "local_file":
            return video_input, input_type
        
        return None, input_type
    
    def get_video_properties(self, video_path: str) -> Dict:
        """
        Get video properties using OpenCV
        
        Args:
            video_path: Path to video file or stream URL
            
        Returns:
            Dictionary with video properties
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {
                'valid': False,
                'error': f"Could not open video source: {video_path}"
            }
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 and total_frames > 0 else 0
            
            # Determine if it's a live stream
            is_live_stream = total_frames <= 0 or fps <= 0
            
            properties = {
                'valid': True,
                'fps': fps if fps > 0 else None,
                'total_frames': total_frames if total_frames > 0 else None,
                'width': width if width > 0 else None,
                'height': height if height > 0 else None,
                'duration': duration if duration > 0 else None,
                'is_live_stream': is_live_stream,
                'resolution': f"{width}x{height}" if width > 0 and height > 0 else None
            }
            
            return properties
            
        finally:
            cap.release()
    
    def generate_output_filename(self, video_path: str, input_type: str) -> str:
        """
        Generate appropriate output filename based on input type
        
        Args:
            video_path: Video path or URL
            input_type: Type of input (local_file, network_stream, etc.)
            
        Returns:
            Base filename for output files
        """
        if input_type == "network_stream":
            # For network streams, create filename from protocol and address
            match = re.match(r'^(\w+)://([^:/]+)', video_path)
            if match:
                protocol, address = match.groups()
                return f"{protocol}_{address.replace('.', '_')}"
            else:
                return "network_stream"
        
        elif input_type == "local_file":
            return Path(video_path).stem
        
        else:
            return "video_analysis"
    
    def print_video_info(self, video_path: str, properties: Dict, frame_skip: int = 1):
        """
        Print formatted video information
        
        Args:
            video_path: Video path or URL
            properties: Video properties dictionary
            frame_skip: Frame skip setting
        """
        print(f"🎬 Processing video source: {video_path}")
        print(f"Video properties:")
        
        if properties.get('valid'):
            if properties.get('fps'):
                print(f"  FPS: {properties['fps']}")
            else:
                print(f"  FPS: Unknown (live stream)")
            
            if properties.get('total_frames'):
                print(f"  Total frames: {properties['total_frames']}")
                if properties.get('duration'):
                    print(f"  Duration: {properties['duration']:.2f} seconds")
            else:
                print(f"  Total frames: Unknown (live stream)")
                print(f"  Duration: Live stream")
            
            if properties.get('resolution'):
                print(f"  Resolution: {properties['resolution']}")
            
            print(f"  Processing every {frame_skip} frame(s)")
            
            if properties.get('is_live_stream'):
                print(f"  📡 Live stream detected - some features may be limited")
        else:
            print(f"  ❌ Error: {properties.get('error', 'Unknown error')}")
    
    def should_disable_clipping(self, input_type: str, clip_videos: bool) -> Tuple[bool, str]:
        """
        Determine if video clipping should be disabled based on input type
        
        Args:
            input_type: Type of video input
            clip_videos: Original clipping preference
            
        Returns:
            Tuple of (final_clip_setting, reason)
        """
        if not clip_videos:
            return False, "Video clipping disabled by user"
        
        if input_type == "network_stream":
            return False, "Video clipping disabled for live streams"
        
        return True, "Video clipping enabled"
