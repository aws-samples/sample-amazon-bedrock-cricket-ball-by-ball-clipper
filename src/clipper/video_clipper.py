"""
Video Clipper Module
Handles video clipping and processing using Python FFmpeg libraries
Located in src/clipper/ to emphasize its video clipping functionality
"""

import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Import Python ffmpeg libraries
try:
    import ffmpeg
    FFMPEG_PYTHON_AVAILABLE = True
except ImportError:
    FFMPEG_PYTHON_AVAILABLE = False
    print("⚠️  ffmpeg-python not available. Install with: pip install ffmpeg-python")

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("⚠️  moviepy not available. Install with: pip install moviepy")


class VideoClipper:
    """Handles video clipping and processing operations using Python libraries"""
    
    def __init__(self, prefer_ffmpeg_python: bool = True):
        """
        Initialize video clipper
        
        Args:
            prefer_ffmpeg_python: Whether to prefer ffmpeg-python over moviepy
        """
        self.successful_clips = 0
        self.failed_clips = 0
        self.prefer_ffmpeg_python = prefer_ffmpeg_python
        
        # Determine which library to use
        if prefer_ffmpeg_python and FFMPEG_PYTHON_AVAILABLE:
            self.library = "ffmpeg-python"
            print("Video Clipper initialized with ffmpeg-python!")
        elif MOVIEPY_AVAILABLE:
            self.library = "moviepy"
            print("Video Clipper initialized with moviepy!")
        else:
            self.library = "none"
            print("❌ No Python ffmpeg libraries available!")
            print("   Install with: pip install ffmpeg-python moviepy")
            raise ImportError("No video processing libraries available")
        
        print(f"   📚 Using library: {self.library}")
    
    def _validate_file_path(self, path: str) -> bool:
        """
        Validate file path to prevent directory traversal and command injection
        
        Args:
            path: File path to validate
            
        Returns:
            True if path is safe, False otherwise
        """
        try:
            # Convert to Path object for validation
            path_obj = Path(path).resolve()
            
            # Check for dangerous characters and patterns
            dangerous_patterns = [
                r'[;&|`$(){}[\]<>]',  # Shell metacharacters
                r'\.\.',              # Directory traversal
                r'^-',                # Options that start with dash
            ]
            
            path_str = str(path_obj)
            for pattern in dangerous_patterns:
                if re.search(pattern, path_str):
                    return False
            
            return True
        except (OSError, ValueError):
            return False
    
    def _validate_numeric_parameter(self, value: float, min_val: float = 0, max_val: float = 86400) -> bool:
        """
        Validate numeric parameters (time values) to prevent injection
        
        Args:
            value: Numeric value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value (default 24 hours)
            
        Returns:
            True if value is safe, False otherwise
        """
        try:
            float_val = float(value)
            return min_val <= float_val <= max_val
        except (ValueError, TypeError):
            return False
    
    def clip_video_by_balls(self, input_video_path: str, ball_timing_log: List[Dict], 
                           output_dir: str = None) -> Tuple[Optional[Path], int, int]:
        """
        Clip video based on ball timing data using FFmpeg
        
        Args:
            input_video_path: Path to the original video file
            ball_timing_log: List of ball timing entries
            output_dir: Directory to save clipped videos
            
        Returns:
            Tuple of (output_dir, successful_clips, failed_clips)
        """
        if not ball_timing_log:
            print(f"⚠️  No ball timing data available for video clipping")
            return None, 0, 0
        
        try:
            # Set output directory in output folder
            if output_dir is None:
                video_path = Path(input_video_path)
                output_base_dir = Path("output")
                output_base_dir.mkdir(exist_ok=True)
                output_dir = output_base_dir / f"{video_path.stem}_ball_clips"
            
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            print(f"\n🎬 CLIPPING VIDEO BY BALLS USING {self.library.upper()}:")
            print(f"   📁 Input Video: {input_video_path}")
            print(f"   📁 Output Directory: {output_dir}")
            print(f"   📊 Total Balls to Clip: {len(ball_timing_log)}")
            
            self.successful_clips = 0
            self.failed_clips = 0
            
            for i, ball_entry in enumerate(ball_timing_log, 1):
                over_value = ball_entry['over_value']
                start_time = ball_entry['start_time']
                end_time = ball_entry['end_time']
                duration = ball_entry['duration']
                
                # Create output filename
                output_filename = f"{over_value}.mp4"
                output_path = output_dir / output_filename
                
                print(f"\n   {i:2d}. Clipping Ball {over_value}:")
                print(f"       ⏰ Time Range: {start_time}s - {end_time}s ({duration}s)")
                print(f"       📁 Output: {output_filename}")
                
                if self._clip_single_segment(input_video_path, output_path, start_time, duration):
                    self.successful_clips += 1
                else:
                    self.failed_clips += 1
            
            self._print_clipping_summary(output_dir)
            return output_dir, self.successful_clips, self.failed_clips
            
        except Exception as e:
            print(f"❌ Error during video clipping setup: {str(e)}")
            return None, 0, 0
    
    def clip_video_by_overs(self, input_video_path: str, over_aggregation: Dict, 
                           output_dir: str = None) -> Tuple[Optional[Path], int, int]:
        """
        Clip video by complete overs using over aggregation data
        
        Args:
            input_video_path: Path to the original video file
            over_aggregation: Over aggregation data with statistics
            output_dir: Directory to save clipped videos
            
        Returns:
            Tuple of (output_dir, successful_clips, failed_clips)
        """
        if not over_aggregation or 'over_statistics' not in over_aggregation:
            print(f"⚠️  No over aggregation data available for video clipping")
            return None, 0, 0
        
        try:
            # Set output directory in output folder
            if output_dir is None:
                video_path = Path(input_video_path)
                output_base_dir = Path("output")
                output_base_dir.mkdir(exist_ok=True)
                output_dir = output_base_dir / f"{video_path.stem}_over_clips"
            
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            over_stats = over_aggregation['over_statistics']
            
            print(f"\n🎬 CLIPPING VIDEO BY OVERS USING {self.library.upper()}:")
            print(f"   📁 Input Video: {input_video_path}")
            print(f"   📁 Output Directory: {output_dir}")
            print(f"   📊 Total Overs to Clip: {len(over_stats)}")
            
            self.successful_clips = 0
            self.failed_clips = 0
            
            for i, (over_number, stats) in enumerate(over_stats.items(), 1):
                if not stats.get('balls_data'):
                    continue
                
                # Calculate over time range from balls data
                balls_data = stats['balls_data']
                start_time = balls_data[0]['start_time']
                end_time = balls_data[-1]['end_time']
                duration = end_time - start_time
                
                # Create output filename
                output_filename = f"over_{over_number}.mp4"
                output_path = output_dir / output_filename
                
                print(f"\n   {i:2d}. Clipping Over {over_number}:")
                print(f"       ⏰ Time Range: {start_time}s - {end_time}s ({duration:.2f}s)")
                print(f"       🏃 Balls: {stats['total_balls']}")
                print(f"       📊 Runs: {stats['total_runs_in_over']}")
                print(f"       📁 Output: {output_filename}")
                
                if self._clip_single_segment(input_video_path, output_path, start_time, duration):
                    self.successful_clips += 1
                else:
                    self.failed_clips += 1
            
            self._print_clipping_summary(output_dir)
            return output_dir, self.successful_clips, self.failed_clips
            
        except Exception as e:
            print(f"❌ Error during over clipping setup: {str(e)}")
            return None, 0, 0
    
    def _clip_single_segment_ffmpeg_python(self, input_path: str, output_path: Path, 
                                         start_time: float, duration: float) -> bool:
        """
        Clip a single video segment using ffmpeg-python library
        
        Args:
            input_path: Input video path
            output_path: Output video path
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create ffmpeg stream
            input_stream = ffmpeg.input(str(input_path), ss=start_time, t=duration)
            
            # Configure output with stream copy for speed
            output_stream = ffmpeg.output(
                input_stream,
                str(output_path),
                vcodec='copy',
                acodec='copy',
                avoid_negative_ts='make_zero'
            )
            
            # Run the ffmpeg command
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
            
            # Check if output file was created successfully
            if output_path.exists() and output_path.stat().st_size > 0:
                file_size = output_path.stat().st_size / (1024 * 1024)
                print(f"       ✅ Success: {output_path.name} ({file_size:.1f} MB)")
                return True
            else:
                print(f"       ❌ Error: Output file not created or is empty")
                return False
                
        except ffmpeg.Error as e:
            print(f"       ❌ FFmpeg Error: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"           stderr: {e.stderr.decode()}")
            return False
        except Exception as e:
            print(f"       ❌ Error: {str(e)}")
            return False
    
    def _clip_single_segment_moviepy(self, input_path: str, output_path: Path, 
                                   start_time: float, duration: float) -> bool:
        """
        Clip a single video segment using moviepy library
        
        Args:
            input_path: Input video path
            output_path: Output video path
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load video clip
            with VideoFileClip(str(input_path)) as video:
                # Extract subclip
                end_time = start_time + duration
                subclip = video.subclip(start_time, end_time)
                
                # Write the subclip
                subclip.write_videofile(
                    str(output_path),
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    verbose=False,
                    logger=None
                )
                
                # Close the subclip
                subclip.close()
            
            # Check if output file was created successfully
            if output_path.exists() and output_path.stat().st_size > 0:
                file_size = output_path.stat().st_size / (1024 * 1024)
                print(f"       ✅ Success: {output_path.name} ({file_size:.1f} MB)")
                return True
            else:
                print(f"       ❌ Error: Output file not created or is empty")
                return False
                
        except Exception as e:
            print(f"       ❌ MoviePy Error: {str(e)}")
            return False

    def _clip_single_segment(self, input_path: str, output_path: Path, 
                           start_time: float, duration: float) -> bool:
        """
        Clip a single video segment using the configured Python library
        
        Args:
            input_path: Input video path
            output_path: Output video path
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            True if successful, False otherwise
        """
        # Security validation: Validate all input parameters
        if not self._validate_file_path(input_path):
            print(f"       ❌ Security Error: Invalid or potentially dangerous input path")
            return False
        
        if not self._validate_file_path(str(output_path)):
            print(f"       ❌ Security Error: Invalid or potentially dangerous output path")
            return False
        
        if not self._validate_numeric_parameter(start_time):
            print(f"       ❌ Security Error: Invalid start time parameter")
            return False
        
        if not self._validate_numeric_parameter(duration):
            print(f"       ❌ Security Error: Invalid duration parameter")
            return False
        
        # Ensure input file exists
        input_file = Path(input_path)
        if not input_file.exists():
            print(f"       ❌ Error: Input file does not exist: {input_path}")
            return False
        
        # Route to appropriate clipping method
        if self.library == "ffmpeg-python":
            return self._clip_single_segment_ffmpeg_python(input_path, output_path, start_time, duration)
        elif self.library == "moviepy":
            return self._clip_single_segment_moviepy(input_path, output_path, start_time, duration)
        else:
            print(f"       ❌ Error: No video processing library available")
            return False
    
    def _print_clipping_summary(self, output_dir: Path):
        """Print clipping summary"""
        print(f"\n🎬 VIDEO CLIPPING SUMMARY:")
        print(f"   📚 Library Used: {self.library}")
        print(f"   ✅ Successful clips: {self.successful_clips}")
        print(f"   ❌ Failed clips: {self.failed_clips}")
        print(f"   📁 Output directory: {output_dir}")
        
        if self.successful_clips > 0:
            print(f"   📋 Generated files:")
            for clip_file in sorted(output_dir.glob("*.mp4")):
                file_size = clip_file.stat().st_size / (1024 * 1024)
                print(f"      - {clip_file.name} ({file_size:.1f} MB)")
    
    def get_clipping_stats(self) -> Dict:
        """
        Get video clipping statistics
        
        Returns:
            Dictionary with clipping metrics
        """
        return {
            'library_used': self.library,
            'successful_clips': self.successful_clips,
            'failed_clips': self.failed_clips,
            'total_clips_attempted': self.successful_clips + self.failed_clips,
            'success_rate': self.successful_clips / (self.successful_clips + self.failed_clips) * 100 if (self.successful_clips + self.failed_clips) > 0 else 0
        }
    
    def get_video_info(self, video_path: str) -> Optional[Dict]:
        """
        Get video information using Python libraries
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information or None if failed
        """
        try:
            if not self._validate_file_path(video_path):
                print(f"❌ Security Error: Invalid video path")
                return None
            
            video_file = Path(video_path)
            if not video_file.exists():
                print(f"❌ Error: Video file does not exist: {video_path}")
                return None
            
            if self.library == "ffmpeg-python":
                probe = ffmpeg.probe(video_path)
                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
                
                # Safely parse frame rate fraction (e.g., "30/1" -> 30.0)
                fps = None
                if video_stream and 'r_frame_rate' in video_stream:
                    try:
                        fps_str = video_stream['r_frame_rate']
                        if '/' in fps_str:
                            numerator, denominator = fps_str.split('/')
                            fps = float(numerator) / float(denominator) if float(denominator) != 0 else None
                        else:
                            fps = float(fps_str)
                    except (ValueError, ZeroDivisionError):
                        fps = None
                
                return {
                    'duration': float(probe['format']['duration']),
                    'width': int(video_stream['width']) if video_stream else None,
                    'height': int(video_stream['height']) if video_stream else None,
                    'fps': fps,
                    'video_codec': video_stream['codec_name'] if video_stream else None,
                    'audio_codec': audio_stream['codec_name'] if audio_stream else None,
                    'file_size': video_file.stat().st_size,
                    'format': probe['format']['format_name'],
                    'library_used': 'ffmpeg-python'
                }
            elif self.library == "moviepy":
                with VideoFileClip(video_path) as clip:
                    return {
                        'duration': clip.duration,
                        'width': clip.w,
                        'height': clip.h,
                        'fps': clip.fps,
                        'file_size': video_file.stat().st_size,
                        'has_audio': clip.audio is not None,
                        'library_used': 'moviepy'
                    }
            else:
                print(f"❌ Error: No video processing library available")
                return None
                
        except Exception as e:
            print(f"❌ Error getting video info: {str(e)}")
            return None
