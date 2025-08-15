#!/usr/bin/env python3
"""
Strands Text Extractor for Cricket Analysis - Main Application
A modular system for extracting and analyzing cricket information from video streams
"""

import cv2
import time
import argparse
import sys
from pathlib import Path

# Import modular components
from src import (
    OCRProcessor,
    CricketAnalyzer,
    VideoClipper,
    InputProcessor,
    FrameProcessor
)


class StrandsTextExtractor:
    """Main application class that coordinates all modules"""
    
    def __init__(self, languages=['en'], gpu=True, ocr_region=None):
        """
        Initialize the Strands Text Extractor with all modules
        
        Args:
            languages: Languages for OCR
            gpu: Use GPU for OCR
            ocr_region: OCR region configuration dict with 'type', 'x', 'y', 'width', 'height'
        """
        print("Initializing Strands Text Extractor...")
        
        # Initialize all modules
        self.ocr_processor = OCRProcessor(languages=languages, gpu=gpu, ocr_region=ocr_region)
        self.cricket_analyzer = CricketAnalyzer()
        
        # Initialize video clipper with preference for ffmpeg-python
        try:
            self.video_clipper = VideoClipper(prefer_ffmpeg_python=True)
        except ImportError as e:
            print(f"⚠️  Warning: {e}")
            print("   Video clipping will not be available")
            self.video_clipper = None
        
        self.input_processor = InputProcessor()
        self.frame_processor = FrameProcessor(self.ocr_processor, self.cricket_analyzer)
        
        print("Strands Text Extractor initialized successfully!")
    
    def process_video(self, video_input: str, show_video: bool = True, 
                     frame_skip: int = 1, clip_videos: bool = False):
        """
        Process video file or network stream
        
        Args:
            video_input: Path to video file or network stream URL
            show_video: Whether to display video with annotations
            frame_skip: Process every Nth frame (1 = all frames)
            clip_videos: Whether to create video clips for each over
        """
        # Start overall processing timer
        overall_start_time = time.time()
        
        print(f"🚀 STARTING VIDEO PROCESSING")
        print(f"   📹 Input: {video_input}")
        print(f"   👁️  Show video: {show_video}")
        print(f"   ⏭️  Frame skip: {frame_skip}")
        print(f"   🎬 Clip videos: {clip_videos}")
        
        # Prepare video source
        video_path, input_type = self.input_processor.prepare_video_source(video_input)
        
        if not video_path:
            print("❌ Failed to prepare video source")
            return
        
        # Get video properties
        properties = self.input_processor.get_video_properties(video_path)
        if not properties.get('valid'):
            print(f"❌ {properties.get('error')}")
            return
        
        # Print video information
        self.input_processor.print_video_info(video_path, properties, frame_skip)
        
        # Check if clipping should be disabled
        clip_videos, clip_reason = self.input_processor.should_disable_clipping(input_type, clip_videos)
        print(f"🎬 {clip_reason}")
        
        # Set up JSON output path
        output_filename = self.input_processor.generate_output_filename(video_path, input_type)
        json_output_path = f"{output_filename}_ball_by_ball_log.json"
        self.frame_processor.set_json_output_path(json_output_path)
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source {video_path}")
            return
        
        # Processing variables
        frame_number = 0
        processed_count = 0
        processing_start_time = time.time()
        
        try:
            print(f"\n🎬 STARTING FRAME PROCESSING...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                
                # Skip frames if needed
                if frame_number % frame_skip != 0:
                    continue
                
                processed_count += 1
                timestamp = self._calculate_timestamp(frame_number, properties.get('fps'), processed_count, frame_skip)
                
                # Process frame
                try:
                    annotated_frame = self.frame_processor.process_single_frame(
                        frame, frame_number, timestamp, show_video
                    )
                    
                    # Display video
                    if show_video and annotated_frame is not None:
                        cv2.imshow('Strands Cricket Text Extractor - Press Q to quit', annotated_frame)
                        
                        # Check for quit
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == ord('Q'):
                            print("Quit requested by user")
                            break
                
                except Exception as e:
                    print(f"❌ Error processing frame {frame_number}: {str(e)}")
                    continue
        
        except KeyboardInterrupt:
            print("\n⚠️  Processing interrupted by user")
        
        finally:
            cap.release()
            if show_video:
                cv2.destroyAllWindows()
            
            # Calculate processing times
            processing_end_time = time.time()
            frame_processing_time = processing_end_time - processing_start_time
            
            # Finalize processing
            if processed_count > 0:
                final_timestamp = self._calculate_timestamp(frame_number, properties.get('fps'), processed_count, frame_skip)
                
                # 🔥 FINAL FORCE JSON UPDATE: Before finalization
                self.frame_processor.force_json_update("final_processing_update")
                
                self.frame_processor.finalize_processing(final_timestamp, frame_number)
            
            # Print comprehensive summary
            self._print_final_summary(
                processed_count, properties, frame_processing_time, 
                overall_start_time, processing_end_time
            )
            
            # Handle video clipping
            if clip_videos:
                self._handle_video_clipping(video_path, input_type)
        
        # Calculate and return overall processing time (moved outside finally block)
        overall_processing_time = processing_end_time - overall_start_time
        return overall_processing_time
    
    def _calculate_timestamp(self, frame_number: int, fps: float, processed_count: int, frame_skip: int) -> float:
        """Calculate timestamp for current frame"""
        if fps and fps > 0:
            return frame_number / fps
        else:
            # For live streams, estimate based on processed frames
            return processed_count * frame_skip
    
    def _print_final_summary(self, processed_count: int, properties: dict, 
                           frame_processing_time: float, overall_start_time: float, 
                           processing_end_time: float):
        """Print comprehensive final processing summary"""
        overall_processing_time = processing_end_time - overall_start_time
        
        print(f"\n📊 FINAL PROCESSING SUMMARY:")
        print(f"  Total frames processed: {processed_count}")
        
        if properties.get('total_frames'):
            print(f"  Total frames in video: {properties['total_frames']}")
            print(f"  Processing rate: {processed_count/properties['total_frames']*100:.1f}%")
        else:
            print(f"  Total frames in video: Unknown (live stream)")
            print(f"  Processing rate: N/A (live stream)")
        
        # Get detailed statistics from modules
        frame_stats = self.frame_processor.get_processing_stats()
        
        print(f"\n⏱️  PROCESSING TIME BREAKDOWN:")
        print(f"  🕐 Total processing time: {overall_processing_time:.2f}s ({overall_processing_time/60:.2f} minutes)")
        print(f"  🎬 Frame processing time: {frame_processing_time:.2f}s ({frame_processing_time/overall_processing_time*100:.1f}% of total)")
        print(f"  ⚙️  Setup/cleanup time: {overall_processing_time - frame_processing_time:.2f}s ({(overall_processing_time - frame_processing_time)/overall_processing_time*100:.1f}% of total)")
        
        if processed_count > 0:
            print(f"  📊 Processing efficiency: {processed_count/overall_processing_time:.2f} frames/second")
            print(f"  ⏱️  Average time per frame: {frame_processing_time/processed_count:.3f}s")
            
            if properties.get('total_frames') and properties.get('fps'):
                real_time_ratio = (properties['total_frames']/properties['fps']) / overall_processing_time
                print(f"  ⚡ Real-time performance: {real_time_ratio:.2f}x {'(faster than real-time)' if real_time_ratio < 1 else '(slower than real-time)'}")
        
        print(f"\n🤖 MODULE STATISTICS:")
        print(f"  OCR Processor:")
        ocr_stats = frame_stats['ocr_stats']
        print(f"    🎯 Bbox match skips: {ocr_stats['bbox_match_skips']}")
        print(f"    📝 Text cache skips: {ocr_stats['text_cache_skips']}")
        print(f"    ⏭️  Total OCR skips: {ocr_stats['total_skips']}")
        
        print(f"  Cricket Analyzer:")
        cricket_stats = frame_stats['cricket_stats']
        print(f"    🤖 Agent calls made: {cricket_stats['agent_calls_made']}")
        print(f"    🏏 Balls tracked: {cricket_stats['balls_tracked']}")
        print(f"    🏏 Unique overs: {cricket_stats['unique_overs']}")
        print(f"    ⭐ Total highlights: {cricket_stats['total_highlights']}")
        print(f"    📈 Highlight rate: {cricket_stats['highlight_rate']:.1f}%")
        print(f"    🤖 AI agent available: {cricket_stats['ai_agent_available']}")
        
        if cricket_stats['balls_tracked'] > 0:
            efficiency = ocr_stats['total_skips'] / (cricket_stats['agent_calls_made'] + ocr_stats['total_skips']) * 100
            print(f"  ⚡ Overall optimization efficiency: {efficiency:.1f}%")
        
        print(f"  🏁 Processing completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(processing_end_time))}")
    
    def _handle_video_clipping(self, video_path: str, input_type: str):
        """Handle video clipping operations"""
        if self.video_clipper is None:
            print(f"\n❌ Video clipping requested but no video processing libraries available")
            print(f"   Install with: pip install ffmpeg-python moviepy")
            return
        
        if input_type == "network_stream":
            print(f"\n⚠️  Video clipping skipped for network streams")
            return
        
        if not self.cricket_analyzer.ball_by_ball_log:
            print(f"\n⚠️  Video clipping requested but no ball timing data available")
            return
        
        print(f"\n🎬 Starting video clipping process...")
        
        # Clip by individual balls
        output_dir, successful_clips, failed_clips = self.video_clipper.clip_video_by_balls(
            video_path, self.cricket_analyzer.ball_by_ball_log
        )
        
        if output_dir and successful_clips > 0:
            print(f"\n✅ Video clipping completed successfully!")
            print(f"   📁 Clipped videos saved to: {output_dir}")
        elif failed_clips > 0:
            print(f"\n⚠️  Video clipping completed with some failures")
        else:
            print(f"\n❌ Video clipping failed")
        
        # Print clipping statistics
        clipping_stats = self.video_clipper.get_clipping_stats()
        print(f"   📊 Clipping success rate: {clipping_stats['success_rate']:.1f}%")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Strands Text Extractor for Cricket Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4                                    # Process local video (full frame)
  %(prog)s udp://127.0.0.1:1234                        # Process UDP stream
  %(prog)s rtsp://camera.local:554/stream               # Process RTSP stream
  %(prog)s video.mp4 --frame-skip 5 --clip-videos      # Skip frames and create clips
  
  # OCR Region Examples:
  %(prog)s video.mp4 --ocr-region 0.0 0.7 0.4 0.3      # Bottom-left corner (40%% width, 30%% height)
  %(prog)s video.mp4 --ocr-region 0.6 0.0 0.4 0.2      # Top-right corner (40%% width, 20%% height)
  %(prog)s video.mp4 --ocr-region-pixels 0 756 768 324 # Bottom-left in pixels (1920x1080 frame)
        """
    )
    
    parser.add_argument(
        'video_input', 
        help='Path to video file or network stream URL (udp://, rtsp://, etc.)'
    )
    parser.add_argument(
        '--no-video', 
        action='store_true', 
        help='Disable video display'
    )
    parser.add_argument(
        '--frame-skip', 
        type=int, 
        default=1, 
        help='Process every Nth frame (default: 1)'
    )
    parser.add_argument(
        '--no-gpu', 
        action='store_true', 
        help='Disable GPU acceleration'
    )
    parser.add_argument(
        '--languages', 
        nargs='+', 
        default=['en'], 
        help='OCR languages (default: en)'
    )
    parser.add_argument(
        '--clip-videos', 
        action='store_true', 
        help='Enable video clipping by balls (disabled by default, not available for live streams)'
    )
    
    # OCR Region Arguments
    parser.add_argument(
        '--ocr-region',
        nargs=4,
        type=float,
        metavar=('X', 'Y', 'WIDTH', 'HEIGHT'),
        help='OCR region as fractions of frame (x, y, width, height). Example: --ocr-region 0.0 0.7 0.4 0.3 for bottom-left corner'
    )
    parser.add_argument(
        '--ocr-region-pixels',
        nargs=4,
        type=int,
        metavar=('X', 'Y', 'WIDTH', 'HEIGHT'),
        help='OCR region in pixels (x, y, width, height). Example: --ocr-region-pixels 0 756 768 324'
    )
    
    args = parser.parse_args()
    
    # Validate OCR region arguments
    if args.ocr_region and args.ocr_region_pixels:
        print("❌ Error: Cannot specify both --ocr-region and --ocr-region-pixels")
        sys.exit(1)
    
    # Prepare OCR region configuration
    ocr_region_config = None
    if args.ocr_region:
        x, y, width, height = args.ocr_region
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
            print("❌ Error: --ocr-region values must be between 0.0 and 1.0")
            sys.exit(1)
        if x + width > 1 or y + height > 1:
            print("❌ Error: OCR region extends beyond frame boundaries")
            sys.exit(1)
        ocr_region_config = {
            'type': 'fraction',
            'x': x, 'y': y, 'width': width, 'height': height
        }
        print(f"🎯 OCR Region (fraction): x={x}, y={y}, width={width}, height={height}")
    elif args.ocr_region_pixels:
        x, y, width, height = args.ocr_region_pixels
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            print("❌ Error: --ocr-region-pixels values must be positive")
            sys.exit(1)
        ocr_region_config = {
            'type': 'pixels',
            'x': x, 'y': y, 'width': width, 'height': height
        }
        print(f"🎯 OCR Region (pixels): x={x}, y={y}, width={width}, height={height}")
    
    # Initialize extractor
    try:
        print(f"🚀 Initializing Strands Text Extractor...")
        extractor = StrandsTextExtractor(
            languages=args.languages,
            gpu=not args.no_gpu,
            ocr_region=ocr_region_config  # Pass OCR region configuration
        )
    except Exception as e:
        print(f"❌ Error initializing Strands text extractor: {str(e)}")
        sys.exit(1)
    
    # Process video
    try:
        processing_time = extractor.process_video(
            video_input=args.video_input,
            show_video=not args.no_video,
            frame_skip=args.frame_skip,
            clip_videos=args.clip_videos
        )
        
        if processing_time:
            print(f"\n🎉 Processing completed successfully in {processing_time:.2f} seconds!")
        
    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
