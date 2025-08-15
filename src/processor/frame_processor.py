"""
Frame Processor Module
Coordinates frame-by-frame processing using OCR and Cricket Analysis
Located in src/processor/ to emphasize its processing coordination functionality
"""

import cv2
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..text_extract.ocr_processor import OCRProcessor
from ..agent.cricket_analyzer_agent import CricketAnalyzer


class FrameProcessor:
    """Coordinates frame processing with OCR and cricket analysis"""
    
    def __init__(self, ocr_processor: OCRProcessor, cricket_analyzer: CricketAnalyzer):
        """
        Initialize frame processor
        
        Args:
            ocr_processor: OCR processor instance
            cricket_analyzer: Cricket analyzer instance
        """
        self.ocr_processor = ocr_processor
        self.cricket_analyzer = cricket_analyzer
        
        # Processing statistics
        self.frames_processed = 0
        self.frames_skipped = 0
        self.total_processing_time = 0.0
        
        # JSON output management
        self.json_output_path = None
        
        # Global attributes for over transition tracking
        self.transition_over_value = None
        self.transition_score_value = None
        self.transition_in_progress = False
        self.transition_detection_time = None  # Track when transition was first detected
        self.transition_detection_frame = None  # Track frame when transition was detected
        
        print("Frame Processor initialized successfully!")
    
    def set_json_output_path(self, output_path: str):
        """
        Set the JSON output path for incremental updates
        
        Args:
            output_path: Path for JSON output
        """
        # Ensure output directory exists
        from pathlib import Path
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Set JSON output path in output directory
        output_filename = Path(output_path).name
        self.json_output_path = str(output_dir / output_filename)
        print(f"📁 Output directory: {output_dir.absolute()}")
        print(f"📝 JSON files will be saved to: {self.json_output_path}")
    
    def process_single_frame(self, frame: np.ndarray, frame_number: int, 
                           timestamp: float, show_video: bool = True) -> Optional[np.ndarray]:
        """
        Process a single video frame
        
        Args:
            frame: Video frame
            frame_number: Frame number
            timestamp: Video timestamp
            show_video: Whether to create annotated frame for display
            
        Returns:
            Annotated frame if show_video is True, None otherwise
        """
        frame_start_time = time.time()
        
        try:
            # Extract text using OCR
            ocr_results = self.ocr_processor.extract_text_from_frame(frame)
            detected_texts = [text for bbox, text, confidence in ocr_results]
            
            # Create text-bbox mapping
            text_bbox_map = self.ocr_processor.create_text_bbox_mapping(ocr_results)
            
            # Check if we should be in score search mode
            if (hasattr(self.cricket_analyzer, 'current_over_value') and 
                self.cricket_analyzer.current_over_value and 
                (not self.cricket_analyzer.current_over_score or 
                 self.cricket_analyzer.current_over_score == 'null')):
                # We have a tracked over but no score (or null score) - ensure score search is active
                print(f"🔍 DETECTED MISSING SCORE: Over={self.cricket_analyzer.current_over_value}, Score={self.cricket_analyzer.current_over_score or 'null'}")
                self.cricket_analyzer.score_search_active = True
                self.cricket_analyzer.mandatory_score_search = True
            
            # Check for transition mode
            if self.transition_in_progress:
                print(f"🔄 TRANSITION MODE ACTIVE:")
                print(f"   📊 Global over: {self.transition_over_value or 'Missing'}")
                print(f"   📊 Global score: {self.transition_score_value or 'Missing'}")
                print(f"   🚨 Processing every frame until BOTH values are found")
                
                # Force agent processing during transition
                self.cricket_analyzer.score_search_active = True
                self.cricket_analyzer.mandatory_score_search = True
            
            # Check for optimization opportunities
            cricket_analysis = None
            
            # Check if we need to search for score (score_search_active flag)
            score_search_needed = getattr(self.cricket_analyzer, 'score_search_active', False)
            mandatory_score_search = getattr(self.cricket_analyzer, 'mandatory_score_search', False)
            
            if score_search_needed:
                if mandatory_score_search:
                    print(f"🚨 MANDATORY SCORE SEARCH: Over transition requires score identification")
                    print(f"   📊 Current over: {getattr(self.cricket_analyzer, 'current_over_value', 'None')}")
                    print(f"   📊 Current score: {getattr(self.cricket_analyzer, 'current_over_score', 'None')}")
                    print(f"   🚫 ALL optimizations disabled until score found")
                else:
                    print(f"🔍 SCORE SEARCH MODE: Processing every frame until score is found")
                    print(f"   📊 Current over: {getattr(self.cricket_analyzer, 'current_over_value', 'None')}")
                    print(f"   📊 Current score: {getattr(self.cricket_analyzer, 'current_over_score', 'None')}")
                    print(f"   🚫 Skipping optimizations (no frame skipping, no caching)")
            else:
                current_over = getattr(self.cricket_analyzer, 'current_over_value', None)
                current_score = getattr(self.cricket_analyzer, 'current_over_score', None)
                if current_over:
                    # 🔥 BUG FIX: Check if score is actually available before enabling optimizations
                    if current_score and current_score != 'null':
                        print(f"✅ OPTIMIZATION MODE: Over={current_over}, Score={current_score}")
                        print(f"   ⚡ Optimizations active (frame skipping, caching enabled)")
                    else:
                        print(f"🚨 SCORE MISSING: Over={current_over}, Score={current_score or 'null'}")
                        print(f"   🔍 Should be in score search mode - fixing...")
                        # Force score search mode
                        self.cricket_analyzer.score_search_active = True
                        self.cricket_analyzer.mandatory_score_search = True
                        score_search_needed = True
                        mandatory_score_search = True
                        print(f"   🚨 MANDATORY SCORE SEARCH ACTIVATED")
                        print(f"   🚫 ALL optimizations disabled until score found")
            
            # Check text similarity cache first (but skip if we need to find score)
            if not score_search_needed and self.ocr_processor.check_text_similarity_cache(detected_texts):
                self.frames_skipped += 1
                print(f"⏭️  SKIPPING ANALYSIS - Same OCR texts as previous successful analysis")
                print(f"   ✅ Optimizations active (score already found)")
                cricket_analysis = self.cricket_analyzer.last_successful_analysis
            else:
                # Check over bounding box match (but skip if we need to find score)
                over_text_match = self.ocr_processor.check_over_bounding_box_match(
                    text_bbox_map, timestamp, frame_number
                )
                
                if over_text_match and not score_search_needed:
                    self.frames_skipped += 1
                    print(f"⏭️  SKIPPING ANALYSIS - Using bounding box optimization")
                    print(f"   ✅ Optimizations active (score already found)")
                    
                    # 🔧 NORMALIZE OVER VALUE from bounding box match
                    original_over_text_match = over_text_match
                    over_text_match = self._normalize_over_value(over_text_match)
                    
                    if original_over_text_match != over_text_match:
                        print(f"🔧 Bounding box over value normalized: {original_over_text_match} → {over_text_match}")
                    
                    # Extract team score for tracking
                    team_score = self.cricket_analyzer.extract_team_score_from_texts(text_bbox_map)
                    self._track_over_change(over_text_match, timestamp, frame_number, team_score)
                    
                    cricket_analysis = {
                        "current_over": over_text_match,
                        "team_score": team_score,
                        "confidence": 0.95,
                        "detected_texts_used": [over_text_match],
                        "analysis_notes": "Over info from bounding box optimization"
                    }
                elif over_text_match and score_search_needed:
                    # We have over match but need to find score - use agent
                    print(f"🔍 SCORE SEARCH ACTIVE - Using agent despite bounding box match")
                    team_score = self.cricket_analyzer.extract_team_score_from_texts(text_bbox_map)
                    
                    if team_score:
                        # Found score! Use bounding box optimization
                        print(f"✅ SCORE FOUND via extraction: {team_score}")
                        self.frames_skipped += 1
                        self._track_over_change(over_text_match, timestamp, frame_number, team_score)
                        
                        cricket_analysis = {
                            "current_over": over_text_match,
                            "team_score": team_score,
                            "confidence": 0.95,
                            "detected_texts_used": [over_text_match],
                            "analysis_notes": "Over info from bounding box optimization + score found"
                        }
                    else:
                        # Still no score - use agent
                        if mandatory_score_search:
                            print(f"🚨 MANDATORY AGENT CALL - Over transition requires score, using AI analysis")
                            print(f"   🚨 Mandatory search: Processing frame without any optimizations")
                        else:
                            print(f"🤖 AGENT CALL - Score still missing, using AI analysis")
                            print(f"   🔍 Score search mode: Processing frame without skipping")
                        if detected_texts:
                            cricket_analysis = self.cricket_analyzer.analyze_cricket_texts(
                                detected_texts, text_bbox_map
                            )
                else:
                    # Perform full cricket analysis
                    if detected_texts:
                        if score_search_needed:
                            if mandatory_score_search:
                                print(f"🚨 MANDATORY AGENT CALL - Over transition requires score identification")
                            else:
                                print(f"🤖 AGENT CALL - Score search active, processing every frame")
                        
                        cricket_analysis = self.cricket_analyzer.analyze_cricket_texts(
                            detected_texts, text_bbox_map
                        )
                        
                        if cricket_analysis:
                            # Update text cache only if we're not actively searching for score
                            if not score_search_needed:
                                self.ocr_processor.update_text_cache(detected_texts)
                                print(f"   💾 Text cache updated (optimizations active)")
                            else:
                                if mandatory_score_search:
                                    print(f"   🚨 Text cache disabled (mandatory score search)")
                                else:
                                    print(f"   🔍 Text cache skipped (score search mode)")
                            
                            # Store over bounding box if found
                            current_over = cricket_analysis.get('current_over')
                            if current_over and current_over != 'null':
                                detected_texts_used = cricket_analysis.get('detected_texts_used', [])
                                for text in detected_texts_used:
                                    if self.cricket_analyzer.is_over_format(text):
                                        self.ocr_processor.store_over_bounding_box(
                                            text, text_bbox_map, frame_number, timestamp
                                        )
                                        break
            
            # Process cricket analysis results
            if cricket_analysis:
                current_over = cricket_analysis.get('current_over')
                team_score = cricket_analysis.get('team_score')
                
                # 🔧 NORMALIZE OVER VALUE: Handle two decimal places by keeping only one
                if current_over and current_over != 'null':
                    original_current_over = current_over
                    current_over = self._normalize_over_value(current_over)
                    
                    if original_current_over != current_over:
                        print(f"🔧 Cricket analysis over value normalized: {original_current_over} → {current_over}")
                        # Update the cricket_analysis dict with normalized value
                        cricket_analysis['current_over'] = current_over
                
                # Handle transition mode - collect both values
                if self.transition_in_progress:
                    print(f"🔄 TRANSITION MODE - Processing analysis results:")
                    
                    # Update global values if found
                    if current_over and current_over != 'null':
                        if not self.transition_over_value:
                            print(f"   ✅ Over value found: {current_over}")
                            self.transition_over_value = current_over
                        elif self.transition_over_value != current_over:
                            print(f"   🔄 Over value updated: {self.transition_over_value} → {current_over}")
                            self.transition_over_value = current_over
                    
                    if team_score and team_score != 'null':
                        if not self.transition_score_value:
                            print(f"   ✅ Score value found: {team_score}")
                            self.transition_score_value = team_score
                        elif self.transition_score_value != team_score:
                            print(f"   🔄 Score value updated: {self.transition_score_value} → {team_score}")
                            self.transition_score_value = team_score
                    elif team_score == 'null':
                        print(f"   ❌ Score value is null - continuing search")
                    
                    # Check if we have both valid values
                    if self.transition_over_value and self.transition_score_value:
                        print(f"   🎉 BOTH VALUES COLLECTED!")
                        print(f"      📊 Final over: {self.transition_over_value}")
                        print(f"      📊 Final score: {self.transition_score_value}")
                        print(f"      ⏰ Using transition start_time: {self.transition_detection_time:.2f}s")
                        
                        # Complete the transition with correct timing
                        new_over_info = self.cricket_analyzer.parse_over_value(self.transition_over_value)
                        current_over_info = self.cricket_analyzer.parse_over_value(self.cricket_analyzer.current_over_value) if self.cricket_analyzer.current_over_value else None
                        
                        # Check if this is the first ball (transition_detection_time = 0.0)
                        if self.transition_detection_time == 0.0:
                            print(f"      🏏 First ball completion - using video start as start_time")
                            
                            # Handle first ball completion
                            self.cricket_analyzer.current_over_value = self.transition_over_value
                            self.cricket_analyzer.current_over_start_time = 0.0
                            self.cricket_analyzer.current_over_start_frame = 1
                            self.cricket_analyzer.current_over_agent_calls = 0
                            self.cricket_analyzer.current_over_score = self.transition_score_value
                            self.cricket_analyzer.score_search_active = False
                            self.cricket_analyzer.mandatory_score_search = False
                            if not self.cricket_analyzer.previous_score:
                                self.cricket_analyzer.previous_score = self.transition_score_value
                            
                            # Create first ball record
                            self._create_ball_record(
                                over_value=self.transition_over_value,
                                over_number=new_over_info.get("over_number"),
                                ball_number=new_over_info.get("ball_number"),
                                timestamp=timestamp,  # end_time (current frame time)
                                frame_number=frame_number,  # end_frame
                                team_score=self.transition_score_value,
                                description=f"Over {new_over_info.get('over_number')}, ball {new_over_info.get('ball_number')} completed",
                                start_time_override=0.0,  # Start of video
                                start_frame_override=1    # Start frame
                            )
                        else:
                            # Handle subsequent ball transitions
                            self._complete_over_transition(
                                self.transition_over_value, 
                                self.transition_score_value, 
                                timestamp,  # end_time (current frame time)
                                frame_number,  # end_frame
                                new_over_info, 
                                current_over_info,
                                transition_start_time=self.transition_detection_time,  # start_time (transition detection time)
                                transition_start_frame=self.transition_detection_frame  # start_frame
                            )
                        
                        # Clear transition mode
                        self.transition_in_progress = False
                        self.transition_over_value = None
                        self.transition_score_value = None
                        self.transition_detection_time = None
                        self.transition_detection_frame = None
                    else:
                        print(f"   ⏳ TRANSITION CONTINUES - Missing values:")
                        if not self.transition_over_value:
                            print(f"      ❌ Over: Still missing")
                        if not self.transition_score_value:
                            print(f"      ❌ Score: Still missing")
                        print(f"   🤖 Will continue agent processing")
                
                # Normal processing (not in transition mode)
                elif current_over and current_over != 'null':
                    self._track_over_change(current_over, timestamp, frame_number, team_score)
                elif not current_over or current_over == 'null':
                    # No over detected but we might need to continue score search
                    if (hasattr(self.cricket_analyzer, 'current_over_value') and 
                        self.cricket_analyzer.current_over_value and 
                        not self.cricket_analyzer.current_over_score):
                        # We have a tracked over but still no score - keep searching
                        if not getattr(self.cricket_analyzer, 'score_search_active', False):
                            print(f"🔍 SCORE SEARCH CONTINUES - No over detected but score still missing")
                        self.cricket_analyzer.score_search_active = True
            else:
                # No cricket analysis results
                if (hasattr(self.cricket_analyzer, 'current_over_value') and 
                    self.cricket_analyzer.current_over_value and 
                    not self.cricket_analyzer.current_over_score):
                    # We have a tracked over but still no score and no analysis - keep searching
                    if not getattr(self.cricket_analyzer, 'score_search_active', False):
                        print(f"🔍 SCORE SEARCH CONTINUES - No analysis results but score still missing")
                    self.cricket_analyzer.score_search_active = True
            
            # Print frame information
            self._print_frame_info(frame_number, timestamp, text_bbox_map, cricket_analysis)
            
            # Create annotated frame if requested
            annotated_frame = None
            if show_video:
                cricket_relevant_texts = []
                if cricket_analysis:
                    cricket_relevant_texts = cricket_analysis.get('detected_texts_used', [])
                
                annotated_frame = self.ocr_processor.annotate_frame(
                    frame, ocr_results, cricket_relevant_texts
                )
                
                # Add frame info overlay
                annotated_frame = self._add_frame_overlay(
                    annotated_frame, frame_number, timestamp, len(text_bbox_map), cricket_analysis
                )
            
            self.frames_processed += 1
            frame_processing_time = time.time() - frame_start_time
            self.total_processing_time += frame_processing_time
            
            return annotated_frame
            
        except Exception as e:
            print(f"❌ Error processing frame {frame_number}: {str(e)}")
            return frame if show_video else None
    
    def _normalize_over_value(self, over_value: str) -> str:
        """
        Normalize over value to ensure only one decimal place
        
        Args:
            over_value: Raw over value (e.g., "5.12", "3.45", "4.1", "2.0")
            
        Returns:
            Normalized over value with one decimal place (e.g., "5.1", "3.4", "4.1", "2.0")
        """
        if not over_value or not isinstance(over_value, str):
            return over_value
        
        # Check if it contains decimal point
        if '.' in over_value:
            parts = over_value.split('.')
            if len(parts) == 2:
                integer_part = parts[0]
                decimal_part = parts[1]
                
                # If decimal part has more than 1 digit, keep only the first digit
                if len(decimal_part) > 1:
                    normalized_decimal = decimal_part[0]  # Keep only first decimal digit
                    normalized_value = f"{integer_part}.{normalized_decimal}"
                    print(f"   🔧 NORMALIZED OVER VALUE: {over_value} → {normalized_value}")
                    return normalized_value
        
        # Return as-is if no normalization needed
        return over_value
    
    def _track_over_change(self, over_value: str, timestamp: float, 
                          frame_number: int, team_score: str = None):
        """
        Track over value changes with proper ball completion logic
        
        Args:
            over_value: The over value detected (e.g., "3.0", "3.1", "4.0")
            timestamp: Current video timestamp
            frame_number: Current frame number
            team_score: Current team score (optional)
        """
        # 🔧 NORMALIZE OVER VALUE: Handle two decimal places by keeping only one
        original_over_value = over_value
        over_value = self._normalize_over_value(over_value)
        
        if original_over_value != over_value:
            print(f"🔧 Over value normalized for processing: {original_over_value} → {over_value}")
        
        # Calculate ball-by-ball runs if we have both current and last ball scores
        if team_score and self.cricket_analyzer.last_ball_score:
            ball_stats = self.cricket_analyzer.calculate_over_stats(
                team_score, self.cricket_analyzer.last_ball_score
            )
            if ball_stats and ball_stats.get("calculation_possible"):
                ball_runs = ball_stats.get("runs_scored", 0)
                ball_wickets = ball_stats.get("wickets_taken", 0)
                
                ball_highlight_info = self.cricket_analyzer.is_ball_highlight(ball_runs, ball_wickets)
                
                print(f"🏏 BALL ANALYSIS: {self.cricket_analyzer.last_ball_score} → {team_score}")
                if ball_runs is not None:
                    print(f"   🏃 Runs on this ball: {ball_runs}")
                if ball_wickets is not None and ball_wickets > 0:
                    print(f"   🎯 Wickets on this ball: {ball_wickets}")
                
                if ball_highlight_info and ball_highlight_info.get("highlight"):
                    print(f"   ⭐ BALL HIGHLIGHT DETECTED!")
                    print(f"      🎬 Highlight Type: {ball_highlight_info.get('highlight_type', 'unknown')}")
                    print(f"      📝 Reasons: {', '.join(ball_highlight_info.get('highlight_reasons', []))}")
        
        # Update last ball score for next calculation
        if team_score:
            self.cricket_analyzer.last_ball_score = team_score
        
        # Handle over changes with proper ball completion logic
        if self.cricket_analyzer.current_over_value is None:
            # First over detected - check if it's worth tracking
            
            # Quick check: if it ends with .0, ignore it completely
            if over_value.endswith('.0'):
                print(f"🚫 IGNORING FIRST DETECTION: {over_value} at {timestamp:.2f}s (Frame {frame_number})")
                print(f"   📊 This represents over completion state - not tracking")
                print(f"   ⏳ Waiting for actual ball completion (X.1, X.2, etc.)")
                
                # Store previous score if available for future calculations
                if team_score:
                    self.cricket_analyzer.previous_score = team_score
                
                return  # Exit completely, don't track the over value
            
            # Parse to double-check
            over_info = self.cricket_analyzer.parse_over_value(over_value)
            
            # Only start tracking if it's an actual ball completion (X.1, X.2, etc.), not X.0
            if over_info and over_info.get("ball_number") > 0:
                # This is an actual ball completion - START TRANSITION MODE FOR FIRST BALL
                
                print(f"🏏 FIRST BALL COMPLETION DETECTED: {over_value} at {timestamp:.2f}s (Frame {frame_number})")
                
                # 🚨 RESET GLOBAL ATTRIBUTES FOR FIRST BALL TRANSITION
                self.transition_over_value = None
                self.transition_score_value = None
                self.transition_in_progress = True
                self.transition_detection_time = 0.0  # 🔥 FIRST BALL STARTS AT VIDEO BEGINNING
                self.transition_detection_frame = 1   # 🔥 FIRST BALL STARTS AT FRAME 1
                
                print(f"   🔄 FIRST BALL TRANSITION MODE ACTIVATED")
                print(f"   📊 Global over value: {self.transition_over_value}")
                print(f"   📊 Global score value: {self.transition_score_value}")
                print(f"   ⏰ First ball start_time will be: 0.0s (video beginning)")
                
                # Check if current analysis already has both values
                if over_value and team_score and team_score != 'null':
                    print(f"   ✅ BOTH VALUES FOUND in current analysis:")
                    print(f"      📊 Over: {over_value}")
                    print(f"      📊 Score: {team_score}")
                    self.transition_over_value = over_value
                    self.transition_score_value = team_score
                    self.transition_in_progress = False
                    
                    # Complete first ball setup
                    self.cricket_analyzer.current_over_value = over_value
                    self.cricket_analyzer.current_over_start_time = 0.0  # Start of video
                    self.cricket_analyzer.current_over_start_frame = 1   # Start frame
                    self.cricket_analyzer.current_over_agent_calls = 0
                    self.cricket_analyzer.current_over_score = team_score
                    self.cricket_analyzer.score_search_active = False
                    self.cricket_analyzer.mandatory_score_search = False
                    self.cricket_analyzer.previous_score = team_score
                    
                    print(f"   ✅ First ball complete - optimizations enabled")
                    
                    # Create record for this ball completion with start_time: 0
                    self._create_ball_record(
                        over_value=over_value,
                        over_number=over_info.get("over_number"),
                        ball_number=over_info.get("ball_number"),
                        timestamp=timestamp,
                        frame_number=frame_number,
                        team_score=team_score,
                        description=f"Over {over_info.get('over_number')}, ball {over_info.get('ball_number')} completed",
                        start_time_override=0.0,  # Start of video
                        start_frame_override=1    # Start frame
                    )
                else:
                    print(f"   ⚠️  INCOMPLETE VALUES in current analysis:")
                    if over_value:
                        print(f"      ✅ Over: {over_value}")
                        self.transition_over_value = over_value
                    else:
                        print(f"      ❌ Over: Missing")
                    
                    if team_score:
                        print(f"      ✅ Score: {team_score}")
                        self.transition_score_value = team_score
                    else:
                        print(f"      ❌ Score: Missing")
                    
                    print(f"   🤖 AGENT PROCESSING REQUIRED - First ball transition incomplete")
                    
                    # Set flags to disable optimizations
                    self.cricket_analyzer.score_search_active = True
                    self.cricket_analyzer.mandatory_score_search = True
            else:
                # This shouldn't happen due to the .0 check above, but just in case
                print(f"🚫 IGNORING FIRST DETECTION: {over_value} at {timestamp:.2f}s (Frame {frame_number})")
                print(f"   📊 Over completion state - not tracking")
                return
            
        elif over_value != self.cricket_analyzer.current_over_value:
            # Over value has changed
            
            # Special case: if current_over_value is None, it means we ignored the first detection
            # and this might be the first actual ball completion we want to track
            if self.cricket_analyzer.current_over_value is None:
                # Quick check: if this is also X.0, ignore it too
                if over_value.endswith('.0'):
                    print(f"🚫 IGNORING TRANSITION TO: {over_value} at {timestamp:.2f}s (Frame {frame_number})")
                    print(f"   📊 Still over completion state - continuing to ignore")
                    
                    # Store previous score if available for future calculations
                    if team_score:
                        self.cricket_analyzer.previous_score = team_score
                    
                    return
                
                # This is the first actual ball completion - start tracking
                over_info = self.cricket_analyzer.parse_over_value(over_value)
                if over_info and over_info.get("ball_number") > 0:
                    print(f"🏏 FIRST BALL COMPLETION FROM IGNORED STATE: None → {over_value}")
                    print(f"   ⏰ DETECTION TIME: {timestamp:.2f}s (Frame {frame_number})")
                    
                    # 🚨 START TRANSITION MODE FOR FIRST BALL (special case)
                    self.transition_over_value = None
                    self.transition_score_value = None
                    self.transition_in_progress = True
                    self.transition_detection_time = 0.0  # 🔥 FIRST BALL STARTS AT VIDEO BEGINNING
                    self.transition_detection_frame = 1   # 🔥 FIRST BALL STARTS AT FRAME 1
                    
                    print(f"   🔄 FIRST BALL TRANSITION MODE ACTIVATED")
                    print(f"   ⏰ First ball start_time will be: 0.0s (video beginning)")
                    print(f"   ⏰ First ball end_time will be: {timestamp:.2f}s (detection time)")
                    
                    # Check if current analysis already has both values
                    if over_value and team_score and team_score != 'null':
                        print(f"   ✅ BOTH VALUES FOUND in current analysis:")
                        print(f"      📊 Over: {over_value}")
                        print(f"      📊 Score: {team_score}")
                        self.transition_over_value = over_value
                        self.transition_score_value = team_score
                        
                        # Complete first ball setup with correct timing
                        new_over_info = self.cricket_analyzer.parse_over_value(over_value)
                        
                        self.cricket_analyzer.current_over_value = over_value
                        self.cricket_analyzer.current_over_start_time = 0.0  # Start of video
                        self.cricket_analyzer.current_over_start_frame = 1   # Start frame
                        self.cricket_analyzer.current_over_agent_calls = 0
                        self.cricket_analyzer.current_over_score = team_score
                        self.cricket_analyzer.score_search_active = False
                        self.cricket_analyzer.mandatory_score_search = False
                        self.cricket_analyzer.previous_score = team_score
                        
                        print(f"   ✅ First ball complete - optimizations enabled")
                        print(f"   ⏰ Timing: 0.0s → {timestamp:.2f}s (duration: {timestamp:.2f}s)")
                        
                        # Create record with proper timing for first ball
                        self._create_ball_record(
                            over_value=over_value,
                            over_number=new_over_info.get("over_number"),
                            ball_number=new_over_info.get("ball_number"),
                            timestamp=timestamp,  # end_time (detection time)
                            frame_number=frame_number,  # end_frame
                            team_score=team_score,
                            description=f"Over {new_over_info.get('over_number')}, ball {new_over_info.get('ball_number')} completed",
                            start_time_override=0.0,  # Start of video
                            start_frame_override=1    # Start frame
                        )
                        
                        # Clear transition mode
                        self.transition_in_progress = False
                        self.transition_detection_time = None
                        self.transition_detection_frame = None
                    else:
                        print(f"   ⚠️  INCOMPLETE VALUES in current analysis:")
                        if over_value:
                            print(f"      ✅ Over: {over_value}")
                            self.transition_over_value = over_value
                        else:
                            print(f"      ❌ Over: Missing")
                        
                        if team_score and team_score != 'null':
                            print(f"      ✅ Score: {team_score}")
                            self.transition_score_value = team_score
                        else:
                            print(f"      ❌ Score: {team_score or 'Missing'} (invalid)")
                        
                        print(f"   🤖 AGENT PROCESSING REQUIRED - First ball transition incomplete")
                        print(f"   ⏰ Will use start_time: 0.0s when complete")
                        
                        # Set flags to disable optimizations
                        self.cricket_analyzer.score_search_active = True
                        self.cricket_analyzer.mandatory_score_search = True
                return
            
            # Normal transition logic (when we were already tracking something)
            # Parse current and new over values to understand the transition
            current_over_info = self.cricket_analyzer.parse_over_value(self.cricket_analyzer.current_over_value)
            new_over_info = self.cricket_analyzer.parse_over_value(over_value)
            
            print(f"🏏 OVER TRANSITION: {self.cricket_analyzer.current_over_value} → {over_value}")
            
            # Only create records for actual ball completions (X.1, X.2, etc.), not X.0
            if new_over_info and new_over_info.get("ball_number") > 0:
                # This is an actual ball completion - START TRANSITION MODE
                
                print(f"🔄 OVER TRANSITION DETECTED: {self.cricket_analyzer.current_over_value or 'None'} → {over_value}")
                print(f"   ⏰ TRANSITION TIME: {timestamp:.2f}s (Frame {frame_number})")
                
                # 🔥 CALCULATE CORRECT START TIME AND FRAME
                # For subsequent balls, start_time should be the end_time of the previous ball
                # and start_frame should be the end_frame of the previous ball
                if (hasattr(self.cricket_analyzer, 'current_over_start_time') and 
                    self.cricket_analyzer.current_over_start_time is not None):
                    # We have a previous ball - find its end_time and end_frame from the log
                    previous_ball_end_time = None
                    previous_ball_end_frame = None
                    if self.cricket_analyzer.ball_by_ball_log:
                        # Get the last record's end_time and end_frame
                        last_record = self.cricket_analyzer.ball_by_ball_log[-1]
                        previous_ball_end_time = last_record.get('end_time')
                        previous_ball_end_frame = last_record.get('end_frame')
                        print(f"   📊 Previous ball end_time: {previous_ball_end_time}")
                        print(f"   🎬 Previous ball end_frame: {previous_ball_end_frame}")
                    
                    if previous_ball_end_time is not None and previous_ball_end_frame is not None:
                        transition_start_time = previous_ball_end_time
                        transition_start_frame = previous_ball_end_frame
                        print(f"   ⏰ Using previous ball's end_time as start_time: {transition_start_time}")
                        print(f"   🎬 Using previous ball's end_frame as start_frame: {transition_start_frame}")
                    else:
                        transition_start_time = timestamp
                        transition_start_frame = frame_number
                        print(f"   ⚠️  No previous ball timing found, using current: {transition_start_time}s, frame {transition_start_frame}")
                else:
                    # This is the first ball - use video start
                    transition_start_time = 0.0
                    transition_start_frame = 1
                    print(f"   🏏 First ball - using video start: {transition_start_time}s, frame {transition_start_frame}")
                
                # 🚨 RESET GLOBAL ATTRIBUTES FOR TRANSITION
                self.transition_over_value = None
                self.transition_score_value = None
                self.transition_in_progress = True
                self.transition_detection_time = transition_start_time  # 🔥 USE CALCULATED START TIME
                self.transition_detection_frame = transition_start_frame  # 🔥 USE CALCULATED START FRAME
                
                print(f"   🔄 TRANSITION MODE ACTIVATED")
                print(f"   📊 Global over value: {self.transition_over_value}")
                print(f"   📊 Global score value: {self.transition_score_value}")
                print(f"   ⏰ New ball start_time will be: {self.transition_detection_time}")
                print(f"   🎬 New ball start_frame will be: {self.transition_detection_frame}")
                print(f"   ⏰ New ball end_time will be: {timestamp:.2f}s (or later when complete)")
                print(f"   🎬 New ball end_frame will be: {frame_number} (or later when complete)")
                print(f"   🚨 Will process every frame until BOTH over and score are identified")
                
                # Check if current analysis already has both values
                if over_value and team_score and team_score != 'null':
                    print(f"   ✅ BOTH VALUES FOUND in current analysis:")
                    print(f"      📊 Over: {over_value}")
                    print(f"      📊 Score: {team_score}")
                    self.transition_over_value = over_value
                    self.transition_score_value = team_score
                    
                    # Complete transition with correct start_time
                    self._complete_over_transition(
                        over_value, team_score, timestamp, frame_number, 
                        new_over_info, current_over_info, 
                        transition_start_time=self.transition_detection_time,
                        transition_start_frame=self.transition_detection_frame
                    )
                    
                    # Clear transition mode
                    self.transition_in_progress = False
                    self.transition_detection_time = None
                    self.transition_detection_frame = None
                else:
                    print(f"   ⚠️  INCOMPLETE VALUES in current analysis:")
                    if over_value:
                        print(f"      ✅ Over: {over_value}")
                        self.transition_over_value = over_value
                    else:
                        print(f"      ❌ Over: Missing")
                    
                    if team_score and team_score != 'null':
                        print(f"      ✅ Score: {team_score}")
                        self.transition_score_value = team_score
                    else:
                        print(f"      ❌ Score: {team_score or 'Missing'} (invalid)")
                    
                    print(f"   🤖 AGENT PROCESSING REQUIRED - Transition incomplete")
                    print(f"   ⏰ Will use transition time {self.transition_detection_time:.2f}s as start_time when complete")
                    
                    # Set flags to disable optimizations
                    self.cricket_analyzer.score_search_active = True
                    self.cricket_analyzer.mandatory_score_search = True
                
            else:
                # This is X.0 format - ignore it
                print(f"   📊 Over completion detected ({over_value}) - ignoring, not updating tracking")
                # Don't update current_over_value, keep tracking the previous valid ball completion
            
        elif team_score and team_score != 'null' and not self.cricket_analyzer.current_over_score:
            # Same over but found valid score - deactivate score search
            self.cricket_analyzer.current_over_score = team_score
            self.cricket_analyzer.score_search_active = False
            self.cricket_analyzer.mandatory_score_search = False  # Clear mandatory flag
            
            if not self.cricket_analyzer.previous_score:
                self.cricket_analyzer.previous_score = team_score
                
            was_mandatory = getattr(self.cricket_analyzer, 'mandatory_score_search', False)
            if was_mandatory:
                print(f"🎯 MANDATORY SCORE FOUND FOR OVER {over_value}: {team_score}")
                print(f"   ✅ Over transition complete - resuming optimizations")
            else:
                print(f"🎯 SCORE FOUND FOR CURRENT OVER {over_value}: {team_score}")
                print(f"   🔍 Score search deactivated - will resume optimizations")
        
        elif (not team_score or team_score == 'null') and self.cricket_analyzer.current_over_value and (not self.cricket_analyzer.current_over_score or self.cricket_analyzer.current_over_score == 'null'):
            # Same over but still no valid score - keep score search active
            if not getattr(self.cricket_analyzer, 'score_search_active', False):
                print(f"🔍 SCORE SEARCH ACTIVATED - Will continue agent processing until valid score found")
                print(f"   📊 Current over: {over_value}")
                print(f"   📊 Current score: {team_score or 'null'} (invalid)")
            self.cricket_analyzer.score_search_active = True
            self.cricket_analyzer.mandatory_score_search = True
    
    def _complete_over_transition(self, over_value: str, team_score: str, timestamp: float, 
                                 frame_number: int, new_over_info: dict, current_over_info: dict = None,
                                 transition_start_time: float = None, transition_start_frame: int = None):
        """
        Complete the over transition when both over and score values are available
        
        Args:
            over_value: The over value (e.g., "3.1")
            team_score: The team score (e.g., "45-2")
            timestamp: Current timestamp (used as end_time)
            frame_number: Current frame number (used as end_frame)
            new_over_info: Parsed new over information
            current_over_info: Parsed current over information
            transition_start_time: Time when transition was first detected (used as start_time)
            transition_start_frame: Frame when transition was first detected (used as start_frame)
        """
        print(f"✅ COMPLETING OVER TRANSITION:")
        print(f"   📊 Over: {over_value}")
        print(f"   📊 Score: {team_score}")
        
        # Use transition detection time as start_time, current time as end_time
        actual_start_time = transition_start_time if transition_start_time is not None else timestamp
        actual_start_frame = transition_start_frame if transition_start_frame is not None else frame_number
        
        print(f"   ⏰ Timing: {actual_start_time:.2f}s → {timestamp:.2f}s (duration: {timestamp - actual_start_time:.2f}s)")
        print(f"   🎬 Frames: {actual_start_frame} → {frame_number}")
        
        # Check if this is a new over or continuation
        if (current_over_info and new_over_info.get("over_number") != current_over_info.get("over_number")):
            # Different over number - check if we should finalize previous over
            
            # Don't finalize if previous over was X.0 (over completion state)
            if current_over_info.get("ball_number") == 0:
                print(f"   🔄 Moving to new over: {new_over_info.get('over_number')}")
                print(f"   ⏭️  Skipping finalization of previous over {self.cricket_analyzer.current_over_value} (over completion state)")
            else:
                # Previous over was actual ball completion (X.1, X.2, etc.) - finalize it
                print(f"   🔄 Moving to new over: {new_over_info.get('over_number')}")
                print(f"   📝 Finalizing previous over {self.cricket_analyzer.current_over_value}")
                self._finalize_current_over(actual_start_time, actual_start_frame)
        
        # Create record for the new ball with correct timing
        self._create_ball_record(
            over_value=over_value,
            over_number=new_over_info.get("over_number"),
            ball_number=new_over_info.get("ball_number"),
            timestamp=timestamp,  # end_time
            frame_number=frame_number,  # end_frame
            team_score=team_score,
            description=f"Over {new_over_info.get('over_number')}, ball {new_over_info.get('ball_number')} completed",
            start_time_override=actual_start_time,  # transition detection time
            start_frame_override=actual_start_frame  # transition detection frame
        )
        
        # Update current over tracking
        self.cricket_analyzer.current_over_value = over_value
        self.cricket_analyzer.current_over_start_time = actual_start_time  # Use transition time
        self.cricket_analyzer.current_over_start_frame = actual_start_frame  # Use transition frame
        self.cricket_analyzer.current_over_agent_calls = 0
        self.cricket_analyzer.current_over_score = team_score
        self.cricket_analyzer.score_search_active = False
        self.cricket_analyzer.mandatory_score_search = False
        
        print(f"   ✅ Over transition complete - optimizations enabled")
    
    def _cleanup_duplicate_records(self):
        """
        Clean up duplicate over_value records from ball_by_ball_log
        Keep the most recent/complete record for each over_value
        """
        if not hasattr(self.cricket_analyzer, 'ball_by_ball_log'):
            return
        
        print(f"🧹 CLEANING UP DUPLICATE RECORDS...")
        original_count = len(self.cricket_analyzer.ball_by_ball_log)
        
        # Group by over_value, keeping the most complete record
        over_value_map = {}
        duplicates_found = []
        
        for entry in self.cricket_analyzer.ball_by_ball_log:
            over_value = entry.get('over_value')
            if not over_value:
                continue
            
            if over_value in over_value_map:
                # Duplicate found
                existing_entry = over_value_map[over_value]
                duplicates_found.append(over_value)
                
                # Keep the record with more complete information (has score)
                if entry.get('score') and not existing_entry.get('score'):
                    print(f"   🔄 Replacing {over_value}: Better record found (has score)")
                    over_value_map[over_value] = entry
                elif existing_entry.get('score') and not entry.get('score'):
                    print(f"   ⏭️  Keeping existing {over_value}: Has score")
                    # Keep existing
                else:
                    # Keep the later one (higher end_time)
                    if entry.get('end_time', 0) > existing_entry.get('end_time', 0):
                        print(f"   🔄 Replacing {over_value}: Later timestamp")
                        over_value_map[over_value] = entry
                    else:
                        print(f"   ⏭️  Keeping existing {over_value}: Earlier timestamp")
            else:
                over_value_map[over_value] = entry
        
        if duplicates_found:
            print(f"   🚨 Found duplicates for: {', '.join(set(duplicates_found))}")
            
            # Replace the log with cleaned records
            self.cricket_analyzer.ball_by_ball_log = list(over_value_map.values())
            
            # Sort by over_number and ball_number
            self.cricket_analyzer.ball_by_ball_log.sort(
                key=lambda x: (x.get('over_number', 0), x.get('ball_number', 0))
            )
            
            final_count = len(self.cricket_analyzer.ball_by_ball_log)
            print(f"   ✅ Cleanup complete: {original_count} → {final_count} records")
            
            # Save cleaned data
            if self.json_output_path:
                self._save_incremental_json()
                self._save_incremental_over_aggregation()
                print(f"   💾 Cleaned data saved to JSON")
        else:
            print(f"   ✅ No duplicates found - {original_count} records are clean")
    
    def _create_ball_record(self, over_value: str, over_number: int, ball_number: int, 
                           timestamp: float, frame_number: int, team_score: str = None, 
                           description: str = None, start_time_override: float = None, 
                           start_frame_override: int = None):
        """
        Create a ball-by-ball record entry
        
        Args:
            over_value: Over value string (e.g., "3.1")
            over_number: Over number (e.g., 3)
            ball_number: Ball number (e.g., 1)
            timestamp: Current timestamp
            frame_number: Current frame number
            team_score: Current team score
            description: Description of the ball
        """
        # 🚫 GUARD: Never create records for X.0 values (over completion states)
        if ball_number == 0:
            print(f"🚫 BLOCKED: Attempted to create record for over completion state {over_value}")
            print(f"   ⏭️  Over completion states should not have records")
            return
        
        # Parse over value to double-check
        over_info = self.cricket_analyzer.parse_over_value(over_value)
        if over_info and over_info.get("ball_number") == 0:
            print(f"🚫 BLOCKED: Attempted to create record for over completion state {over_value}")
            print(f"   ⏭️  Over completion states should not have records")
            return
        
        # 🔥 DUPLICATE PREVENTION: Check if this over_value already exists
        existing_record = None
        for existing_entry in self.cricket_analyzer.ball_by_ball_log:
            if existing_entry.get('over_value') == over_value:
                existing_record = existing_entry
                break
        
        if existing_record:
            print(f"🚫 DUPLICATE PREVENTED: Record for {over_value} already exists")
            print(f"   📊 Existing record: Over {existing_record.get('over_number')}, Ball {existing_record.get('ball_number')}")
            print(f"   📊 Existing score: {existing_record.get('score')}")
            print(f"   ⏭️  Skipping duplicate record creation")
            
            # Update the existing record if new information is available
            if team_score and team_score != existing_record.get('score'):
                print(f"   🔄 Updating existing record score: {existing_record.get('score')} → {team_score}")
                existing_record['score'] = team_score
                existing_record['end_time'] = round(timestamp, 2)
                existing_record['duration'] = round(timestamp - existing_record.get('start_time', timestamp), 2)
                existing_record['end_frame'] = frame_number
                
                # Save JSON immediately after update
                if self.json_output_path:
                    self._save_incremental_json()
                    self._save_incremental_over_aggregation()
                    print(f"   💾 JSON updated with score correction")
            
            return
        
        # Calculate runs and wickets if possible
        runs_scored = None
        wickets_taken = None
        highlight_info = {"highlight": False, "highlight_reasons": [], "highlight_type": None}
        
        if team_score and self.cricket_analyzer.previous_score:
            ball_stats = self.cricket_analyzer.calculate_over_stats(
                team_score, self.cricket_analyzer.previous_score
            )
            if ball_stats and ball_stats.get("calculation_possible"):
                runs_scored = ball_stats.get("runs_scored")
                wickets_taken = ball_stats.get("wickets_taken")
                highlight_info = self.cricket_analyzer.is_ball_highlight(runs_scored, wickets_taken)
        
        # Determine start_time and start_frame (use overrides if provided)
        actual_start_time = start_time_override if start_time_override is not None else (self.cricket_analyzer.current_over_start_time or timestamp)
        actual_start_frame = start_frame_override if start_frame_override is not None else getattr(self.cricket_analyzer, 'current_over_start_frame', frame_number)
        
        ball_timing_entry = {
            "over_value": over_value,
            "over_number": over_number,
            "ball_number": ball_number,
            "balls_completed": ball_number,
            "over_status": "completed" if ball_number == 6 else "in_progress",
            "over_description": description or f"Over {over_number}, ball {ball_number} completed",
            "start_time": round(actual_start_time, 2),
            "end_time": round(timestamp, 2),
            "duration": round(timestamp - actual_start_time, 2),
            "start_frame": actual_start_frame,
            "end_frame": frame_number,
            "score": team_score,
            "agent_calls": self.cricket_analyzer.current_over_agent_calls,
            "runs_scored": runs_scored,
            "wickets_taken": wickets_taken,
            "previous_score": self.cricket_analyzer.previous_score,
            "highlight": highlight_info.get("highlight", False),
            "highlight_reasons": highlight_info.get("highlight_reasons", []),
            "highlight_type": highlight_info.get("highlight_type")
        }
        
        self.cricket_analyzer.ball_by_ball_log.append(ball_timing_entry)
        
        print(f"📝 BALL RECORD CREATED:")
        print(f"   🏏 {description}")
        print(f"   📊 Score: {team_score}")
        if runs_scored is not None:
            print(f"   🏃 Runs: {runs_scored}")
        if wickets_taken is not None and wickets_taken > 0:
            print(f"   🎯 Wickets: {wickets_taken}")
        if highlight_info.get("highlight"):
            print(f"   ⭐ HIGHLIGHT: {', '.join(highlight_info.get('highlight_reasons', []))}")
        
        # 🔥 IMMEDIATE JSON UPDATE: Save JSON after each ball record is created
        if self.json_output_path:
            self._save_incremental_json()
            self._save_incremental_over_aggregation()
            print(f"   💾 JSON updated with new ball record")
        
        # Update previous score for next calculation
        if team_score:
            self.cricket_analyzer.previous_score = team_score
    
    def _finalize_current_over(self, current_timestamp: float, frame_number: int):
        """
        Finalize the current over and add to ball-by-ball log
        
        Args:
            current_timestamp: Current video timestamp
            frame_number: Current frame number
        """
        if not self.cricket_analyzer.current_over_value:
            return
        
        # Parse the current over value to check if it's X.0 (over completion)
        over_info = self.cricket_analyzer.parse_over_value(self.cricket_analyzer.current_over_value)
        
        # Don't create records for X.0 values (over completion states)
        if over_info and over_info.get("ball_number") == 0:
            print(f"🏏 OVER FINALIZATION SKIPPED:")
            print(f"   📊 Current Over: {self.cricket_analyzer.current_over_value} (over completion state - no record created)")
            print(f"   ⏳ Waiting for actual ball completion to create record")
            
            # Save JSON immediately after over transition (even without creating record)
            if self.json_output_path:
                self._save_incremental_json()
                self._save_incremental_over_aggregation()
            
            # Update previous score for next calculation
            final_score = self.cricket_analyzer.current_over_score
            self.cricket_analyzer.previous_score = final_score
            return
        
        # Only create record for actual ball completions (X.1, X.2, etc.)
        over_duration = current_timestamp - self.cricket_analyzer.current_over_start_time
        final_score = self.cricket_analyzer.current_over_score
        
        # Calculate runs and wickets for this over
        over_stats = None
        highlight_info = {"highlight": False, "highlight_reasons": [], "highlight_type": None}
        
        if final_score and self.cricket_analyzer.previous_score:
            over_stats = self.cricket_analyzer.calculate_over_stats(
                final_score, self.cricket_analyzer.previous_score
            )
            if over_stats and over_stats.get("calculation_possible"):
                runs_scored = over_stats.get("runs_scored")
                wickets_taken = over_stats.get("wickets_taken")
                highlight_info = self.cricket_analyzer.is_ball_highlight(runs_scored, wickets_taken)
        
        ball_timing_entry = {
            "over_value": self.cricket_analyzer.current_over_value,
            "over_number": over_info.get("over_number") if over_info else None,
            "ball_number": over_info.get("ball_number") if over_info else None,
            "balls_completed": over_info.get("balls_completed") if over_info else None,
            "over_status": over_info.get("over_status", "unknown") if over_info else "unknown",
            "over_description": over_info.get("description", f"Over {self.cricket_analyzer.current_over_value}") if over_info else f"Over {self.cricket_analyzer.current_over_value}",
            "start_time": round(self.cricket_analyzer.current_over_start_time, 2),
            "end_time": round(current_timestamp, 2),
            "duration": round(over_duration, 2),
            "start_frame": getattr(self.cricket_analyzer, 'current_over_start_frame', 'unknown'),
            "end_frame": frame_number,
            "score": final_score,
            "agent_calls": self.cricket_analyzer.current_over_agent_calls,
            "runs_scored": over_stats.get("runs_scored") if over_stats and isinstance(over_stats, dict) else None,
            "wickets_taken": over_stats.get("wickets_taken") if over_stats and isinstance(over_stats, dict) else None,
            "previous_score": self.cricket_analyzer.previous_score,
            "highlight": highlight_info.get("highlight", False),
            "highlight_reasons": highlight_info.get("highlight_reasons", []),
            "highlight_type": highlight_info.get("highlight_type")
        }
        
        self.cricket_analyzer.ball_by_ball_log.append(ball_timing_entry)
        
        print(f"🏏 OVER CHANGE DETECTED:")
        over_desc = over_info.get('description', f"Over {self.cricket_analyzer.current_over_value}") if over_info else f"Over {self.cricket_analyzer.current_over_value}"
        print(f"   📊 Previous Over: {self.cricket_analyzer.current_over_value} ({over_desc})")
        print(f"   ⏰ Previous Over Duration: {over_duration:.2f}s")
        
        # Save JSON immediately after over transition
        if self.json_output_path:
            self._save_incremental_json()
            self._save_incremental_over_aggregation()
        
        # Update previous score for next calculation
        self.cricket_analyzer.previous_score = final_score
    
    def _print_frame_info(self, frame_number: int, timestamp: float, 
                         text_bbox_map: Dict, cricket_analysis: Dict = None):
        """
        Print frame processing information
        
        Args:
            frame_number: Frame number
            timestamp: Video timestamp
            text_bbox_map: Text-bbox mapping
            cricket_analysis: Cricket analysis results
        """
        print(f"\n--- Frame {frame_number} (Time: {timestamp:.2f}s) ---")
        
        # Display text-bbox mapping
        if text_bbox_map:
            print(f"📝 Text-Bbox Mapping ({len(text_bbox_map)} items):")
            for i, (text, bbox_info) in enumerate(text_bbox_map.items(), 1):
                dims = bbox_info['dimensions']
                conf = bbox_info['confidence']
                print(f"   {i:2d}. '{text}' → {dims['width']}x{dims['height']} (conf: {conf:.2f})")
        
        # Display cricket analysis results
        if cricket_analysis:
            print(f"🏏 CRICKET ANALYSIS RESULTS:")
            print(f"   📊 Current Over: {cricket_analysis.get('current_over', 'N/A')}")
            print(f"   🏃 Team Score: {cricket_analysis.get('team_score', 'N/A')}")
            print(f"   🎯 Confidence: {cricket_analysis.get('confidence', 0.0)}")
        
        # Display processing statistics
        ocr_stats = self.ocr_processor.get_optimization_stats()
        cricket_stats = self.cricket_analyzer.get_analysis_stats()
        
        print(f"📈 PROCESSING STATS:")
        print(f"   🤖 Agent calls made: {cricket_stats['agent_calls_made']}")
        print(f"   ⏭️  Total frames skipped: {ocr_stats['total_skips']}")
        print(f"   🏏 Balls tracked: {cricket_stats['balls_tracked']}")
        print(f"   🏏 Current over: {cricket_stats['current_over'] or 'None'}")
    
    def _add_frame_overlay(self, frame: np.ndarray, frame_number: int, timestamp: float,
                          text_count: int, cricket_analysis: Dict = None) -> np.ndarray:
        """
        Add information overlay to the frame
        
        Args:
            frame: Input frame
            frame_number: Frame number
            timestamp: Video timestamp
            text_count: Number of texts detected
            cricket_analysis: Cricket analysis results
            
        Returns:
            Frame with overlay
        """
        overlay_y = 30
        overlay_color = (255, 255, 255)  # White text
        overlay_font = cv2.FONT_HERSHEY_SIMPLEX
        overlay_scale = 0.7
        overlay_thickness = 2
        
        # Frame info
        frame_info = f"Frame: {frame_number} | Time: {timestamp:.2f}s | Texts: {text_count}"
        cv2.putText(frame, frame_info, (10, overlay_y), 
                   overlay_font, overlay_scale, overlay_color, overlay_thickness)
        
        # Cricket info
        if cricket_analysis:
            cricket_info = f"Over: {cricket_analysis.get('current_over', 'N/A')} | Score: {cricket_analysis.get('team_score', 'N/A')}"
            cv2.putText(frame, cricket_info, (10, overlay_y + 30), 
                       overlay_font, overlay_scale, (0, 255, 0), overlay_thickness)
        
        # OCR Region info
        if hasattr(self.ocr_processor, 'ocr_region') and self.ocr_processor.ocr_region:
            region = self.ocr_processor.ocr_region
            region_type = region.get('type', 'fraction')
            if region_type == 'fraction':
                region_info = f"OCR Region: {region['x']:.1f},{region['y']:.1f} {region['width']:.1f}x{region['height']:.1f} (fraction)"
            else:
                region_info = f"OCR Region: {region['x']},{region['y']} {region['width']}x{region['height']} (pixels)"
            cv2.putText(frame, region_info, (10, overlay_y + 60), 
                       overlay_font, 0.6, (255, 0, 255), overlay_thickness)  # Magenta text
        else:
            region_info = "OCR Region: Full Frame"
            cv2.putText(frame, region_info, (10, overlay_y + 60), 
                       overlay_font, 0.6, (255, 255, 255), overlay_thickness)  # White text
        
        # Processing stats
        ocr_stats = self.ocr_processor.get_optimization_stats()
        cricket_stats = self.cricket_analyzer.get_analysis_stats()
        stats_info = f"Agent calls: {cricket_stats['agent_calls_made']} | Skipped: {ocr_stats['total_skips']}"
        cv2.putText(frame, stats_info, (10, overlay_y + 60), 
                   overlay_font, overlay_scale, overlay_color, overlay_thickness)
        
        return frame
    
    def _save_incremental_json(self):
        """Save incremental ball-by-ball JSON with real-time updates"""
        if not self.json_output_path:
            return
        
        try:
            json_data = {
                "ball_by_ball_log": self.cricket_analyzer.ball_by_ball_log,
                "total_balls_tracked": len(self.cricket_analyzer.ball_by_ball_log),
                "current_over": self.cricket_analyzer.current_over_value,
                "current_over_start_time": self.cricket_analyzer.current_over_start_time,
                "current_over_score": self.cricket_analyzer.current_over_score,
                "score_search_active": self.cricket_analyzer.score_search_active,
                "last_updated": time.time(),
                "last_updated_readable": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                "status": "processing"
            }
            
            with open(self.json_output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Warning: Failed to save incremental JSON: {e}")
    
    def _save_incremental_over_aggregation(self):
        """Save incremental over aggregation JSON"""
        if not self.json_output_path:
            return
        
        try:
            from pathlib import Path
            
            # Generate over aggregation path in output directory
            json_path = Path(self.json_output_path)
            base_name = json_path.stem.replace('_ball_by_ball_log', '')
            over_aggregation_path = json_path.parent / f"{base_name}_over_aggregation.json"
            
            # Group balls by over number
            grouped_overs = {}
            seen_over_values = set()  # Track seen over_values to prevent duplicates
            
            for entry in self.cricket_analyzer.ball_by_ball_log:
                over_number = entry.get('over_number')
                over_value = entry.get('over_value')
                
                if over_number is None or over_value is None:
                    continue
                
                # 🔥 DUPLICATE PREVENTION: Skip if over_value already seen
                if over_value in seen_over_values:
                    print(f"⚠️  DUPLICATE DETECTED in aggregation: {over_value} - skipping")
                    continue
                
                seen_over_values.add(over_value)
                
                if over_number not in grouped_overs:
                    grouped_overs[over_number] = []
                grouped_overs[over_number].append(entry)
            
            # Calculate over statistics
            over_statistics = {}
            for over_number, balls in grouped_overs.items():
                balls.sort(key=lambda x: x.get('ball_number', 0))
                
                over_stats = {
                    "total_balls": len(balls),
                    "over_duration": sum(ball.get('duration', 0) for ball in balls),
                    "total_runs_in_over": sum(ball.get('runs_scored', 0) or 0 for ball in balls),
                    "total_wickets_in_over": sum(ball.get('wickets_taken', 0) or 0 for ball in balls),
                    "balls_data": balls,
                    "over_highlights": [],
                    "first_ball_time": balls[0].get('start_time') if balls else None,
                    "last_ball_time": balls[-1].get('end_time') if balls else None
                }
                
                # Collect highlights
                for ball in balls:
                    if ball.get('highlight', False):
                        over_stats["over_highlights"].append({
                            "ball_number": ball.get('ball_number'),
                            "highlight_type": ball.get('highlight_type'),
                            "highlight_reasons": ball.get('highlight_reasons', []),
                            "runs": ball.get('runs_scored', 0) or 0,
                            "wickets": ball.get('wickets_taken', 0) or 0
                        })
                
                over_stats["is_highlight_over"] = len(over_stats["over_highlights"]) > 0
                over_stats["highlight_balls_count"] = len(over_stats["over_highlights"])
                over_statistics[over_number] = over_stats
            
            aggregation_data = {
                "total_overs": len(grouped_overs),
                "over_statistics": over_statistics,
                "last_updated": time.time(),
                "status": "processing",
                "current_over_being_processed": self.cricket_analyzer.current_over_value
            }
            
            with open(over_aggregation_path, 'w', encoding='utf-8') as f:
                json.dump(aggregation_data, f, indent=2)
                
            print(f"   📊 Real-time over aggregation JSON updated: {over_aggregation_path}")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to save incremental over aggregation JSON: {e}")
    
    def finalize_processing(self, final_timestamp: float, final_frame: int):
        """
        Finalize processing and save final data
        
        Args:
            final_timestamp: Final video timestamp
            final_frame: Final frame number
        """
        # Finalize current over if exists and it's not an X.0 value
        if self.cricket_analyzer.current_over_value:
            # Check if current over is X.0 (over completion state)
            over_info = self.cricket_analyzer.parse_over_value(self.cricket_analyzer.current_over_value)
            if over_info and over_info.get("ball_number") == 0:
                print(f"🏁 FINAL PROCESSING: Skipping finalization of {self.cricket_analyzer.current_over_value} (over completion state)")
            else:
                print(f"🏁 FINAL PROCESSING: Finalizing {self.cricket_analyzer.current_over_value}")
                self._finalize_current_over(final_timestamp, final_frame)
        else:
            print(f"🏁 FINAL PROCESSING: No current over to finalize")
        
        # Clean up any duplicate records
        self._cleanup_duplicate_records()
        
        # Save final JSON files
        if self.json_output_path:
            self._save_final_json()
            self._save_final_over_aggregation()
    
    def _save_final_json(self):
        """Save final ball-by-ball JSON"""
        try:
            with open(self.json_output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "ball_by_ball_log": self.cricket_analyzer.ball_by_ball_log,
                    "total_balls_tracked": len(self.cricket_analyzer.ball_by_ball_log),
                    "current_over": self.cricket_analyzer.current_over_value,
                    "current_over_start_time": self.cricket_analyzer.current_over_start_time,
                    "current_over_score": self.cricket_analyzer.current_over_score,
                    "score_search_active": False,
                    "last_updated": time.time(),
                    "status": "completed"
                }, f, indent=2)
            print(f"📁 Final ball-by-ball timing log saved: {self.json_output_path}")
        except Exception as e:
            print(f"❌ Error saving final ball-by-ball timing log: {e}")
    
    def _save_final_over_aggregation(self):
        """Save final over aggregation JSON"""
        try:
            from pathlib import Path
            
            # Generate over aggregation path in output directory
            json_path = Path(self.json_output_path)
            base_name = json_path.stem.replace('_ball_by_ball_log', '')
            over_aggregation_path = json_path.parent / f"{base_name}_over_aggregation.json"
            
            # Use the same logic as incremental save but mark as completed
            self._save_incremental_over_aggregation()
            
            # Update status to completed
            with open(over_aggregation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data["status"] = "completed"
            data["final_over_processed"] = self.cricket_analyzer.current_over_value
            
            with open(over_aggregation_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            print(f"📁 Final over aggregation JSON saved: {over_aggregation_path}")
            
        except Exception as e:
            print(f"❌ Error saving final over aggregation JSON: {e}")
    
    def force_json_update(self, reason: str = "manual"):
        """
        Force an immediate JSON update with reason logging
        
        Args:
            reason: Reason for the forced update
        """
        if self.json_output_path:
            print(f"🔄 FORCING JSON UPDATE: {reason}")
            self._save_incremental_json()
            self._save_incremental_over_aggregation()
    
    def get_processing_stats(self) -> Dict:
        """
        Get comprehensive processing statistics
        
        Returns:
            Dictionary with processing metrics
        """
        ocr_stats = self.ocr_processor.get_optimization_stats()
        cricket_stats = self.cricket_analyzer.get_analysis_stats()
        
        return {
            'frames_processed': self.frames_processed,
            'frames_skipped': self.frames_skipped,
            'total_processing_time': self.total_processing_time,
            'average_frame_time': self.total_processing_time / self.frames_processed if self.frames_processed > 0 else 0,
            'processing_fps': self.frames_processed / self.total_processing_time if self.total_processing_time > 0 else 0,
            'ocr_stats': ocr_stats,
            'cricket_stats': cricket_stats
        }
