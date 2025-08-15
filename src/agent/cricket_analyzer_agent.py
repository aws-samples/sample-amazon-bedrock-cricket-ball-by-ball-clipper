"""
Cricket Analyzer Agent Module
Handles cricket-specific analysis using AWS Strands and pattern matching
Located in src/agent/ to emphasize its AI agent functionality
"""

import json
import time
import re
from typing import List, Dict, Optional, Tuple

try:
    from strands import Agent
    from strands.models import BedrockModel
except ImportError:
    print("Warning: strands library not found. Cricket analysis will be limited.")
    Agent = None
    BedrockModel = None


class CricketAnalyzer:
    """Handles cricket-specific analysis and data processing"""
    
    def __init__(self):
        """Initialize cricket analyzer with AI agent"""
        self.agent_calls_made = 0
        self.last_successful_analysis = None
        
        # Ball and over tracking
        self.ball_by_ball_log = []
        self.current_over_value = None
        self.current_over_start_time = None
        self.current_over_start_frame = None
        self.current_over_agent_calls = 0
        self.current_over_score = None
        self.score_search_active = False
        self.previous_score = None
        self.last_ball_score = None
        
        # Initialize AI agent
        if Agent and BedrockModel:
            print("Setting up AWS Strands Agent for cricket analysis...")
            self.cricket_agent = self._create_cricket_agent()
            print("Cricket Analyzer initialized successfully!")
        else:
            self.cricket_agent = None
            print("Cricket Analyzer initialized without AI agent (strands not available)")
    
    def _create_cricket_agent(self):
        """Create a specialized cricket analysis agent"""
        model_id = "us.amazon.nova-lite-v1:0"
        bdm = BedrockModel(model_id=model_id)
        
        agent = Agent(
            model=bdm,
            system_prompt="""You are an expert cricket graphics analyzer. Your task is to analyze OCR text extracted from cricket broadcast graphics and identify current over and score information.

CORE PRINCIPLES:
1. Focus ONLY on cricket over and score information
2. Identify current over (format: X.Y where X is over number, Y is ball number)
3. Extract team score (format: runs/wickets or just runs)
4. Ignore non-cricket text like advertisements, player names, etc.
5. Return structured JSON response only

CRICKET OVER FORMAT:
- Current over: "3.4" means 3rd over, 4th ball
- Current over: "15.2" means 15th over, 2nd ball
- Current over: "0.6" means 0th over, 6th ball (end of over)
- Current over: "1.0" means 1 over complete
- current over: "2.0" means second over is complete.
- Over value will always have one value after decimal. if you get over value as x.yx ("0.44"), then return only x.y("0.4").
- if input has multiple values of x.y in the input text, then please consider exact x.y value as current over, Say if "0.1" and "1-0 0.1", consider 0.1 as over.
- Over format will not be x-y, say "2-0" format, it may be score , do not interpret as Over.

SCORE FORMAT:
- Team score: "156/3" means 156 runs, 3 wickets
- Team score: "89/2" means 89 runs, 2 wickets
- Team score: "8-0" means 8 runs, 0 wickets
- Team score: "120-3" means 120 runs, 3 wickets
- Team score: "234" means 234 runs (wickets not shown)

CRITICAL: You MUST respond with ONLY a valid JSON object. No additional text, explanations, or formatting outside the JSON.

RESPONSE FORMAT (JSON ONLY):
{
    "current_over": "X.Y or null if not found",
    "team_score": "runs/wickets or runs or null if not found", 
    "confidence": 0.95,
    "detected_texts_used": ["list of OCR texts that were relevant"],
    "analysis_notes": "brief explanation of what was identified"
}"""
        )
        
        return agent
    
    def is_over_format(self, text: str) -> bool:
        """
        Check if text matches cricket over format (X.Y)
        
        Args:
            text: Text to check
            
        Returns:
            True if text matches over format
        """
        over_pattern = r'^\d+\.\d+$'
        return bool(re.match(over_pattern, text.strip()))
    
    def extract_team_score_from_texts(self, text_bbox_map: Dict[str, Dict]) -> Optional[str]:
        """
        Extract team score from OCR texts using pattern matching
        
        Args:
            text_bbox_map: Text-bbox mapping
            
        Returns:
            Team score string if found
        """
        score_patterns = [
            r'^\d+/\d+$',  # Format: 156/3
            r'^\d+$',      # Format: 156 (just runs)
            r'^\d+-\d+$',  # Format: 156-3 (alternative format)
        ]
        
        for text, bbox_info in text_bbox_map.items():
            text_clean = text.strip()
            for pattern in score_patterns:
                if re.match(pattern, text_clean):
                    # Additional validation: score should be reasonable numbers
                    if '/' in text_clean:
                        runs, wickets = text_clean.split('/')
                        if runs.isdigit() and wickets.isdigit():
                            runs_num, wickets_num = int(runs), int(wickets)
                            if 0 <= runs_num <= 500 and 0 <= wickets_num <= 10:
                                return text_clean
                    elif '-' in text_clean:
                        runs, wickets = text_clean.split('-')
                        if runs.isdigit() and wickets.isdigit():
                            runs_num, wickets_num = int(runs), int(wickets)
                            if 0 <= runs_num <= 500 and 0 <= wickets_num <= 10:
                                return text_clean
                    elif text_clean.isdigit():
                        runs_num = int(text_clean)
                        if 0 <= runs_num <= 500:
                            return text_clean
        
        return None
    
    def analyze_cricket_texts(self, ocr_texts: List[str], text_bbox_map: Dict[str, Dict] = None) -> Optional[Dict]:
        """
        Analyze OCR texts using AWS Strands for cricket information
        
        Args:
            ocr_texts: List of OCR detected texts
            text_bbox_map: Text-bbox mapping for pattern matching fallback
            
        Returns:
            Cricket analysis result or None if analysis fails
        """
        if not ocr_texts:
            return None
        
        # Try AI analysis first if available
        if self.cricket_agent:
            try:
                print(f"🤖 RUNNING STRANDS ANALYSIS with {len(ocr_texts)} texts")
                self.agent_calls_made += 1
                self.current_over_agent_calls += 1
                
                prompt = f"""Analyze these OCR texts from cricket broadcast graphics:

OCR Texts: {ocr_texts}

Please identify the current over and team score information from these texts and return the structured JSON response."""
                
                analysis_start = time.time()
                agent_response = self.cricket_agent(prompt)
                analysis_time = time.time() - analysis_start
                
                print(f"⏱️  Strands analysis time: {analysis_time:.3f}s")
                
                # Handle different response types
                if hasattr(agent_response, 'content'):
                    response_text = agent_response.content
                elif hasattr(agent_response, 'text'):
                    response_text = agent_response.text
                elif isinstance(agent_response, str):
                    response_text = agent_response
                else:
                    response_text = str(agent_response)
                
                # Clean response text
                cleaned_response = response_text.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                    if cleaned_response.endswith('```'):
                        cleaned_response = cleaned_response[:-3]
                elif cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response[3:]
                    if cleaned_response.endswith('```'):
                        cleaned_response = cleaned_response[:-3]
                
                cleaned_response = cleaned_response.strip()
                
                # Parse JSON response
                cricket_analysis = json.loads(cleaned_response)
                
                # Validate and cache successful analysis
                if isinstance(cricket_analysis, dict):
                    current_over = cricket_analysis.get('current_over')
                    team_score = cricket_analysis.get('team_score')
                    
                    if current_over or team_score:
                        self.last_successful_analysis = cricket_analysis
                        return cricket_analysis
                
            except Exception as e:
                print(f"❌ AI analysis error: {e}")
        
        # Fallback to pattern matching if AI fails or unavailable
        if text_bbox_map:
            return self._pattern_matching_analysis(ocr_texts, text_bbox_map)
        
        return None
    
    def _pattern_matching_analysis(self, ocr_texts: List[str], text_bbox_map: Dict[str, Dict]) -> Optional[Dict]:
        """
        Fallback pattern matching analysis when AI is unavailable
        
        Args:
            ocr_texts: OCR texts
            text_bbox_map: Text-bbox mapping
            
        Returns:
            Cricket analysis result
        """
        current_over = None
        team_score = None
        detected_texts_used = []
        
        # Look for over format
        for text in ocr_texts:
            if self.is_over_format(text):
                current_over = text
                detected_texts_used.append(text)
                break
        
        # Look for score
        team_score = self.extract_team_score_from_texts(text_bbox_map)
        if team_score:
            detected_texts_used.append(team_score)
        
        if current_over or team_score:
            return {
                "current_over": current_over,
                "team_score": team_score,
                "confidence": 0.8,  # Lower confidence for pattern matching
                "detected_texts_used": detected_texts_used,
                "analysis_notes": "Pattern matching analysis (AI unavailable)"
            }
        
        return None
    
    def parse_over_value(self, over_value: str) -> Dict:
        """
        Parse over value to extract over number and ball number
        
        Args:
            over_value: Over value string (e.g., "4.5", "3.1", "4.0")
            
        Returns:
            Dictionary with over_number, ball_number, and status
        """
        try:
            if '.' in over_value:
                over_part, ball_part = over_value.split('.')
                over_part_num = int(over_part)
                ball_part_num = int(ball_part)
                
                if ball_part_num == 0:
                    # X.0 means over X is completed (6 balls completed)
                    return {
                        "over_number": over_part_num,
                        "ball_number": 6,
                        "balls_completed": 6,
                        "over_status": "completed",
                        "description": f"Over {over_part_num} completed (6 balls)"
                    }
                else:
                    # X.Y means over (X+1), ball Y completed
                    actual_over_number = over_part_num + 1
                    return {
                        "over_number": actual_over_number,
                        "ball_number": ball_part_num,
                        "balls_completed": ball_part_num,
                        "over_status": "in_progress",
                        "description": f"Over {actual_over_number}, ball {ball_part_num} completed"
                    }
            else:
                # Just a number means over completed
                over_number = int(over_value)
                return {
                    "over_number": over_number,
                    "ball_number": 6,
                    "balls_completed": 6,
                    "over_status": "completed",
                    "description": f"Over {over_number} completed (6 balls)"
                }
        except (ValueError, IndexError):
            return {
                "over_number": None,
                "ball_number": None,
                "balls_completed": None,
                "over_status": "unknown",
                "description": f"Unable to parse over value: {over_value}"
            }
    
    def parse_score(self, score_str: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Parse score string to extract runs and wickets
        
        Args:
            score_str: Score string (e.g., "156/3", "156-3", "156")
            
        Returns:
            Tuple of (runs, wickets) or (None, None) if parsing fails
        """
        if not score_str:
            return None, None
        
        # Handle different score formats
        if '/' in score_str:
            parts = score_str.split('/')
        elif '-' in score_str:
            parts = score_str.split('-')
        else:
            # Just runs, no wickets shown
            try:
                runs = int(score_str.strip())
                return runs, None
            except ValueError:
                return None, None
        
        if len(parts) == 2:
            try:
                runs = int(parts[0].strip())
                wickets = int(parts[1].strip())
                return runs, wickets
            except ValueError:
                return None, None
        
        return None, None
    
    def is_ball_highlight(self, ball_runs: int = None, ball_wickets: int = None) -> Dict:
        """
        Determine if the ball qualifies as a cricket highlight
        
        Args:
            ball_runs: Number of runs scored on the ball
            ball_wickets: Number of wickets taken on the ball
            
        Returns:
            Dictionary with highlight status and reasons
        """
        highlight_reasons = []
        is_highlight = False
        highlight_type = None
        
        # Boundary hits
        if ball_runs is not None and ball_runs > 0:
            if ball_runs == 6:
                highlight_reasons.append("Six hit")
                is_highlight = True
                highlight_type = "boundary"
            elif ball_runs == 4:
                highlight_reasons.append("Four hit")
                is_highlight = True
                highlight_type = "boundary"
            elif ball_runs >= 7:
                highlight_reasons.append(f"Exceptional ball ({ball_runs} runs)")
                is_highlight = True
                highlight_type = "boundary"
        
        # Wicket taken
        if ball_wickets is not None and ball_wickets >= 1:
            if ball_wickets == 1:
                highlight_reasons.append("Wicket taken")
            else:
                highlight_reasons.append(f"Multiple wickets on one ball ({ball_wickets} wickets)")
            is_highlight = True
            if highlight_type != "boundary":
                highlight_type = "wicket"
        
        # Special combinations
        if (ball_runs is not None and ball_runs >= 4 and 
            ball_wickets is not None and ball_wickets >= 1):
            highlight_reasons.append("Boundary + wicket on same ball")
            is_highlight = True
            highlight_type = "special"
        
        # Ensure highlight_type is None if no highlight
        if not is_highlight:
            highlight_type = None
        
        return {
            "highlight": is_highlight,
            "highlight_reasons": highlight_reasons,
            "highlight_type": highlight_type
        }
    
    def calculate_over_stats(self, current_score: str, previous_score: str) -> Dict:
        """
        Calculate runs and wickets scored during an over
        
        Args:
            current_score: Current score string
            previous_score: Previous score string
            
        Returns:
            Dictionary with runs_scored, wickets_taken, and calculation details
        """
        current_runs, current_wickets = self.parse_score(current_score)
        previous_runs, previous_wickets = self.parse_score(previous_score)
        
        result = {
            "runs_scored": None,
            "wickets_taken": None,
            "current_score_parsed": {"runs": current_runs, "wickets": current_wickets},
            "previous_score_parsed": {"runs": previous_runs, "wickets": previous_wickets},
            "calculation_possible": False
        }
        
        # Calculate runs scored
        if current_runs is not None and previous_runs is not None:
            result["runs_scored"] = current_runs - previous_runs
            result["calculation_possible"] = True
        
        # Calculate wickets taken
        if current_wickets is not None and previous_wickets is not None:
            result["wickets_taken"] = current_wickets - previous_wickets
            result["calculation_possible"] = True
        
        return result
    
    def get_analysis_stats(self) -> Dict:
        """
        Get cricket analysis statistics
        
        Returns:
            Dictionary with analysis metrics
        """
        unique_overs = len(set(entry.get('over_number') for entry in self.ball_by_ball_log if entry.get('over_number')))
        total_highlights = sum(1 for entry in self.ball_by_ball_log if entry.get('highlight', False))
        
        return {
            'agent_calls_made': self.agent_calls_made,
            'balls_tracked': len(self.ball_by_ball_log),
            'unique_overs': unique_overs,
            'total_highlights': total_highlights,
            'highlight_rate': total_highlights / len(self.ball_by_ball_log) * 100 if self.ball_by_ball_log else 0,
            'current_over': self.current_over_value,
            'ai_agent_available': self.cricket_agent is not None
        }
