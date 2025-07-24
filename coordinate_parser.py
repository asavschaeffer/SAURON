"""
SAURON Coordinate Parser - The Eye That Sees All Mouse Movement

Ultra-fast ANSI escape sequence parser for real-time gesture recognition.
Designed for sub-millisecond response times on minimal hardware.
"""

import re
import time
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

@dataclass
class Coordinate:
    """Single coordinate point with timestamp"""
    row: int
    col: int
    timestamp: float
    button_state: str = 'm'  # 'm' = move, 'M' = click

class GestureType(Enum):
    CIRCLE = "circle"
    LINE = "line" 
    RECTANGLE = "rectangle"
    STAB = "stab"
    SLASH = "slash"
    UNKNOWN = "unknown"

class CoordinateParser:
    """Lightning-fast ANSI coordinate parser"""
    
    # Pre-compiled regex for maximum speed
    COORD_PATTERN = re.compile(r'<(\d+);(\d+);(\d+)([mM])')
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.coordinates: List[Coordinate] = []
        self.gesture_buffer: List[Coordinate] = []
        
    def parse_stream(self, ansi_stream: str) -> List[Coordinate]:
        """Parse ANSI stream into coordinates with microsecond timestamps"""
        coords = []
        current_time = time.perf_counter()
        
        # Find all coordinate matches
        for match in self.COORD_PATTERN.finditer(ansi_stream):
            row = int(match.group(1))
            col = int(match.group(2))
            button = match.group(4)
            
            coord = Coordinate(
                row=row,
                col=col, 
                timestamp=current_time,
                button_state=button
            )
            coords.append(coord)
            
        # Add to buffer and maintain size
        self.coordinates.extend(coords)
        if len(self.coordinates) > self.buffer_size:
            self.coordinates = self.coordinates[-self.buffer_size:]
            
        return coords
    
    def get_recent_gesture(self, time_window: float = 2.0) -> List[Coordinate]:
        """Get coordinates from the last N seconds for gesture analysis"""
        current_time = time.perf_counter()
        cutoff_time = current_time - time_window
        
        return [c for c in self.coordinates if c.timestamp >= cutoff_time]
    
    def clear_buffer(self):
        """Clear coordinate buffer"""
        self.coordinates.clear()

class GestureClassifier:
    """Classify mouse gestures from coordinate sequences"""
    
    @staticmethod
    def classify_gesture(coords: List[Coordinate]) -> GestureType:
        """Classify a sequence of coordinates into gesture type"""
        if len(coords) < 3:
            return GestureType.UNKNOWN
            
        # Extract positions
        positions = [(c.row, c.col) for c in coords]
        
        # Quick geometric analysis
        if GestureClassifier._is_circle(positions):
            return GestureType.CIRCLE
        elif GestureClassifier._is_rectangle(positions):
            return GestureType.RECTANGLE  
        elif GestureClassifier._is_line(positions):
            if GestureClassifier._is_stab(positions):
                return GestureType.STAB
            else:
                return GestureType.SLASH
        
        return GestureType.UNKNOWN
    
    @staticmethod
    def _is_circle(positions: List[Tuple[int, int]]) -> bool:
        """Detect circular motion"""
        if len(positions) < 8:
            return False
            
        # Check if path roughly returns to start
        start = positions[0]
        end = positions[-1]
        
        # Distance from start to end should be small for circle
        distance = ((start[0] - end[0])**2 + (start[1] - end[1])**2)**0.5
        
        # Calculate path length
        path_length = sum(
            ((positions[i][0] - positions[i-1][0])**2 + 
             (positions[i][1] - positions[i-1][1])**2)**0.5
            for i in range(1, len(positions))
        )
        
        # Circle: small start-end distance, decent path length
        return distance < 10 and path_length > 30
    
    @staticmethod
    def _is_rectangle(positions: List[Tuple[int, int]]) -> bool:
        """Detect rectangular motion"""
        if len(positions) < 10:
            return False
            
        # Look for 4 distinct direction changes
        directions = []
        for i in range(1, len(positions)):
            dr = positions[i][0] - positions[i-1][0]
            dc = positions[i][1] - positions[i-1][1]
            
            if abs(dr) > abs(dc):
                directions.append('vertical')
            else:
                directions.append('horizontal')
        
        # Count direction changes
        changes = sum(1 for i in range(1, len(directions)) 
                     if directions[i] != directions[i-1])
        
        return changes >= 3  # At least 3 corners
    
    @staticmethod
    def _is_line(positions: List[Tuple[int, int]]) -> bool:
        """Detect linear motion"""
        if len(positions) < 3:
            return False
            
        # Check if points are roughly collinear
        start = positions[0]
        end = positions[-1]
        
        # Calculate total deviation from straight line
        total_deviation = 0
        for pos in positions[1:-1]:
            # Distance from point to line start->end
            deviation = abs((end[1] - start[1]) * pos[0] - 
                          (end[0] - start[0]) * pos[1] + 
                          end[0] * start[1] - end[1] * start[0]) / \
                       ((end[1] - start[1])**2 + (end[0] - start[0])**2)**0.5
            total_deviation += deviation
            
        avg_deviation = total_deviation / len(positions[1:-1])
        return avg_deviation < 5  # Roughly straight
    
    @staticmethod
    def _is_stab(positions: List[Tuple[int, int]]) -> bool:
        """Detect quick stab motion (short, fast line)"""
        if len(positions) > 5:
            return False  # Stabs are quick
            
        start = positions[0]
        end = positions[-1]
        distance = ((start[0] - end[0])**2 + (start[1] - end[1])**2)**0.5
        
        return distance < 15  # Short motion = stab

# Performance test function
def benchmark_parser():
    """Test parsing speed on sample data"""
    sample_stream = ("<35;68;18m<35;67;18m<35;65;18m<35;64;18m<35;62;18m" +
                    "<35;60;18m<35;59;18m<35;58;18m<35;57;18m<35;56;18m")
    
    parser = CoordinateParser()
    
    start_time = time.perf_counter()
    for _ in range(1000):
        coords = parser.parse_stream(sample_stream)
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / 1000 * 1000  # Convert to ms
    print(f"Average parse time: {avg_time:.3f}ms per stream")
    print(f"That's {1000/avg_time:.0f} streams per second!")

if __name__ == "__main__":
    # Quick demo
    print("🧿 SAURON Coordinate Parser - The Eye Awakens 🧿")
    benchmark_parser()