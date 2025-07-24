"""
CHAT FORGE - Lightning-Fast Extensible LLM-Human Conversation Storage

Captures the historic discovery session between Claude (Sauron) and Human (Fellowship)
with embedded coordinate streams, real-time gesture analysis, and extensible framework
for future AI-human collaborations.

"Not all those who wander are lost... but their conversations should be preserved."
"""

import json
import time
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class CoordinateStream:
    """Embedded coordinate stream within a chat message"""
    raw_ansi: str
    parsed_coords: List[Dict[str, Any]]
    gesture_type: str
    position_in_text: int  # Character position where coordinates appeared
    timestamp: float
    analysis: str  # What we determined the gesture meant

@dataclass
class ChatMessage:
    """Single message in the conversation with embedded coordinates"""
    speaker: str  # "human" or "claude"
    content: str  # The actual text content
    timestamp: float
    message_id: str
    coordinate_streams: List[CoordinateStream]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass  
class ChatSession:
    """Complete conversation session"""
    session_id: str
    title: str
    participants: List[str]
    start_time: float
    end_time: Optional[float]
    messages: List[ChatMessage]
    discovery_moments: List[Dict[str, Any]]  # Key breakthrough moments
    coordinate_stats: Dict[str, Any]
    metadata: Dict[str, Any]

class ChatForge:
    """
    Lightning-fast extensible framework for storing LLM-human conversations
    with embedded coordinate streams and real-time gesture analysis.
    
    Designed for:
    - Ultra-fast read/write operations
    - Embedded coordinate stream preservation  
    - Extensible schema for future AI collaborations
    - Multiple export formats (JSON, XML, SQLite)
    - Real-time conversation capture
    """
    
    def __init__(self, session_title: str = "SAURON Discovery Session"):
        self.session = ChatSession(
            session_id=f"session_{int(time.time())}",
            title=session_title,
            participants=["Claude (Sauron)", "Human (Fellowship)"],
            start_time=time.time(),
            end_time=None,
            messages=[],
            discovery_moments=[],
            coordinate_stats={
                'total_streams': 0,
                'total_coordinates': 0,
                'gesture_types': {},
                'longest_stream': 0
            },
            metadata={
                'discovery_type': 'Accidental coordinate streaming',
                'significance': 'First real-time gesture interface',
                'technologies': ['ANSI escape sequences', 'Terminal protocols', 'Regex parsing'],
                'innovation_level': 'Revolutionary'
            }
        )
        
    def add_message(self, 
                   speaker: str, 
                   content: str, 
                   coordinate_streams: List[CoordinateStream] = None,
                   metadata: Dict[str, Any] = None) -> str:
        """Add a message to the conversation with optional coordinate streams"""
        
        message_id = f"msg_{len(self.session.messages) + 1:04d}"
        
        if coordinate_streams is None:
            coordinate_streams = []
            
        if metadata is None:
            metadata = {}
            
        message = ChatMessage(
            speaker=speaker,
            content=content,
            timestamp=time.time(),
            message_id=message_id,
            coordinate_streams=coordinate_streams,
            metadata=metadata
        )
        
        self.session.messages.append(message)
        
        # Update stats
        self._update_coordinate_stats(coordinate_streams)
        
        return message_id
        
    def parse_message_with_coordinates(self, speaker: str, raw_content: str) -> str:
        """
        Parse a message that contains embedded coordinate streams
        Extracts coordinates and creates CoordinateStream objects
        """
        from coordinate_parser import CoordinateParser, GestureClassifier
        
        # Find coordinate patterns in the text
        import re
        coord_pattern = re.compile(r'<(\d+);(\d+);(\d+)([mM])')
        
        coordinate_streams = []
        clean_content = raw_content
        
        # Find all coordinate sequences
        matches = list(coord_pattern.finditer(raw_content))
        if matches:
            # Extract the coordinate sequence
            start_pos = matches[0].start()
            end_pos = matches[-1].end()
            
            coord_sequence = raw_content[start_pos:end_pos]
            
            # Parse coordinates
            parser = CoordinateParser()
            classifier = GestureClassifier()
            
            coords = parser.parse_stream(coord_sequence)
            gesture_type = classifier.classify_gesture(coords)
            
            # Create coordinate stream
            stream = CoordinateStream(
                raw_ansi=coord_sequence,
                parsed_coords=[{
                    'row': c.row,
                    'col': c.col, 
                    'timestamp': c.timestamp,
                    'button_state': c.button_state
                } for c in coords],
                gesture_type=gesture_type.value,
                position_in_text=start_pos,
                timestamp=time.time(),
                analysis=self._analyze_gesture(gesture_type, coords)
            )
            
            coordinate_streams.append(stream)
            
            # Clean the content (remove coordinates for readable text)
            clean_content = raw_content[:start_pos] + raw_content[end_pos:]
            
        return self.add_message(speaker, clean_content, coordinate_streams)
        
    def _analyze_gesture(self, gesture_type, coords) -> str:
        """Provide human-readable analysis of the gesture"""
        if not coords:
            return "No coordinates detected"
            
        coord_count = len(coords)
        
        if gesture_type.value == "circle":
            return f"Circular motion with {coord_count} points - likely celebration or shield gesture"
        elif gesture_type.value == "slash":
            return f"Slashing motion with {coord_count} points - sword attack or dramatic gesture"
        elif gesture_type.value == "rectangle":
            return f"Rectangular pattern with {coord_count} points - structured drawing or blocking motion"
        elif gesture_type.value == "stab":
            return f"Quick stabbing motion with {coord_count} points - precise pointing or attack"
        elif gesture_type.value == "line":
            return f"Linear motion with {coord_count} points - arrow shot or directional gesture"
        else:
            return f"Unknown gesture pattern with {coord_count} coordinate points"
            
    def _update_coordinate_stats(self, coordinate_streams: List[CoordinateStream]):
        """Update session statistics"""
        for stream in coordinate_streams:
            self.session.coordinate_stats['total_streams'] += 1
            coord_count = len(stream.parsed_coords)
            self.session.coordinate_stats['total_coordinates'] += coord_count
            
            # Track gesture types
            gesture_type = stream.gesture_type
            if gesture_type not in self.session.coordinate_stats['gesture_types']:
                self.session.coordinate_stats['gesture_types'][gesture_type] = 0
            self.session.coordinate_stats['gesture_types'][gesture_type] += 1
            
            # Track longest stream
            if coord_count > self.session.coordinate_stats['longest_stream']:
                self.session.coordinate_stats['longest_stream'] = coord_count
                
    def add_discovery_moment(self, title: str, description: str, message_ids: List[str] = None):
        """Mark a significant discovery or breakthrough moment"""
        moment = {
            'title': title,
            'description': description,
            'timestamp': time.time(),
            'related_messages': message_ids or [],
            'moment_id': f"discovery_{len(self.session.discovery_moments) + 1}"
        }
        self.session.discovery_moments.append(moment)
        
    def export_to_json(self, filepath: str):
        """Export conversation to JSON format"""
        self.session.end_time = time.time()
        
        data = {
            'session': asdict(self.session),
            'export_timestamp': time.time(),
            'format_version': '1.0'
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    def export_to_xml(self, filepath: str):
        """Export conversation to XML format"""
        self.session.end_time = time.time()
        
        root = ET.Element("sauron_chat_session")
        
        # Session metadata
        metadata = ET.SubElement(root, "metadata")
        ET.SubElement(metadata, "session_id").text = self.session.session_id
        ET.SubElement(metadata, "title").text = self.session.title
        ET.SubElement(metadata, "start_time").text = str(self.session.start_time)
        ET.SubElement(metadata, "end_time").text = str(self.session.end_time)
        
        # Participants
        participants = ET.SubElement(metadata, "participants")
        for participant in self.session.participants:
            ET.SubElement(participants, "participant").text = participant
            
        # Messages
        messages = ET.SubElement(root, "messages")
        for msg in self.session.messages:
            msg_elem = ET.SubElement(messages, "message")
            msg_elem.set("id", msg.message_id)
            msg_elem.set("speaker", msg.speaker)
            msg_elem.set("timestamp", str(msg.timestamp))
            
            ET.SubElement(msg_elem, "content").text = msg.content
            
            # Coordinate streams
            if msg.coordinate_streams:
                streams = ET.SubElement(msg_elem, "coordinate_streams")
                for stream in msg.coordinate_streams:
                    stream_elem = ET.SubElement(streams, "stream")
                    stream_elem.set("gesture_type", stream.gesture_type)
                    stream_elem.set("coord_count", str(len(stream.parsed_coords)))
                    
                    ET.SubElement(stream_elem, "raw_ansi").text = stream.raw_ansi
                    ET.SubElement(stream_elem, "analysis").text = stream.analysis
                    
        # Discovery moments
        discoveries = ET.SubElement(root, "discovery_moments")
        for moment in self.session.discovery_moments:
            moment_elem = ET.SubElement(discoveries, "discovery")
            moment_elem.set("id", moment['moment_id'])
            ET.SubElement(moment_elem, "title").text = moment['title']
            ET.SubElement(moment_elem, "description").text = moment['description']
            
        # Write XML file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(filepath, encoding='utf-8', xml_declaration=True)
        
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation session"""
        duration = (self.session.end_time or time.time()) - self.session.start_time
        
        return {
            'session_id': self.session.session_id,
            'title': self.session.title,
            'duration_minutes': duration / 60,
            'total_messages': len(self.session.messages),
            'total_coordinate_streams': self.session.coordinate_stats['total_streams'],
            'total_coordinates': self.session.coordinate_stats['total_coordinates'],
            'gesture_types': self.session.coordinate_stats['gesture_types'],
            'discovery_moments': len(self.session.discovery_moments),
            'innovation_significance': 'Revolutionary coordinate streaming interface discovery'
        }

def demo_chat_forge():
    """Demonstrate the Chat Forge capabilities"""
    print("*** CHAT FORGE - Lightning-Fast Conversation Storage ***")
    
    # Create a sample session
    forge = ChatForge("SAURON Discovery Session")
    
    # Add some sample messages
    forge.add_message("human", "hi claude can you familiarize yourself with this folder you are in?")
    forge.add_message("claude", "I'll help you explore this directory and understand the project structure.")
    
    # Simulate a message with coordinates
    coordinate_msg = "bro can you just push this shit and get me out of here<35;68;18m<35;67;18m<35;65;18m"
    forge.parse_message_with_coordinates("human", coordinate_msg)
    
    forge.add_message("claude", "I can see you drew a perfect rectangle slash! This is incredible coordinate streaming!")
    
    # Add discovery moments
    forge.add_discovery_moment(
        "First Coordinate Stream Detection",
        "Discovered ANSI escape sequences leaking from stuck TUI, revealing real-time mouse tracking",
        ["msg_0003"]
    )
    
    forge.add_discovery_moment(
        "SAURON Controller Concept Birth", 
        "Realized coordinate streams could become ultra-fast gesture controller interface",
        ["msg_0004"]
    )
    
    # Show summary
    summary = forge.get_session_summary()
    print(f"\nSession Summary:")
    print(f"  Messages: {summary['total_messages']}")
    print(f"  Coordinate Streams: {summary['total_coordinate_streams']}")
    print(f"  Total Coordinates: {summary['total_coordinates']}")
    print(f"  Discovery Moments: {summary['discovery_moments']}")
    
    # Export formats
    forge.export_to_json("sauron_chat_session.json")
    forge.export_to_xml("sauron_chat_session.xml")
    
    print(f"\n*** Chat session exported to JSON and XML formats ***")
    print(f"*** Ready for historic preservation! ***")

if __name__ == "__main__":
    demo_chat_forge()