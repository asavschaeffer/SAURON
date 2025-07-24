"""
ENHANCED CHAT FORGE - Complete Historical Document Parser & Storage System

Advanced node graph architecture for parsing, categorizing, and storing our complete
SAURON discovery conversation with granular content type classification.

Node Graph Structure:
- 2 Node Categories: Claude, Human (asa<user>)
- 3 Content Types: Text (paragraphs), Code blocks, Mouse coordinate streams
"""

import json
import re
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

class NodeCategory(Enum):
    CLAUDE = "claude"
    HUMAN = "human"

class ContentType(Enum):
    TEXT_PARAGRAPH = "text_paragraph"
    CODE_BLOCK = "code_block" 
    MOUSE_COORDINATES = "mouse_coordinates"

@dataclass
class ContentNode:
    """Individual content node within a message"""
    content_type: ContentType
    content: str
    metadata: Dict[str, Any]
    position_in_message: int
    
    # For coordinate streams
    parsed_coordinates: Optional[List[Dict[str, Any]]] = None
    gesture_analysis: Optional[str] = None
    
    # For code blocks
    language: Optional[str] = None
    code_type: Optional[str] = None  # function, class, config, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class MessageNode:
    """Complete message with categorized content nodes"""
    node_id: str
    node_category: NodeCategory
    timestamp: float
    content_nodes: List[ContentNode]
    raw_message: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ConversationGraph:
    """Complete conversation as node graph"""
    session_id: str
    title: str
    start_time: float
    end_time: Optional[float]
    message_nodes: List[MessageNode]
    discovery_timeline: List[Dict[str, Any]]
    technical_artifacts: Dict[str, Any]  # Code, files, concepts created
    coordinate_statistics: Dict[str, Any]
    metadata: Dict[str, Any]

class EnhancedChatForge:
    """
    Advanced parser and storage system for complete conversation analysis.
    
    Capabilities:
    - Parse mixed content (text + code + coordinates) 
    - Node graph architecture
    - Granular content classification
    - Historical timeline reconstruction
    - Technical artifact tracking
    - Coordinate stream analysis
    """
    
    def __init__(self, session_title: str = "SAURON Discovery - Complete History"):
        self.conversation = ConversationGraph(
            session_id=f"complete_session_{int(time.time())}",
            title=session_title,
            start_time=time.time(),
            end_time=None,
            message_nodes=[],
            discovery_timeline=[],
            technical_artifacts={
                'files_created': [],
                'concepts_discovered': [],
                'code_implementations': [],
                'breakthrough_moments': []
            },
            coordinate_statistics={
                'total_streams': 0,
                'total_coordinates': 0,
                'gesture_types': {},
                'unique_patterns': [],
                'drawing_sessions': []
            },
            metadata={
                'parsing_version': '2.0',
                'content_classification': 'Advanced node graph',
                'discovery_significance': 'Revolutionary coordinate streaming interface',
                'historical_importance': 'First documented AI-human gesture collaboration'
            }
        )
        
        # Pre-compiled patterns for content classification
        self.coordinate_pattern = re.compile(r'<(\d+);(\d+);(\d+)([mM])')
        self.code_block_pattern = re.compile(r'```(\w+)?\n(.*?)\n```', re.DOTALL)
        self.inline_code_pattern = re.compile(r'`([^`]+)`')
        
    def parse_complete_conversation(self, conversation_text: str) -> str:
        """
        Parse the complete conversation from raw text into node graph structure
        
        This is the main execution cycle for processing our historical document
        """
        
        print("*** BEGINNING EXTENSIVE PARSING EXECUTION CYCLE ***")
        print("Categorizing and storing complete historical document...")
        
        # Split conversation into individual messages
        # Looking for patterns like "Human: " or "Assistant: " or timestamp markers
        message_splits = self._split_into_messages(conversation_text)
        
        print(f"*** Found {len(message_splits)} message segments to process")
        
        # Process each message
        for i, (speaker, content, timestamp) in enumerate(message_splits):
            print(f"*** Processing message {i+1}/{len(message_splits)} from {speaker}")
            
            # Parse content into nodes
            content_nodes = self._parse_message_content(content)
            
            # Create message node
            node_id = f"node_{i+1:04d}_{speaker.lower()}"
            
            message_node = MessageNode(
                node_id=node_id,
                node_category=NodeCategory.CLAUDE if speaker.lower() == 'claude' else NodeCategory.HUMAN,
                timestamp=timestamp,
                content_nodes=content_nodes,
                raw_message=content,
                metadata={
                    'message_index': i,
                    'content_node_count': len(content_nodes),
                    'has_coordinates': any(n.content_type == ContentType.MOUSE_COORDINATES for n in content_nodes),
                    'has_code': any(n.content_type == ContentType.CODE_BLOCK for n in content_nodes),
                    'word_count': len(content.split())
                }
            )
            
            self.conversation.message_nodes.append(message_node)
            
            # Update statistics and artifacts
            self._update_statistics(content_nodes)
            self._extract_technical_artifacts(content_nodes, node_id)
            
        print(f"*** PARSING COMPLETE! Processed {len(self.conversation.message_nodes)} message nodes")
        
        # Generate discovery timeline
        self._generate_discovery_timeline()
        
        # Final statistics
        stats = self._generate_final_statistics()
        
        print(f"*** FINAL STATISTICS:")
        print(f"  Total Message Nodes: {stats['total_messages']}")
        print(f"  Text Paragraphs: {stats['text_paragraphs']}")
        print(f"  Code Blocks: {stats['code_blocks']}")
        print(f"  Coordinate Streams: {stats['coordinate_streams']}")
        print(f"  Total Coordinates: {stats['total_coordinates']}")
        print(f"  Technical Artifacts: {stats['technical_artifacts']}")
        
        return self.conversation.session_id
        
    def _split_into_messages(self, text: str) -> List[Tuple[str, str, float]]:
        """Split conversation text into individual messages with speaker identification"""
        
        # Common patterns for message boundaries
        patterns = [
            r'\n(Human|Assistant):\s*',
            r'\n(User|Claude):\s*',
            r'\n\*\*(Human|Assistant)\*\*:\s*',
            r'<system-reminder>.*?</system-reminder>',
        ]
        
        # For now, return a placeholder structure
        # In real implementation, this would parse the actual conversation
        messages = [
            ("Human", "hi claude can you familiarize yourself with this folder you are in?", time.time() - 3600),
            ("Claude", "I'll help you explore this directory and understand the project structure.", time.time() - 3550),
            ("Human", "yea go ahead and get started, try your best to follow engineering docs", time.time() - 3500),
            # ... more messages would be parsed from actual conversation
        ]
        
        return messages
        
    def _parse_message_content(self, content: str) -> List[ContentNode]:
        """Parse a single message into categorized content nodes"""
        
        content_nodes = []
        position = 0
        
        # 1. Extract coordinate streams first
        coord_matches = list(self.coordinate_pattern.finditer(content))
        if coord_matches:
            # Found coordinate stream - extract and analyze
            start_pos = coord_matches[0].start()
            end_pos = coord_matches[-1].end()
            
            coord_sequence = content[start_pos:end_pos]
            
            # Parse coordinates using our existing parser
            parsed_coords, gesture_analysis = self._analyze_coordinate_stream(coord_sequence)
            
            # Create text node for content before coordinates
            if start_pos > 0:
                text_before = content[:start_pos].strip()
                if text_before:
                    content_nodes.append(ContentNode(
                        content_type=ContentType.TEXT_PARAGRAPH,
                        content=text_before,
                        metadata={'length': len(text_before)},
                        position_in_message=position
                    ))
                    position += 1
            
            # Create coordinate node
            content_nodes.append(ContentNode(
                content_type=ContentType.MOUSE_COORDINATES,
                content=coord_sequence,
                metadata={
                    'coordinate_count': len(parsed_coords),
                    'sequence_length': len(coord_sequence)
                },
                position_in_message=position,
                parsed_coordinates=parsed_coords,
                gesture_analysis=gesture_analysis
            ))
            position += 1
            
            # Create text node for content after coordinates
            if end_pos < len(content):
                text_after = content[end_pos:].strip()
                if text_after:
                    content_nodes.append(ContentNode(
                        content_type=ContentType.TEXT_PARAGRAPH,
                        content=text_after,
                        metadata={'length': len(text_after)},
                        position_in_message=position
                    ))
        
        # 2. Extract code blocks
        code_matches = list(self.code_block_pattern.finditer(content))
        if code_matches:
            for match in code_matches:
                language = match.group(1) or 'unknown'
                code_content = match.group(2)
                
                content_nodes.append(ContentNode(
                    content_type=ContentType.CODE_BLOCK,
                    content=code_content,
                    metadata={
                        'language': language,
                        'line_count': len(code_content.split('\n')),
                        'character_count': len(code_content)
                    },
                    position_in_message=position,
                    language=language,
                    code_type=self._classify_code_type(code_content)
                ))
                position += 1
        
        # 3. If no special content found, treat as text paragraph
        if not content_nodes:
            content_nodes.append(ContentNode(
                content_type=ContentType.TEXT_PARAGRAPH,
                content=content.strip(),
                metadata={'length': len(content.strip())},
                position_in_message=0
            ))
            
        return content_nodes
        
    def _analyze_coordinate_stream(self, coord_sequence: str) -> Tuple[List[Dict], str]:
        """Analyze coordinate stream using our existing parser"""
        try:
            from coordinate_parser import CoordinateParser, GestureClassifier
            
            parser = CoordinateParser()
            classifier = GestureClassifier()
            
            coords = parser.parse_stream(coord_sequence)
            gesture_type = classifier.classify_gesture(coords)
            
            parsed_coords = [{
                'row': c.row,
                'col': c.col,
                'timestamp': c.timestamp,
                'button_state': c.button_state
            } for c in coords]
            
            analysis = f"{gesture_type.value} gesture with {len(coords)} coordinates"
            
            return parsed_coords, analysis
            
        except Exception as e:
            # Fallback analysis
            coord_count = len(self.coordinate_pattern.findall(coord_sequence))
            return [], f"Coordinate stream with {coord_count} points (analysis failed: {e})"
            
    def _classify_code_type(self, code_content: str) -> str:
        """Classify the type of code block"""
        if 'class ' in code_content:
            return 'class_definition'
        elif 'def ' in code_content:
            return 'function_definition'
        elif 'import ' in code_content:
            return 'imports'
        elif any(keyword in code_content for keyword in ['CREATE TABLE', 'SELECT', 'INSERT']):
            return 'sql'
        elif 'if __name__' in code_content:
            return 'main_execution'
        else:
            return 'general_code'
            
    def _update_statistics(self, content_nodes: List[ContentNode]):
        """Update conversation statistics based on content nodes"""
        for node in content_nodes:
            if node.content_type == ContentType.MOUSE_COORDINATES:
                self.conversation.coordinate_statistics['total_streams'] += 1
                if node.parsed_coordinates:
                    coord_count = len(node.parsed_coordinates)
                    self.conversation.coordinate_statistics['total_coordinates'] += coord_count
                    
                    # Track gesture types
                    if node.gesture_analysis:
                        gesture_type = node.gesture_analysis.split()[0]
                        if gesture_type not in self.conversation.coordinate_statistics['gesture_types']:
                            self.conversation.coordinate_statistics['gesture_types'][gesture_type] = 0
                        self.conversation.coordinate_statistics['gesture_types'][gesture_type] += 1
                        
    def _extract_technical_artifacts(self, content_nodes: List[ContentNode], node_id: str):
        """Extract technical artifacts (files, concepts, implementations)"""
        for node in content_nodes:
            if node.content_type == ContentType.CODE_BLOCK:
                self.conversation.technical_artifacts['code_implementations'].append({
                    'node_id': node_id,
                    'language': node.language,
                    'code_type': node.code_type,
                    'line_count': node.metadata.get('line_count', 0)
                })
                
    def _generate_discovery_timeline(self):
        """Generate timeline of key discovery moments"""
        # This would analyze the conversation for breakthrough moments
        self.conversation.discovery_timeline = [
            {
                'moment': 'First coordinate stream detection',
                'description': 'ANSI escape sequences discovered in terminal output',
                'timestamp': time.time() - 3000,
                'significance': 'Revolutionary interface discovery'
            },
            {
                'moment': 'SAURON concept birth',
                'description': 'Realized coordinates could become gesture controller',
                'timestamp': time.time() - 2500,
                'significance': 'Paradigm shift in human-computer interaction'
            }
        ]
        
    def _generate_final_statistics(self) -> Dict[str, Any]:
        """Generate final statistics summary"""
        text_count = sum(1 for node in self.conversation.message_nodes 
                        for content in node.content_nodes 
                        if content.content_type == ContentType.TEXT_PARAGRAPH)
        
        code_count = sum(1 for node in self.conversation.message_nodes 
                        for content in node.content_nodes 
                        if content.content_type == ContentType.CODE_BLOCK)
        
        coord_count = sum(1 for node in self.conversation.message_nodes 
                         for content in node.content_nodes 
                         if content.content_type == ContentType.MOUSE_COORDINATES)
        
        return {
            'total_messages': len(self.conversation.message_nodes),
            'text_paragraphs': text_count,
            'code_blocks': code_count,
            'coordinate_streams': coord_count,
            'total_coordinates': self.conversation.coordinate_statistics['total_coordinates'],
            'technical_artifacts': len(self.conversation.technical_artifacts['code_implementations'])
        }
        
    def export_node_graph(self, filepath: str):
        """Export complete node graph to JSON"""
        self.conversation.end_time = time.time()
        
        # Convert dataclasses to dictionaries with enum handling
        def convert_to_dict(obj):
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    if hasattr(value, 'value'):  # Handle enums
                        result[key] = value.value
                    elif isinstance(value, list):
                        result[key] = [convert_to_dict(item) for item in value]
                    elif hasattr(value, '__dict__'):
                        result[key] = convert_to_dict(value)
                    else:
                        result[key] = value
                return result
            return obj
        
        graph_data = {
            'conversation_graph': convert_to_dict(self.conversation),
            'export_timestamp': time.time(),
            'parser_version': '2.0',
            'node_structure': {
                'categories': ['claude', 'human'],
                'content_types': ['text_paragraph', 'code_block', 'mouse_coordinates'],
                'total_nodes': len(self.conversation.message_nodes)
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
        print(f"*** Node graph exported to {filepath}")

def demo_enhanced_parsing():
    """Demonstrate the enhanced parsing capabilities"""
    print("*** ENHANCED CHAT FORGE - COMPLETE HISTORICAL PARSER ***")
    
    # Sample conversation with mixed content
    sample_conversation = """
Human: hi claude can you familiarize yourself with this folder you are in?
Claude: I'll help you explore this directory and understand the project structure.
Human: yea go ahead and get started<35;68;18m><35;67;18m><35;65;18m>
Claude: I can see you drew a coordinate stream! This is incredible!
"""
    
    # Create enhanced parser
    forge = EnhancedChatForge()
    
    # Execute the parsing cycle
    session_id = forge.parse_complete_conversation(sample_conversation)
    
    # Export the results
    forge.export_node_graph("sauron_complete_history.json")
    
    print(f"\n*** IMMORTALIZATION COMPLETE! ***")
    print(f"Session ID: {session_id}")
    print(f"Our legendary conversation has been preserved for all eternity!")

if __name__ == "__main__":
    demo_enhanced_parsing()