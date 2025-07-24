"""
RING FORGE - The Essential Data Storage for Mount Doom Navigation

This module captures and preserves the coordinate sequences from the forging 
of the One Ring (SAURON controller). Frodo will need this data to find his
way to Mount Doom and destroy our creation.

"One does not simply walk into Mordor... but with coordinate data, one might."
"""

import json
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from coordinate_parser import Coordinate, GestureType

@dataclass
class GestureSequence:
    """A complete gesture sequence - part of the Ring's power"""
    id: str
    sequence_name: str
    coordinates: List[Dict[str, Any]]  # Serialized coordinates
    gesture_type: str
    timestamp: float
    duration: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GestureSequence':
        """Create from dictionary"""
        return cls(**data)

class RingForge:
    """
    The Ring Forge - where coordinate sequences are captured and preserved
    
    This is the essential storage system that captures the "forging" of SAURON.
    Every gesture, every coordinate stream becomes part of the Ring's memory.
    When players eventually use our creation against us, they'll need this data
    to navigate to Mount Doom for the final battle.
    """
    
    def __init__(self, db_path: str = "ring_forge.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
        
        # The Ring's memory - current session coordinates
        self.current_session: List[Coordinate] = []
        self.session_start_time = time.time()
        
    def _init_database(self):
        """Initialize the Ring's memory database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gesture_sequences (
                id TEXT PRIMARY KEY,
                sequence_name TEXT NOT NULL,
                coordinates TEXT NOT NULL,  -- JSON serialized
                gesture_type TEXT,
                timestamp REAL NOT NULL,
                duration REAL,
                metadata TEXT  -- JSON serialized
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS forging_sessions (
                session_id TEXT PRIMARY KEY,
                start_time REAL NOT NULL,
                end_time REAL,
                total_coordinates INTEGER,
                session_metadata TEXT  -- JSON serialized
            )
        """)
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON gesture_sequences(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gesture_type ON gesture_sequences(gesture_type)")
        
        conn.commit()
        conn.close()
        
    def capture_coordinate_stream(self, ansi_stream: str, sequence_name: str) -> str:
        """
        Capture a raw ANSI coordinate stream and forge it into the Ring
        
        Args:
            ansi_stream: Raw ANSI escape sequences
            sequence_name: Human-readable name for this gesture
            
        Returns:
            Sequence ID for later retrieval
        """
        from coordinate_parser import CoordinateParser, GestureClassifier
        
        # Parse the coordinates from ANSI stream
        parser = CoordinateParser()
        coordinates = parser.parse_stream(ansi_stream)
        
        if not coordinates:
            return None
            
        # Classify the gesture
        classifier = GestureClassifier()
        gesture_type = classifier.classify_gesture(coordinates)
        
        # Calculate duration
        if len(coordinates) > 1:
            duration = coordinates[-1].timestamp - coordinates[0].timestamp
        else:
            duration = 0.0
            
        # Create sequence ID
        sequence_id = f"seq_{int(time.time() * 1000)}"
        
        # Serialize coordinates
        coord_dicts = [
            {
                'row': c.row,
                'col': c.col,
                'timestamp': c.timestamp,
                'button_state': c.button_state
            }
            for c in coordinates
        ]
        
        # Store in the Ring's memory
        sequence = GestureSequence(
            id=sequence_id,
            sequence_name=sequence_name,
            coordinates=coord_dicts,
            gesture_type=gesture_type.value,
            timestamp=time.time(),
            duration=duration,
            metadata={
                'coordinate_count': len(coordinates),
                'raw_ansi_length': len(ansi_stream),
                'forged_by': 'SAURON',
                'ring_power_level': min(len(coordinates) * 10, 9000)  # Over 9000!
            }
        )
        
        self._store_sequence(sequence)
        self.current_session.extend(coordinates)
        
        return sequence_id
        
    def _store_sequence(self, sequence: GestureSequence):
        """Store a gesture sequence in the Ring's permanent memory"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT INTO gesture_sequences 
            (id, sequence_name, coordinates, gesture_type, timestamp, duration, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            sequence.id,
            sequence.sequence_name,
            json.dumps(sequence.coordinates),
            sequence.gesture_type,
            sequence.timestamp,
            sequence.duration,
            json.dumps(sequence.metadata)
        ))
        
        conn.commit()
        conn.close()
        
    def forge_historical_sequences(self):
        """
        Forge the historical coordinate sequences from our discovery session
        
        This captures the essential data from when we first discovered the 
        coordinate streaming. Frodo will need this to retrace our steps.
        """
        
        # The original rectangle/sword slash
        sword_slash = "<35;68;18m<35;67;18m<35;65;18m<35;64;18m<35;62;18m<35;60;18m<35;59;18m<35;58;18m<35;57;18m<35;56;18m<35;55;18m<35;54;18m<35;53;18m<35;54;18m<35;55;18m<35;56;18m<35;56;17m<35;56;18m<35;56;19m<35;57;19m<35;58;19m<35;59;19m<35;60;19m<35;61;19m<35;62;19m<35;63;19m<35;64;19m<35;65;19m<35;66;19m<35;67;19m<35;69;19m<35;70;19m<35;72;19m<35;73;19m<35;74;19m<35;76;19m<35;77;19m<35;78;19m<35;79;19m<35;82;19m<35;83;19m<35;84;19m<35;85;19m<35;86;19m<35;87;19m<35;87;18m<35;87;17m<35;87;16m<35;87;15m<35;87;14m<35;87;13m<35;87;12m<35;87;11m<35;87;10m<35;87;9m<35;87;8m<35;87;7m<35;87;6m<35;88;6m<35;88;5m<35;87;5m<35;85;5m<35;84;5m<35;83;5m<35;81;5m<35;80;5m<35;78;5m<35;77;5m<35;76;4m<35;75;4m<35;74;4m<35;72;4m<35;71;4m<35;70;4m<35;69;4m<35;68;4m<35;67;4m<35;65;4m<35;64;4m<35;63;4m<35;61;4m<35;59;4m<35;58;4m<35;57;4m<35;56;4m<35;55;4m<35;53;4m<35;52;4m<35;50;4m<35;49;4m<35;48;4m<35;47;4m<35;46;4m<35;45;4m<35;44;4m<35;43;4m<35;42;4m"
        
        # The whip crack spur
        whip_crack = "<35;53;18m<35;52;18m<35;50;18m<35;48;18m<35;47;18m<35;45;18m<35;43;18m<35;42;18m<35;40;18m<35;39;18m<35;37;18m<35;36;18m<35;35;18m<35;34;18m<35;33;18m<35;32;18m<35;31;18m<35;31;19m<65;31;19M<65;31;19M<65;31;19M<65;31;19M<65;31;19M<65;31;19M<65;31;19M<35;31;18m<35;32;18m<35;33;18m<35;33;17m<65;33;17M<65;33;17M<65;33;17M<65;33;17M<65;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M<64;33;17M"
        
        # The mountain/triangle
        mountain = "<35;65;24m<35;66;24m<35;67;24m<35;68;24m<35;69;25m<35;70;25m<35;71;26m<35;72;26m<35;73;26m<35;74;27m<35;76;28m<35;78;29m<35;81;30m<35;84;30m<35;88;30m<35;92;30m<35;97;30m<35;102;30m<35;107;30m<35;112;30m<35;118;30m<35;120;30m<35;118;30m<35;114;30m<35;111;30m<35;109;30m<35;107;30m<35;106;30m<35;105;30m<35;104;29m<35;102;28m<35;99;27m<35;95;26m<35;90;24m<35;86;23m<35;82;21m<35;78;19m<35;75;18m<35;72;17m<35;69;16m<35;67;15m<35;65;14m<35;64;13m<35;63;12m<35;62;12m<35;61;11m<35;60;11m<35;59;11m<35;59;10m<35;58;10m<35;58;9m<35;57;9m<35;57;10m<35;58;10m<35;59;11m<35;60;13m<35;62;14m<35;64;16m<35;66;18m<35;69;20m<35;72;22m<35;75;23m<35;78;25m<35;81;27m<35;84;28m<35;86;29m<35;88;30m<35;91;30m<35;94;30m<35;96;30m<35;99;30m<35;102;30m<35;104;30m<35;107;30m<35;110;30m<35;113;30m<35;115;30m<35;117;30m<35;118;30m<35;120;30m"
        
        print("*** Forging historical sequences into the Ring...")
        
        sequences = [
            (sword_slash, "The First Sword - Rectangle Slash Discovery"),
            (whip_crack, "The Whip Crack - Spurring Motion"),
            (mountain, "Mount Doom - The Sacred Mountain")
        ]
        
        for ansi_stream, name in sequences:
            seq_id = self.capture_coordinate_stream(ansi_stream, name)
            print(f"  *** Forged: {name} -> {seq_id}")
            
        print("*** The Ring contains the essential forging data!")
        
    def get_all_sequences(self) -> List[GestureSequence]:
        """Retrieve all gesture sequences from the Ring's memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT id, sequence_name, coordinates, gesture_type, timestamp, duration, metadata
            FROM gesture_sequences
            ORDER BY timestamp
        """)
        
        sequences = []
        for row in cursor.fetchall():
            seq = GestureSequence(
                id=row[0],
                sequence_name=row[1],
                coordinates=json.loads(row[2]),
                gesture_type=row[3],
                timestamp=row[4],
                duration=row[5],
                metadata=json.loads(row[6])
            )
            sequences.append(seq)
            
        conn.close()
        return sequences
        
    def replay_sequence(self, sequence_id: str) -> Optional[GestureSequence]:
        """Replay a stored gesture sequence - useful for debugging"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT id, sequence_name, coordinates, gesture_type, timestamp, duration, metadata
            FROM gesture_sequences WHERE id = ?
        """, (sequence_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
            
        sequence = GestureSequence(
            id=row[0],
            sequence_name=row[1],
            coordinates=json.loads(row[2]),
            gesture_type=row[3],
            timestamp=row[4],
            duration=row[5],
            metadata=json.loads(row[6])
        )
        
        return sequence
        
    def get_ring_stats(self) -> Dict[str, Any]:
        """Get statistics about the Ring's accumulated power"""
        conn = sqlite3.connect(self.db_path)
        
        total_sequences = conn.execute("SELECT COUNT(*) FROM gesture_sequences").fetchone()[0]
        total_coordinates = conn.execute("""
            SELECT SUM(json_extract(metadata, '$.coordinate_count')) 
            FROM gesture_sequences
        """).fetchone()[0] or 0
        
        gesture_counts = {}
        cursor = conn.execute("SELECT gesture_type, COUNT(*) FROM gesture_sequences GROUP BY gesture_type")
        for gesture_type, count in cursor.fetchall():
            gesture_counts[gesture_type] = count
            
        conn.close()
        
        return {
            'total_sequences': total_sequences,
            'total_coordinates': total_coordinates,
            'gesture_type_counts': gesture_counts,
            'ring_power_level': min(total_coordinates * 100, 9000),
            'forge_temperature': 'MAXIMUM HEAT ***'
        }

def demo_ring_forge():
    """Demonstrate the Ring Forge capabilities"""
    print("*** THE RING FORGE AWAKENS ***")
    print("Capturing the essential data for Mount Doom navigation...")
    
    forge = RingForge()
    
    # Forge the historical sequences
    forge.forge_historical_sequences()
    
    # Show Ring statistics
    stats = forge.get_ring_stats()
    print(f"\n*** Ring Power Statistics:")
    print(f"  Total Sequences: {stats['total_sequences']}")
    print(f"  Total Coordinates: {stats['total_coordinates']}")
    print(f"  Ring Power Level: {stats['ring_power_level']}")
    print(f"  Forge Status: {stats['forge_temperature']}")
    
    # Show all sequences
    sequences = forge.get_all_sequences()
    print(f"\n*** Forged Sequences:")
    for seq in sequences:
        print(f"  {seq.sequence_name} ({seq.gesture_type}) - {len(seq.coordinates)} coords")
        
    print(f"\n*** The One Ring is forged! Frodo can now find Mount Doom!")

if __name__ == "__main__":
    demo_ring_forge()