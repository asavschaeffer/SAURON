"""
SAURON - Supreme Authority Ultra-Responsive Orchestrated Navigator

The ultimate terminal-based gesture controller that turns mouse movements
into lightning-fast combat actions. Fight at the Black Gate with ASCII art!
"""

import sys
import time
import threading
from typing import Optional

from coordinate_parser import CoordinateParser, GestureClassifier, GestureType
from combat_engine import CombatEngine

class SauronController:
    """Main controller that orchestrates the Eye of Sauron"""
    
    def __init__(self):
        self.parser = CoordinateParser()
        self.classifier = GestureClassifier()
        self.combat_engine = CombatEngine()
        self.running = False
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.5  # Minimum time between gestures
        
    def start_watching(self):
        """Start watching for coordinate input from stdin"""
        self.running = True
        
        print("🧿 THE EYE OF SAURON AWAKENS 🧿")
        print("=" * 60)
        print("Move your mouse to fight at the Black Gate!")
        print("Press Ctrl+C to quit")
        print("=" * 60)
        
        # Show initial battlefield
        self._display_battlefield()
        
        try:
            # Start input monitoring thread
            input_thread = threading.Thread(target=self._monitor_input, daemon=True)
            input_thread.start()
            
            # Main game loop
            while self.running:
                time.sleep(0.1)  # Small delay to prevent CPU spinning
                
                # Check for recent gestures
                self._process_recent_gestures()
                
                # Update shield duration
                if self.combat_engine.shield_active:
                    self.combat_engine.shield_duration -= 0.1
                    if self.combat_engine.shield_duration <= 0:
                        self.combat_engine.shield_active = False
                        print("\n🛡️ Shield fades away...")
                
                # Check win/lose conditions
                if self.combat_engine.is_battle_won():
                    print("\n🎉 VICTORY! You have defended Middle-earth!")
                    break
                elif self.combat_engine.is_battle_lost():
                    print("\n💀 DEFEAT! The darkness consumes you...")
                    break
                    
        except KeyboardInterrupt:
            print("\n\n👋 The Eye closes... Farewell, Ranger!")
            self.running = False
    
    def _monitor_input(self):
        """Monitor stdin for coordinate sequences"""
        buffer = ""
        
        while self.running:
            try:
                # Read from stdin with timeout
                import select
                
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    if char:
                        buffer += char
                        
                        # Look for complete coordinate sequences
                        if 'm' in buffer or 'M' in buffer:
                            coords = self.parser.parse_stream(buffer)
                            if coords:
                                self._handle_coordinates(coords)
                            buffer = ""  # Clear buffer after processing
                            
            except Exception as e:
                # Silently continue on input errors
                continue
    
    def _handle_coordinates(self, coords):
        """Process new coordinates"""
        current_time = time.time()
        
        # Respect gesture cooldown
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return
            
        # Get recent gesture data
        recent_coords = self.parser.get_recent_gesture(time_window=1.0)
        
        if len(recent_coords) >= 3:
            # Classify the gesture
            gesture_type = self.classifier.classify_gesture(recent_coords)
            
            if gesture_type != GestureType.UNKNOWN:
                # Execute combat action
                result = self.combat_engine.process_gesture(gesture_type, recent_coords)
                
                # Show immediate feedback
                print(f"\n⚡ {result.message}")
                
                # Update display
                self._display_battlefield()
                
                self.last_gesture_time = current_time
                
                # Clear old coordinates to prevent re-processing
                self.parser.clear_buffer()
    
    def _process_recent_gestures(self):
        """Process any gestures that might have been missed"""
        # This could be expanded for more sophisticated gesture handling
        pass
    
    def _display_battlefield(self):
        """Display the current battlefield state"""
        # Clear screen (simple version)
        print("\n" * 2)
        
        battlefield_lines = self.combat_engine.get_battlefield_display()
        for line in battlefield_lines:
            print(line)
        
        print("\n🎯 Draw gestures with your mouse:")
        print("   Circle=Shield | Slash=Sword | Stab=Pierce | Rectangle=Block | Line=Arrow")
        print("-" * 80)

def main():
    """Entry point for SAURON controller"""
    
    # ASCII art banner
    banner = """
    ███████╗ █████╗ ██╗   ██╗██████╗  ██████╗ ███╗   ██╗
    ██╔════╝██╔══██╗██║   ██║██╔══██╗██╔═══██╗████╗  ██║
    ███████╗███████║██║   ██║██████╔╝██║   ██║██╔██╗ ██║
    ╚════██║██╔══██║██║   ██║██╔══██╗██║   ██║██║╚██╗██║
    ███████║██║  ██║╚██████╔╝██║  ██║╚██████╔╝██║ ╚████║
    ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
    
    Supreme Authority Ultra-Responsive Orchestrated Navigator
    """
    
    print(banner)
    print("The most responsive terminal controller ever built!")
    print("Sub-millisecond gesture recognition through ANSI coordinates")
    print("\nPreparing for battle...\n")
    
    # Initialize and start the controller
    controller = SauronController()
    controller.start_watching()

if __name__ == "__main__":
    main()