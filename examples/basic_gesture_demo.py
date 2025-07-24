#!/usr/bin/env python3
"""
SAURON Basic Gesture Demonstration

A simple example showing how to detect and classify mouse gestures
in real-time using the SAURON framework.

Usage:
    python basic_gesture_demo.py
    
Then move your mouse in the terminal to draw shapes:
- Circle: Returns to starting point
- Slash: Long linear motion
- Stab: Short linear motion  
- Rectangle: Multiple direction changes
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coordinate_parser import HyperDetector, GestureClassifier, GestureType
import time

def print_banner():
    """Display the SAURON demo banner"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    🔥 SAURON GESTURE DEMO 🔥                 ║
║                                                              ║
║  Move your mouse in the terminal to draw shapes:             ║
║  🔵 CIRCLE    - Draw a curved path that returns to start     ║
║  ⚔️ SLASH     - Draw a long straight line                   ║
║  🎯 STAB      - Draw a short quick motion                    ║
║  ⬛ RECTANGLE - Draw with multiple direction changes         ║
║                                                              ║
║  Press Ctrl+C to exit                                        ║
╚══════════════════════════════════════════════════════════════╝
    """)

def gesture_callback(gesture_type: str, metrics):
    """Handle detected gestures with visual feedback"""
    
    # Gesture emoji mapping
    gesture_icons = {
        'circle': '🔵',
        'slash': '⚔️',
        'stab': '🎯', 
        'rectangle': '⬛',
        'line': '📏',
        'unknown': '❓'
    }
    
    # Speed classification icons
    speed_icons = {
        'glacial': '🐌',
        'slow': '🚶',
        'normal': '🏃',
        'fast': '⚡',
        'lightning': '🌩️',
        'ludicrous': '🚀'
    }
    
    icon = gesture_icons.get(gesture_type, '❓')
    speed_icon = speed_icons.get(metrics.speed_category.value, '❓')
    
    print(f"\n{icon} GESTURE DETECTED: {gesture_type.upper()}")
    print(f"   {speed_icon} Speed: {metrics.speed_category.value} ({metrics.velocity:.0f} coords/sec)")
    print(f"   🎯 Intensity: {metrics.intensity:.1f}")
    print(f"   📐 Curvature: {metrics.curvature:.3f}")
    print(f"   ✨ Smoothness: {metrics.smoothness:.3f}")
    print("   " + "="*50)

def main():
    """Main demonstration loop"""
    print_banner()
    
    # Initialize the detector
    detector = HyperDetector()
    
    # Register our callback
    detector.add_gesture_callback(gesture_callback)
    
    print("🎯 SAURON is now watching for gestures...")
    print("📊 Performance stats will be displayed every 10 seconds")
    print()
    
    last_stats_time = time.time()
    
    try:
        while True:
            # In a real application, you'd process incoming coordinate data here
            # For this demo, we simulate by checking for raw input
            
            # Display performance stats periodically
            current_time = time.time()
            if current_time - last_stats_time >= 10:
                stats = detector.get_performance_stats()
                print(f"\n📊 PERFORMANCE UPDATE:")
                print(f"   ⚡ Average processing time: {stats.get('avg_processing_time_ms', 0):.3f}ms")
                print(f"   🚀 Coordinates per second: {stats.get('coordinates_per_second', 0):,.0f}")
                print(f"   ⏱️ Runtime: {stats.get('runtime_seconds', 0):.1f} seconds")
                print(f"   💾 Buffer utilization: {stats.get('buffer_utilization', 0)*100:.1f}%")
                print()
                last_stats_time = current_time
            
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            
    except KeyboardInterrupt:
        print("\n\n🎯 SAURON Demo Complete!")
        print("   Thanks for experiencing the power of gesture recognition!")
        
        # Show final stats
        final_stats = detector.get_performance_stats()
        print(f"\n📊 FINAL STATISTICS:")
        print(f"   ⚡ Total runtime: {final_stats.get('runtime_seconds', 0):.1f} seconds")
        print(f"   🚀 Peak performance: {final_stats.get('coordinates_per_second', 0):,.0f} coords/sec")
        print(f"   💪 Average response: {final_stats.get('avg_processing_time_ms', 0):.3f}ms")
        print("\n👁️ The Eye of SAURON has been closed. Until next time...")

if __name__ == "__main__":
    main()