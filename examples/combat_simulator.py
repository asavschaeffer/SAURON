#!/usr/bin/env python3
"""
SAURON Combat Simulator

Interactive combat demonstration where gestures control battle actions.
Fight orcs using mouse movements in your terminal!

Usage:
    python combat_simulator.py
    
Controls:
- Circle gestures: Cast shield spells
- Slash gestures: Sword attacks  
- Stab gestures: Precision strikes
- Rectangle gestures: Defensive blocks
"""

import sys
import os
import time
import random
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coordinate_parser import HyperDetector, GestureType
from combat_engine import CombatEngine, CombatAction, CombatResult
from hyper_detection import GestureMetrics, GestureSpeed

class CombatSimulator:
    """Interactive combat simulator using gesture controls"""
    
    def __init__(self):
        self.detector = HyperDetector()
        self.combat = CombatEngine()
        self.running = False
        self.last_enemy_action = time.time()
        self.enemy_action_interval = 3.0  # Enemy acts every 3 seconds
        
        # Register gesture callback
        self.detector.add_gesture_callback(self.handle_player_gesture)
    
    def print_banner(self):
        """Display the combat simulator banner"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║                  ⚔️ SAURON COMBAT SIMULATOR ⚔️               ║
║                                                              ║
║  Use mouse gestures to battle orcs at the Black Gate!       ║
║                                                              ║
║  🔵 CIRCLE    → 🛡️ Cast Shield (blocks incoming damage)     ║
║  ⚔️ SLASH     → 🗡️ Sword Attack (15-30 damage)            ║
║  🎯 STAB      → 🏹 Precision Strike (20-35 damage)         ║
║  ⬛ RECTANGLE → 🛡️ Defensive Block (reduces damage)         ║
║                                                              ║
║  The faster your gesture, the more powerful the action!     ║
║                                                              ║
║  Press Ctrl+C to retreat from battle                        ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def handle_player_gesture(self, gesture_type: str, metrics: GestureMetrics):
        """Process player gesture into combat action"""
        try:
            # Convert string to enum
            gesture_enum = GestureType(gesture_type)
            
            # Execute combat action
            result = self.combat.process_gesture(gesture_enum, [])
            
            # Display result with enhanced feedback
            self.display_combat_result(gesture_type, metrics, result)
            
            # Update battlefield display
            self.display_battlefield()
            
            # Check for game over conditions
            if self.combat.player_health <= 0:
                self.handle_defeat()
            elif not self.combat.enemies:
                self.handle_victory()
                
        except ValueError:
            # Unknown gesture type
            print(f"❓ Unknown gesture: {gesture_type}")
    
    def display_combat_result(self, gesture_type: str, metrics: GestureMetrics, result: CombatResult):
        """Display detailed combat action results"""
        
        # Speed-based action descriptions
        speed_descriptions = {
            GestureSpeed.GLACIAL: "carefully executed",
            GestureSpeed.SLOW: "deliberately performed", 
            GestureSpeed.NORMAL: "swiftly executed",
            GestureSpeed.FAST: "rapidly performed",
            GestureSpeed.LIGHTNING: "lightning-fast",
            GestureSpeed.LUDICROUS: "impossibly fast"
        }
        
        speed_desc = speed_descriptions.get(metrics.speed_category, "executed")
        
        print(f"\n⚡ COMBAT ACTION: {speed_desc} {gesture_type}")
        print(f"   📊 Speed: {metrics.velocity:.0f} coords/sec ({metrics.speed_category.value})")
        print(f"   💥 {result.message}")
        
        if result.damage_dealt > 0:
            print(f"   🗡️ Damage dealt: {result.damage_dealt}")
        
        if result.shield_activated:
            print(f"   🛡️ Shield duration: {result.shield_duration} turns")
        
        print("   " + "-"*50)
    
    def display_battlefield(self):
        """Show current battlefield status"""
        battlefield = self.combat.get_battlefield_display()
        
        # Clear screen and show battlefield
        print("\n" * 2)
        print("🏰 BATTLEFIELD STATUS:")
        for line in battlefield:
            print(f"   {line}")
        print()
    
    def simulate_enemy_actions(self):
        """Simulate enemy combat actions"""
        current_time = time.time()
        
        if current_time - self.last_enemy_action >= self.enemy_action_interval:
            if self.combat.enemies:
                # Random enemy attack
                enemy = random.choice(self.combat.enemies)
                damage = random.randint(8, 15)
                
                if self.combat.shield_duration > 0:
                    print(f"👹 {enemy['name']} attacks but hits your shield!")
                    self.combat.shield_duration -= 1
                else:
                    self.combat.player_health -= damage
                    print(f"👹 {enemy['name']} attacks for {damage} damage!")
                    print(f"💚 Your health: {self.combat.player_health}/100")
                
                self.last_enemy_action = current_time
    
    def handle_victory(self):
        """Handle player victory"""
        print("""
🏆 VICTORY! 🏆

You have defeated all orcs at the Black Gate!
The Ring of Power has been mastered through gesture control!

Your mouse movements have proven mightier than the sword!
        """)
        self.running = False
    
    def handle_defeat(self):
        """Handle player defeat"""
        print("""
💀 DEFEAT! 💀

The orcs have overwhelmed you at the Black Gate...
But the SAURON system recorded every gesture for analysis!

Practice your combat gestures and return stronger!
        """)
        self.running = False
    
    def run(self):
        """Main combat simulation loop"""
        self.print_banner()
        
        print("🎯 Preparing for battle...")
        print("👹 Orcs are approaching the Black Gate!")
        print("⚔️ Draw gestures in the terminal to fight!")
        print()
        
        self.display_battlefield()
        self.running = True
        
        try:
            while self.running and self.combat.player_health > 0 and self.combat.enemies:
                # Simulate enemy actions periodically
                self.simulate_enemy_actions()
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n🏃 You have retreated from the battlefield!")
            print("   The orcs celebrate... for now.")
        
        # Show final combat statistics
        self.show_final_stats()
    
    def show_final_stats(self):
        """Display final combat statistics"""
        stats = self.detector.get_performance_stats()
        
        print(f"\n📊 COMBAT STATISTICS:")
        print(f"   ⚔️ Final health: {self.combat.player_health}/100")
        print(f"   👹 Enemies remaining: {len(self.combat.enemies)}")
        print(f"   ⚡ Gesture response time: {stats.get('avg_processing_time_ms', 0):.3f}ms")
        print(f"   🚀 Peak gesture speed: {stats.get('coordinates_per_second', 0):,.0f} coords/sec")
        print(f"   ⏱️ Battle duration: {stats.get('runtime_seconds', 0):.1f} seconds")
        print("\n👁️ SAURON has recorded your combat performance for analysis...")

def main():
    """Launch the combat simulator"""
    simulator = CombatSimulator()
    simulator.run()

if __name__ == "__main__":
    main()