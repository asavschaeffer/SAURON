"""
SAURON Combat Engine - Where Gestures Become Glory

Ultra-responsive terminal combat system driven by mouse gestures.
ASCII battlefield rendering with sub-millisecond input response.
"""

import time
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

from coordinate_parser import GestureType, Coordinate

class CombatAction(Enum):
    SLASH = "slash"
    STAB = "stab"
    BLOCK = "block"
    DODGE = "dodge"
    CAST_SHIELD = "cast_shield"
    ARROW_SHOT = "arrow_shot"

@dataclass
class CombatResult:
    """Result of a combat action"""
    action: CombatAction
    damage: int
    hit: bool
    critical: bool
    message: str
    
class Enemy:
    """Simple enemy with ASCII representation"""
    def __init__(self, name: str, hp: int, symbol: str):
        self.name = name
        self.max_hp = hp
        self.hp = hp
        self.symbol = symbol
        self.position = (10, 40)  # Row, Col
        
    def take_damage(self, damage: int) -> bool:
        """Apply damage, return True if enemy dies"""
        self.hp = max(0, self.hp - damage)
        return self.hp == 0
        
    def is_alive(self) -> bool:
        return self.hp > 0

class CombatEngine:
    """Main combat system"""
    
    # Gesture to action mapping
    GESTURE_ACTIONS = {
        GestureType.SLASH: CombatAction.SLASH,
        GestureType.STAB: CombatAction.STAB, 
        GestureType.CIRCLE: CombatAction.CAST_SHIELD,
        GestureType.RECTANGLE: CombatAction.BLOCK,
        GestureType.LINE: CombatAction.ARROW_SHOT
    }
    
    def __init__(self):
        self.player_hp = 100
        self.player_max_hp = 100
        self.shield_active = False
        self.shield_duration = 0
        self.enemies: List[Enemy] = []
        self.combat_log: List[str] = []
        
        # Spawn some orcs
        self._spawn_enemies()
        
    def _spawn_enemies(self):
        """Spawn enemies at the Black Gate"""
        orc_names = ["Uglúk", "Grishnákh", "Azog", "Bolg", "Gothmog"]
        for i, name in enumerate(orc_names[:3]):
            enemy = Enemy(name, random.randint(30, 60), "👹")
            enemy.position = (8 + i * 2, 35 + i * 10)
            self.enemies.append(enemy)
    
    def process_gesture(self, gesture_type: GestureType, coords: List[Coordinate]) -> CombatResult:
        """Convert gesture into combat action"""
        if gesture_type not in self.GESTURE_ACTIONS:
            return CombatResult(
                action=CombatAction.DODGE,
                damage=0,
                hit=False,
                critical=False,
                message="Unknown gesture - you stumble!"
            )
        
        action = self.GESTURE_ACTIONS[gesture_type]
        return self._execute_action(action, coords)
    
    def _execute_action(self, action: CombatAction, coords: List[Coordinate]) -> CombatResult:
        """Execute combat action with damage calculation"""
        
        if action == CombatAction.SLASH:
            # Damage based on gesture speed and length
            gesture_speed = self._calculate_gesture_speed(coords)
            base_damage = 15
            damage = int(base_damage * (1 + gesture_speed / 100))
            hit = random.random() > 0.2  # 80% hit chance
            critical = random.random() < 0.15  # 15% crit chance
            
            if critical:
                damage *= 2
                message = f"🗡️ CRITICAL SLASH! {damage} damage!"
            elif hit:
                message = f"⚔️ Slash hits for {damage} damage!"
            else:
                damage = 0
                message = "🌪️ Your slash misses!"
                
        elif action == CombatAction.STAB:
            damage = random.randint(20, 35)
            hit = random.random() > 0.3  # 70% hit chance
            critical = random.random() < 0.2  # 20% crit chance
            
            if critical:
                damage *= 2
                message = f"🎯 PERFECT STAB! {damage} damage!"
            elif hit:
                message = f"🗡️ Stab pierces for {damage} damage!"
            else:
                damage = 0
                message = "🌪️ Your stab is deflected!"
                
        elif action == CombatAction.CAST_SHIELD:
            self.shield_active = True
            self.shield_duration = 5
            damage = 0
            hit = True
            critical = False
            message = "🛡️ Magical shield activated!"
            
        elif action == CombatAction.BLOCK:
            damage = 0
            hit = True
            critical = False
            message = "🛡️ Defensive stance - incoming damage reduced!"
            
        elif action == CombatAction.ARROW_SHOT:
            damage = random.randint(12, 25)
            hit = random.random() > 0.15  # 85% hit chance
            critical = random.random() < 0.1  # 10% crit chance
            
            if critical:
                damage *= 2
                message = f"🏹 BULLSEYE! Arrow deals {damage} damage!"
            elif hit:
                message = f"🏹 Arrow hits for {damage} damage!"
            else:
                damage = 0
                message = "🏹 Arrow flies wide!"
        
        else:
            damage = 0
            hit = False
            critical = False
            message = "❓ You hesitate..."
        
        # Apply damage to nearest enemy
        if hit and damage > 0 and self.enemies:
            target = self.enemies[0]  # Simple targeting
            killed = target.take_damage(damage)
            if killed:
                message += f" {target.name} falls!"
                self.enemies.remove(target)
                
        self.combat_log.append(message)
        if len(self.combat_log) > 10:
            self.combat_log = self.combat_log[-10:]
            
        return CombatResult(action, damage, hit, critical, message)
    
    def _calculate_gesture_speed(self, coords: List[Coordinate]) -> float:
        """Calculate gesture speed for damage multiplier"""
        if len(coords) < 2:
            return 0
            
        total_distance = 0
        time_span = coords[-1].timestamp - coords[0].timestamp
        
        for i in range(1, len(coords)):
            dr = coords[i].row - coords[i-1].row
            dc = coords[i].col - coords[i-1].col
            distance = (dr*dr + dc*dc)**0.5
            total_distance += distance
            
        if time_span > 0:
            return total_distance / time_span  # Distance per second
        return 0
    
    def get_battlefield_display(self) -> List[str]:
        """Render ASCII battlefield"""
        lines = []
        
        # Header
        lines.append("═" * 80)
        lines.append("⚔️  BATTLE AT THE BLACK GATE  ⚔️")
        lines.append("═" * 80)
        
        # Player status
        hp_bar = "█" * (self.player_hp // 5) + "░" * ((self.player_max_hp - self.player_hp) // 5)
        shield_status = "🛡️ " if self.shield_active else "   "
        lines.append(f"🧙 YOU: {shield_status}[{hp_bar}] {self.player_hp}/{self.player_max_hp} HP")
        lines.append("")
        
        # Battlefield grid (simplified)
        battlefield = [[" " for _ in range(80)] for _ in range(15)]
        
        # Place player
        battlefield[12][10] = "🧙"
        
        # Place enemies
        for enemy in self.enemies:
            row, col = enemy.position
            if 0 <= row < 15 and 0 <= col < 80:
                hp_percent = enemy.hp / enemy.max_hp
                if hp_percent > 0.7:
                    battlefield[row][col] = "👹"
                elif hp_percent > 0.3:
                    battlefield[row][col] = "😵"
                else:
                    battlefield[row][col] = "💀"
        
        # Convert battlefield to strings
        for row in battlefield:
            lines.append("".join(row))
            
        lines.append("")
        
        # Combat log
        lines.append("📜 BATTLE LOG:")
        for log_entry in self.combat_log[-5:]:
            lines.append(f"  {log_entry}")
            
        return lines
    
    def is_battle_won(self) -> bool:
        """Check if all enemies are defeated"""
        return len(self.enemies) == 0
    
    def is_battle_lost(self) -> bool:
        """Check if player is defeated"""
        return self.player_hp <= 0

# Demo combat scenario
def demo_combat():
    """Demonstrate combat engine"""
    engine = CombatEngine()
    
    print("🧿 SAURON Combat Engine Demo 🧿")
    print("\nGesture mapping:")
    print("  Circle → Shield")
    print("  Slash → Sword attack") 
    print("  Stab → Precise strike")
    print("  Rectangle → Block")
    print("  Line → Arrow shot")
    print("\n" + "="*50)
    
    # Simulate some combat
    from coordinate_parser import GestureType
    
    test_gestures = [
        GestureType.SLASH,
        GestureType.CIRCLE, 
        GestureType.STAB,
        GestureType.SLASH
    ]
    
    for gesture in test_gestures:
        result = engine.process_gesture(gesture, [])
        print(f"\n{gesture.value.upper()}: {result.message}")
        
        # Show battlefield
        battlefield = engine.get_battlefield_display()
        for line in battlefield[-10:]:  # Show last 10 lines
            print(line)
            
        if engine.is_battle_won():
            print("\n🎉 VICTORY! The Black Gate falls!")
            break
            
        time.sleep(1)

if __name__ == "__main__":
    demo_combat()