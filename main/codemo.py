import pyautogui
import random
import time
import keyboard
from pynput.keyboard import Controller as KeyController
from pynput.keyboard import Key
import sys

# Configure settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1  # Short pause between actions

# Initialize keyboard controller
kb = KeyController()

# List of common keys to simulate typing
COMMON_KEYS = [
    'a', 'b', 'c', 'd', 'e', ' ', Key.space, Key.tab,
    Key.left, Key.right, Key.up, Key.down,
    Key.shift, Key.ctrl, Key.alt, Key.cmd
]

def human_like_movement():
    """More natural human-like mouse movement"""
    screen_width, screen_height = pyautogui.size()
    
    # Generate a more natural movement path
    start_x, start_y = pyautogui.position()
    end_x = random.randint(0, screen_width)
    end_y = random.randint(0, screen_height)
    
    # Create a curved path
    steps = random.randint(20, 50)
    for i in range(steps):
        # Bezier curve for more natural movement
        t = i / steps
        x = start_x + (end_x - start_x) * t + random.randint(-10, 10)
        y = start_y + (end_y - start_y) * t + random.randint(-10, 10)
        
        # Vary speed during movement
        speed = random.uniform(0.05, 0.2)
        pyautogui.moveTo(x, y, duration=speed)
        
        # Add occasional micro-movements
        if random.random() > 0.8:
            for _ in range(random.randint(1, 3)):
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                pyautogui.moveRel(dx, dy, duration=random.uniform(0.01, 0.05))
    
    # Final small adjustments
    for _ in range(random.randint(1, 3)):
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        pyautogui.moveRel(dx, dy, duration=random.uniform(0.05, 0.1))

def enhanced_tab_switching():
    """More realistic tab switching behavior"""
    # Random delay before switching
    time.sleep(random.uniform(0.2, 0.5))
    
    # Press Ctrl key
    kb.press(Key.ctrl)
    time.sleep(random.uniform(0.1, 0.3))
    
    # Press Tab (1-3 times)
    tab_presses = random.randint(1, 3)
    for _ in range(tab_presses):
        kb.press(Key.tab)
        time.sleep(random.uniform(0.05, 0.15))
        kb.release(Key.tab)
        time.sleep(random.uniform(0.1, 0.3))
    
    # Release Ctrl
    kb.release(Key.ctrl)
    
    # Random delay after switching
    time.sleep(random.uniform(0.5, 1.5))

def random_click():
    """Occasional realistic clicks"""
    if random.random() > 0.7:  # 30% chance of clicking
        # Move to click position
        x, y = pyautogui.position()
        target_x = x + random.randint(-50, 50)
        target_y = y + random.randint(-50, 50)
        pyautogui.moveTo(target_x, target_y, duration=random.uniform(0.1, 0.3))
        
        # Click with human-like delay
        time.sleep(random.uniform(0.1, 0.4))
        pyautogui.click()
        time.sleep(random.uniform(0.1, 0.3))

def simulate_typing():
    """Simulates realistic typing behavior"""
    # Decide between single key or combination
    if random.random() > 0.7:  # 30% chance for key combo
        # Hold modifier
        modifier = random.choice([Key.shift, Key.ctrl, Key.alt])
        kb.press(modifier)
        time.sleep(random.uniform(0.1, 0.3))
        
        # Press main key
        key = random.choice(COMMON_KEYS[:8])  # Use letter keys for combos
        kb.press(key)
        time.sleep(random.uniform(0.05, 0.15))
        kb.release(key)
        
        # Release modifier
        time.sleep(random.uniform(0.1, 0.3))
        kb.release(modifier)
    else:
        # Single key press
        key = random.choice(COMMON_KEYS)
        kb.press(key)
        time.sleep(random.uniform(0.05, 0.2))
        kb.release(key)
    
    # Small random delay after typing
    time.sleep(random.uniform(0.1, 0.5))

def realistic_scrolling():
    """Simulates natural scrolling behavior"""
    # Determine scroll direction and amount
    direction = -1 if random.random() > 0.5 else 1
    scroll_amount = random.randint(1, 5)
    
    # Sometimes do a big scroll (page up/down)
    if random.random() > 0.8:
        scroll_amount = random.randint(20, 40)
        direction = direction * 3  # Faster scroll
    
    # Break scrolling into chunks with small pauses
    for _ in range(scroll_amount):
        pyautogui.scroll(direction)
        time.sleep(random.uniform(0.05, 0.2))
        
        # Occasionally change direction slightly
        if random.random() > 0.9:
            direction = direction * -1
            time.sleep(random.uniform(0.1, 0.3))
    
    # Sometimes follow scroll with arrow keys
    if random.random() > 0.7:
        arrow_key = Key.down if direction > 0 else Key.up
        presses = random.randint(1, 3)
        for _ in range(presses):
            kb.press(arrow_key)
            time.sleep(random.uniform(0.05, 0.15))
            kb.release(arrow_key)
            time.sleep(random.uniform(0.1, 0.3))

def system_activity():
    """Generate system-wide activity"""
    choice = random.random()
    
    if choice < 0.35:  # 35% chance
        human_like_movement()
        random_click()
    elif choice < 0.6:  # 25% chance
        enhanced_tab_switching()
    elif choice < 0.8:  # 20% chance
        simulate_typing()
    else:  # 20% chance
        realistic_scrolling()

def main():
    print("Ultimate activity simulator with enhanced scrolling")
    print("Press ESC to stop or move mouse to corner for emergency stop")
    
    try:
        while True:
            # Perform activity
            system_activity()
            
            # Random wait period (8-25 seconds)
            wait_time = random.randint(8, 25)
            print(f"Next activity in {wait_time} seconds...")
            
            # Check for exit key during wait
            for _ in range(wait_time):
                if keyboard.is_pressed('esc'):
                    print("\nStopping script...")
                    sys.exit()
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nScript stopped by user")
    except Exception as e:
        print(f"\nError occurred: {e}")

if __name__ == "__main__":
    # Check if script has admin privileges
    try:
        pyautogui.moveTo(1, 1)  # Test mouse control
        main()
    except pyautogui.FailSafeException:
        print("Emergency stop triggered by moving mouse to corner")
    except Exception as e:
        print(f"Error: {e}\nTry running as administrator")