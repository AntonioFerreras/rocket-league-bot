"""
Action Playback Tool - Converts recorded bot actions to keyboard inputs.
Uses pydirectinput (Windows) or pynput (Linux/Mac) for keyboard simulation with a tkinter GUI.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import threading
import time
import json
import os
import platform

# Detect OS and import appropriate keyboard library
SYSTEM = platform.system()

if SYSTEM == "Windows":
    try:
        import pydirectinput
        pydirectinput.PAUSE = 0  # Disable default pause between inputs
        KEYBOARD_LIB = "pydirectinput"
    except ImportError:
        print("ERROR: pydirectinput not installed. Install with: pip install pydirectinput")
        exit(1)
else:
    # Linux/Mac - use pynput
    try:
        from pynput.keyboard import Key, Controller as KeyboardController
        KEYBOARD_LIB = "pynput"
        _pynput_keyboard = KeyboardController()
        
        # Map string key names to pynput keys
        _PYNPUT_SPECIAL_KEYS = {
            "space": Key.space,
            "shift": Key.shift,
            "ctrl": Key.ctrl,
            "alt": Key.alt,
            "tab": Key.tab,
            "enter": Key.enter,
            "backspace": Key.backspace,
            "escape": Key.esc,
            "up": Key.up,
            "down": Key.down,
            "left": Key.left,
            "right": Key.right,
            "f1": Key.f1, "f2": Key.f2, "f3": Key.f3, "f4": Key.f4,
            "f5": Key.f5, "f6": Key.f6, "f7": Key.f7, "f8": Key.f8,
            "f9": Key.f9, "f10": Key.f10, "f11": Key.f11, "f12": Key.f12,
        }
        
        def _get_pynput_key(key_str):
            """Convert string key name to pynput key."""
            key_str = key_str.lower().strip()
            if key_str in _PYNPUT_SPECIAL_KEYS:
                return _PYNPUT_SPECIAL_KEYS[key_str]
            elif len(key_str) == 1:
                return key_str
            else:
                # Try as-is for single chars
                return key_str[0] if key_str else None
                
    except ImportError:
        print("ERROR: pynput not installed. Install with: pip install pynput")
        exit(1)

print(f"Running on {SYSTEM}, using {KEYBOARD_LIB} for keyboard input")

# Default key mappings for Rocket League controls
DEFAULT_KEY_MAPPING = {
    "throttle_forward": "w",
    "throttle_backward": "s",
    "steer_left": "a",
    "steer_right": "d",
    "pitch_up": "down",      # Nose down (pull back)
    "pitch_down": "up",      # Nose up (push forward)
    "yaw_left": "a",         # Same as steer in air
    "yaw_right": "d",        # Same as steer in air
    "roll_left": "q",
    "roll_right": "e",
    "jump": "space",
    "boost": "shift",
    "handbrake": "x",
}

# Rocket League action lookup table structure
# Actions are: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
# Each can be -1, 0, or 1 (except jump/boost/handbrake which are 0 or 1)


class LookupTableDecoder:
    """Decodes action indices back to control values using rlgym's exact lookup table."""
    
    def __init__(self):
        # This is the EXACT lookup table from rlgym's LookupTableAction
        # Format: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        self.lookup_table = np.array([
            [-1, -1, 0, -1, 0, 0, 0, 0],
            [-1, -1, 0, -1, 0, 0, 0, 1],
            [-1, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 1],
            [-1, 1, 0, 1, 0, 0, 0, 0],
            [-1, 1, 0, 1, 0, 0, 0, 1],
            [0, -1, 0, -1, 0, 0, 0, 0],
            [0, -1, 0, -1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 1],
            [1, -1, 0, -1, 0, 0, 0, 0],
            [1, -1, 0, -1, 0, 0, 0, 1],
            [1, -1, 0, -1, 0, 0, 1, 0],
            [1, -1, 0, -1, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, 1, 1],
            [0, -1, -1, -1, -1, 0, 0, 0],
            [1, -1, -1, -1, -1, 0, 1, 0],
            [0, -1, -1, -1, 0, 0, 0, 0],
            [1, -1, -1, -1, 0, 0, 1, 0],
            [0, -1, -1, -1, 1, 0, 0, 0],
            [1, -1, -1, -1, 1, 0, 1, 0],
            [0, 0, -1, 0, -1, 0, 0, 0],
            [1, 0, -1, 0, -1, 0, 1, 0],
            [0, 0, -1, 0, -1, 1, 0, 1],
            [1, 0, -1, 0, -1, 1, 1, 1],
            [0, 0, -1, 0, 0, 0, 0, 0],
            [1, 0, -1, 0, 0, 0, 1, 0],
            [0, 0, -1, 0, 0, 1, 0, 1],
            [1, 0, -1, 0, 0, 1, 1, 1],
            [0, 0, -1, 0, 1, 0, 0, 0],
            [1, 0, -1, 0, 1, 0, 1, 0],
            [0, 0, -1, 0, 1, 1, 0, 1],
            [1, 0, -1, 0, 1, 1, 1, 1],
            [0, 1, -1, 1, -1, 0, 0, 0],
            [1, 1, -1, 1, -1, 0, 1, 0],
            [0, 1, -1, 1, 0, 0, 0, 0],
            [1, 1, -1, 1, 0, 0, 1, 0],
            [0, 1, -1, 1, 1, 0, 0, 0],
            [1, 1, -1, 1, 1, 0, 1, 0],
            [0, -1, 0, -1, -1, 0, 0, 0],
            [1, -1, 0, -1, -1, 0, 1, 0],
            [0, -1, 0, -1, 1, 0, 0, 0],
            [1, -1, 0, -1, 1, 0, 1, 0],
            [0, 0, 0, 0, -1, 0, 0, 0],
            [1, 0, 0, 0, -1, 0, 1, 0],
            [0, 0, 0, 0, -1, 1, 0, 1],
            [1, 0, 0, 0, -1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 1, 1],
            [0, 1, 0, 1, -1, 0, 0, 0],
            [1, 1, 0, 1, -1, 0, 1, 0],
            [0, 1, 0, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 1, 0],
            [0, -1, 1, -1, -1, 0, 0, 0],
            [1, -1, 1, -1, -1, 0, 1, 0],
            [0, -1, 1, -1, 0, 0, 0, 0],
            [1, -1, 1, -1, 0, 0, 1, 0],
            [0, -1, 1, -1, 1, 0, 0, 0],
            [1, -1, 1, -1, 1, 0, 1, 0],
            [0, 0, 1, 0, -1, 0, 0, 0],
            [1, 0, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, -1, 1, 0, 1],
            [1, 0, 1, 0, -1, 1, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 1, 1, 1],
            [0, 0, 1, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 1],
            [0, 1, 1, 1, -1, 0, 0, 0],
            [1, 1, 1, 1, -1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 1, 0],
        ], dtype=np.float32)
    
    def decode(self, action_index):
        """Convert action index to control values."""
        if action_index < 0 or action_index >= len(self.lookup_table):
            return np.zeros(8, dtype=np.float32)
        return self.lookup_table[action_index]


class ActionPlaybackApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Action Playback Tool")
        self.root.geometry("500x650")
        self.root.resizable(False, False)
        
        self.decoder = LookupTableDecoder()
        self.actions_data = None
        self.is_playing = False
        self.playback_thread = None
        
        # Key mapping
        self.key_mapping = DEFAULT_KEY_MAPPING.copy()
        self.key_entries = {}
        
        # Currently pressed keys (for tracking state)
        self.pressed_keys = set()
        
        self._create_widgets()
        self._load_config()
    
    def _create_widgets(self):
        # File selection frame
        file_frame = ttk.LabelFrame(self.root, text="Actions File", padding=10)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        self.file_path_var = tk.StringVar(value="No file selected")
        ttk.Label(file_frame, textvariable=self.file_path_var, wraplength=400).pack(side="left", fill="x", expand=True)
        ttk.Button(file_frame, text="Browse...", command=self._browse_file).pack(side="right")
        
        # File info frame
        info_frame = ttk.LabelFrame(self.root, text="File Info", padding=10)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        self.info_var = tk.StringVar(value="No file loaded")
        ttk.Label(info_frame, textvariable=self.info_var, wraplength=460).pack()
        
        # Key mapping frame
        key_frame = ttk.LabelFrame(self.root, text="Key Mapping", padding=10)
        key_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create key mapping entries in a grid
        controls = [
            ("Throttle Forward", "throttle_forward"),
            ("Throttle Backward", "throttle_backward"),
            ("Steer Left", "steer_left"),
            ("Steer Right", "steer_right"),
            ("Pitch Up (nose down)", "pitch_up"),
            ("Pitch Down (nose up)", "pitch_down"),
            ("Roll Left", "roll_left"),
            ("Roll Right", "roll_right"),
            ("Jump", "jump"),
            ("Boost", "boost"),
            ("Handbrake", "handbrake"),
        ]
        
        for i, (label, key) in enumerate(controls):
            row = i // 2
            col = (i % 2) * 2
            
            ttk.Label(key_frame, text=label + ":").grid(row=row, column=col, sticky="e", padx=5, pady=2)
            entry = ttk.Entry(key_frame, width=10)
            entry.insert(0, self.key_mapping.get(key, ""))
            entry.grid(row=row, column=col+1, sticky="w", padx=5, pady=2)
            self.key_entries[key] = entry
        
        btn_row = len(controls)//2 + 1
        ttk.Button(key_frame, text="Save Key Mapping", command=self._save_key_mapping).grid(
            row=btn_row, column=0, columnspan=2, pady=10
        )
        ttk.Button(key_frame, text="Load Key Mapping", command=self._load_key_mapping_dialog).grid(
            row=btn_row, column=2, columnspan=2, pady=10
        )
        
        # Playback controls frame
        control_frame = ttk.LabelFrame(self.root, text="Playback Controls", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var, font=("Arial", 12, "bold")).pack(pady=5)
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=5)
        
        self.start_btn = ttk.Button(btn_frame, text="▶ Start Playback (3s delay)", command=self._start_playback)
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="■ Stop", command=self._stop_playback, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=5)
        
        # Instructions
        instructions = ttk.LabelFrame(self.root, text="Instructions", padding=10)
        instructions.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(instructions, text=(
            "1. Load an actions file (.npz) recorded from replay\n"
            "2. Configure key mappings to match your Rocket League bindings\n"
            "3. Click Start - you have 3 seconds to focus the game window\n"
            "4. The bot's actions will be played as keyboard inputs"
        ), justify="left").pack()
    
    def _browse_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Actions File",
            filetypes=[("NumPy files", "*.npz"), ("All files", "*.*")]
        )
        if filepath:
            self._load_actions_file(filepath)
    
    def _load_actions_file(self, filepath):
        try:
            data = np.load(filepath)
            self.actions_data = {
                "actions": data["actions"],
                "ticks": data["ticks"] if "ticks" in data else None,
                "action_repeat": int(data["action_repeat"][0]) if "action_repeat" in data else 8,
            }
            self.file_path_var.set(os.path.basename(filepath))
            
            num_actions = len(self.actions_data["actions"])
            action_repeat = self.actions_data["action_repeat"]
            total_ticks = num_actions * action_repeat
            duration = total_ticks / 120.0  # 120 ticks per second
            
            self.info_var.set(
                f"Actions: {num_actions} | Action Repeat: {action_repeat}\n"
                f"Total Ticks: {total_ticks} | Duration: {duration:.2f}s"
            )
            self.status_var.set("File loaded - Ready to play")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
            self.actions_data = None
            self.info_var.set("No file loaded")
    
    def _save_key_mapping(self):
        for key, entry in self.key_entries.items():
            self.key_mapping[key] = entry.get().strip().lower()
        
        # Save to config file
        config_path = os.path.join(os.path.dirname(__file__), "action_playback_config.json")
        try:
            with open(config_path, "w") as f:
                json.dump(self.key_mapping, f, indent=2)
            messagebox.showinfo("Saved", "Key mapping saved!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config:\n{str(e)}")
    
    def _load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "action_playback_config.json")
        self._load_config_from_file(config_path)
    
    def _load_config_from_file(self, config_path):
        """Load key mapping from a specific file."""
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    self.key_mapping = json.load(f)
                # Update entries
                for key, entry in self.key_entries.items():
                    entry.delete(0, tk.END)
                    entry.insert(0, self.key_mapping.get(key, ""))
                return True
            except Exception as e:
                print(f"Error loading config: {e}")
                return False
        return False
    
    def _load_key_mapping_dialog(self):
        """Open file dialog to load a key mapping file."""
        filepath = filedialog.askopenfilename(
            title="Select Key Mapping File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=os.path.dirname(__file__)
        )
        if filepath:
            if self._load_config_from_file(filepath):
                messagebox.showinfo("Loaded", f"Key mapping loaded from:\n{os.path.basename(filepath)}")
            else:
                messagebox.showerror("Error", f"Failed to load key mapping from:\n{filepath}")
    
    def _start_playback(self):
        if self.actions_data is None:
            messagebox.showwarning("No File", "Please load an actions file first!")
            return
        
        # Update key mapping from entries
        for key, entry in self.key_entries.items():
            self.key_mapping[key] = entry.get().strip().lower()
        
        self.is_playing = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
    
    def _stop_playback(self):
        self.is_playing = False
        self._release_all_keys()
        self.status_var.set("Stopped")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.progress_var.set(0)
    
    def _release_all_keys(self):
        """Release all currently pressed keys."""
        for key in list(self.pressed_keys):
            try:
                self._do_key_up(key)
            except:
                pass
        self.pressed_keys.clear()
    
    def _do_key_down(self, key):
        """Low-level key press using appropriate library."""
        if KEYBOARD_LIB == "pydirectinput":
            pydirectinput.keyDown(key)
        else:
            pynput_key = _get_pynput_key(key)
            if pynput_key:
                _pynput_keyboard.press(pynput_key)
    
    def _do_key_up(self, key):
        """Low-level key release using appropriate library."""
        if KEYBOARD_LIB == "pydirectinput":
            pydirectinput.keyUp(key)
        else:
            pynput_key = _get_pynput_key(key)
            if pynput_key:
                _pynput_keyboard.release(pynput_key)
    
    def _press_key(self, key):
        """Press a key if not already pressed."""
        if key and key not in self.pressed_keys:
            try:
                self._do_key_down(key)
                self.pressed_keys.add(key)
            except Exception as e:
                print(f"Error pressing {key}: {e}")
    
    def _release_key(self, key):
        """Release a key if pressed."""
        if key and key in self.pressed_keys:
            try:
                self._do_key_up(key)
                self.pressed_keys.discard(key)
            except Exception as e:
                print(f"Error releasing {key}: {e}")
    
    def _apply_action(self, action_values):
        """Convert action values to key presses/releases."""
        throttle, steer, pitch, yaw, roll, jump, boost, handbrake = action_values
        
        # Throttle
        if throttle > 0.5:
            self._press_key(self.key_mapping.get("throttle_forward"))
            self._release_key(self.key_mapping.get("throttle_backward"))
        elif throttle < -0.5:
            self._release_key(self.key_mapping.get("throttle_forward"))
            self._press_key(self.key_mapping.get("throttle_backward"))
        else:
            self._release_key(self.key_mapping.get("throttle_forward"))
            self._release_key(self.key_mapping.get("throttle_backward"))
        
        # Steer (on ground) / Yaw (in air) - typically same keys
        if steer > 0.5 or yaw > 0.5:
            self._release_key(self.key_mapping.get("steer_left"))
            self._press_key(self.key_mapping.get("steer_right"))
        elif steer < -0.5 or yaw < -0.5:
            self._press_key(self.key_mapping.get("steer_left"))
            self._release_key(self.key_mapping.get("steer_right"))
        else:
            self._release_key(self.key_mapping.get("steer_left"))
            self._release_key(self.key_mapping.get("steer_right"))
        
        # Pitch
        if pitch > 0.5:
            self._press_key(self.key_mapping.get("pitch_up"))
            self._release_key(self.key_mapping.get("pitch_down"))
        elif pitch < -0.5:
            self._release_key(self.key_mapping.get("pitch_up"))
            self._press_key(self.key_mapping.get("pitch_down"))
        else:
            self._release_key(self.key_mapping.get("pitch_up"))
            self._release_key(self.key_mapping.get("pitch_down"))
        
        # Roll
        if roll > 0.5:
            self._release_key(self.key_mapping.get("roll_left"))
            self._press_key(self.key_mapping.get("roll_right"))
        elif roll < -0.5:
            self._press_key(self.key_mapping.get("roll_left"))
            self._release_key(self.key_mapping.get("roll_right"))
        else:
            self._release_key(self.key_mapping.get("roll_left"))
            self._release_key(self.key_mapping.get("roll_right"))
        
        # Jump
        if jump > 0.5:
            self._press_key(self.key_mapping.get("jump"))
        else:
            self._release_key(self.key_mapping.get("jump"))
        
        # Boost
        if boost > 0.5:
            self._press_key(self.key_mapping.get("boost"))
        else:
            self._release_key(self.key_mapping.get("boost"))
        
        # Handbrake
        if handbrake > 0.5:
            self._press_key(self.key_mapping.get("handbrake"))
        else:
            self._release_key(self.key_mapping.get("handbrake"))
    
    def _playback_worker(self):
        """Background thread that plays back actions."""
        try:
            # 3 second countdown
            for i in range(3, 0, -1):
                if not self.is_playing:
                    return
                self.root.after(0, lambda x=i: self.status_var.set(f"Starting in {x}..."))
                time.sleep(1)
            
            if not self.is_playing:
                return
            
            self.root.after(0, lambda: self.status_var.set("Playing..."))
            
            actions = self.actions_data["actions"]
            action_repeat = self.actions_data["action_repeat"]
            
            # Time per action decision (action_repeat ticks at 120 Hz)
            time_per_action = action_repeat / 120.0
            
            num_actions = len(actions)
            start_time = time.perf_counter()
            
            for i, action_idx in enumerate(actions):
                if not self.is_playing:
                    break
                
                # Decode action index to control values
                action_values = self.decoder.decode(action_idx)
                
                # Apply the action (press/release keys)
                self._apply_action(action_values)
                
                # Update progress
                progress = (i + 1) / num_actions * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                
                # Wait for next action timing
                target_time = start_time + (i + 1) * time_per_action
                sleep_time = target_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Release all keys when done
            self._release_all_keys()
            
            if self.is_playing:
                self.root.after(0, lambda: self.status_var.set("Playback complete!"))
                self.root.after(0, lambda: self.start_btn.config(state="normal"))
                self.root.after(0, lambda: self.stop_btn.config(state="disabled"))
                self.is_playing = False
                
        except Exception as e:
            self._release_all_keys()
            self.root.after(0, lambda: messagebox.showerror("Error", f"Playback error:\n{str(e)}"))
            self.root.after(0, lambda: self._stop_playback())


def main():
    root = tk.Tk()
    app = ActionPlaybackApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

