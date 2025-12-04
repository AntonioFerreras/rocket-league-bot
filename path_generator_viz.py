import tkinter as tk
import numpy as np
from path_generator import generate_random_path

# Field constants from rlgym/rocket_league/common_values.py
SIDE_WALL_X = 4096
BACK_WALL_Y = 5120
BACK_NET_Y = 6000  # Added for goal visualization
CEILING_Z = 2044
GOAL_HEIGHT = 642.775
GOAL_CENTER_TO_POST = 892.755
CORNER_CATHETUS_LENGTH = 1152

# Margins
FLOOR_MARGIN = 500
CEILING_MARGIN = 150
WALL_MARGIN = 100

# Derived constants
FIELD_X_RANGE = (-SIDE_WALL_X, SIDE_WALL_X)
FIELD_Y_RANGE = (-BACK_WALL_Y, BACK_WALL_Y)
FIELD_Z_RANGE = (0, CEILING_Z)

# Safe ranges
SAFE_X_RANGE = (-SIDE_WALL_X + WALL_MARGIN, SIDE_WALL_X - WALL_MARGIN)
SAFE_Y_RANGE = (-BACK_WALL_Y + WALL_MARGIN, BACK_WALL_Y - WALL_MARGIN)
SAFE_Z_RANGE = (FLOOR_MARGIN, CEILING_Z - CEILING_MARGIN)

# Goal locations (Y planes)
GOAL_Y_BLUE = -BACK_WALL_Y
GOAL_Y_ORANGE = BACK_WALL_Y

class PathVisualizer:
    def __init__(self, blocking=True):
        self.root = tk.Tk()
        self.root.title("Rocket League Path Generator (Tkinter)")
        
        # Window setup
        self.width = 800
        self.height = 900
        # Expand width for controls sidebar
        self.root.geometry(f"{self.width + 300}x{self.height}")
        
        # Main Container (Canvas + Sidebar)
        container = tk.Frame(self.root)
        container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Canvas for drawing
        self.canvas = tk.Canvas(container, width=self.width, height=self.height - 50, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Sidebar for Controls
        self.sidebar = tk.Frame(container, width=300, bg="#f0f0f0")
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar.pack_propagate(False) # Enforce width

        tk.Label(self.sidebar, text="Conditions", font=("Arial", 24, "bold"), bg="#f0f0f0").pack(pady=10)
        
        self.condition_labels = []
        # Indices 6 to 15
        self.condition_names = ["Gorilla Glue"] + [""] * 9 
        
        for name in self.condition_names:
            # If name is empty, the label exists but might be invisible depending on layout/text.
            # We use a non-breaking space or similar to keep height if needed, 
            # but user asked for "" so we stick to that.
            lbl = tk.Label(self.sidebar, text=name, font=("Arial", 20, "bold"), fg="#bbbbbb", bg="#f0f0f0")
            lbl.pack(anchor="w", padx=10, pady=2)
            self.condition_labels.append(lbl)
        
        self.condition_data = None # Store current condition data

        # Control Frame for Button and Input
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, pady=10)

        # Button
        self.btn = tk.Button(control_frame, text="New Path", command=self.update_path, font=("Arial", 12))
        self.btn.pack(side=tk.LEFT, padx=10)
        
        # Step Distance Input
        tk.Label(control_frame, text="Step Dist:").pack(side=tk.LEFT, padx=5)
        self.step_dist_var = tk.StringVar(value="1000")
        self.entry_step_dist = tk.Entry(control_frame, textvariable=self.step_dist_var, width=5)
        self.entry_step_dist.pack(side=tk.LEFT)
        
        if blocking:
            self.update_path()
            self.root.mainloop()

    def process_events(self):
        self.root.update()
        self.root.update_idletasks()

    def update_controls(self, current_target_index):
        if self.condition_data is None:
            return
        
        # Handle 1D array (from standalone viz) or 2D array (from training)
        if len(self.condition_data.shape) == 1:
            # It's just the glue column (index 6)
             conditions = np.zeros(16)
             if current_target_index < len(self.condition_data):
                 conditions[6] = self.condition_data[current_target_index]
        else:
             if current_target_index < len(self.condition_data):
                 conditions = self.condition_data[current_target_index]
             else:
                 return

        for i, lbl in enumerate(self.condition_labels):
            cond_idx = i + 6
            if cond_idx < len(conditions) and conditions[cond_idx] > 0.5:
                lbl.config(fg="#00FF00")
            else:
                lbl.config(fg="#bbbbbb") # close to background

    def show_path(self, path, start, end, control, condition_data=None):
        self.canvas.delete("all")
        
        self.condition_data = condition_data
        
        self.draw_field_top_down()
        self.draw_field_side_view()
        
        # Draw Path Top-Down
        td_points = [self.transform_top_down(p[0], p[1]) for p in path]
        # Flatten list of tuples for create_line
        td_coords = [coord for point in td_points for coord in point]
        if len(td_coords) >= 4:
            self.canvas.create_line(td_coords, fill="red", width=2, smooth=True)

        # Draw points
        for i, (px, py) in enumerate(td_points):
            color = "yellow"
            is_glue = False
            if condition_data is not None and i < len(condition_data):
                if len(condition_data.shape) == 1:
                     if condition_data[i] > 0.5: is_glue = True
                elif condition_data.shape[1] > 6:
                     if condition_data[i, 6] > 0.5: is_glue = True
            
            if is_glue:
                color = "orange"
            self.canvas.create_oval(px-2, py-2, px+2, py+2, fill=color, outline=color)
        
        # Start/End/Control Top-Down
        sx, sy = self.transform_top_down(start[0], start[1])
        self.canvas.create_oval(sx-4, sy-4, sx+4, sy+4, fill="green")
        
        ex, ey = self.transform_top_down(end[0], end[1])
        self.canvas.create_line(ex-5, ey-5, ex+5, ey+5, fill="red", width=2)
        self.canvas.create_line(ex-5, ey+5, ex+5, ey-5, fill="red", width=2)
        
        cx, cy = self.transform_top_down(control[0], control[1])
        self.canvas.create_oval(cx-2, cy-2, cx+2, cy+2, fill="gray")
        
        # Draw Path Side View
        sv_points = [self.transform_side_view(p[1], p[2]) for p in path]
        sv_coords = [coord for point in sv_points for coord in point]
        if len(sv_coords) >= 4:
            self.canvas.create_line(sv_coords, fill="red", width=2, smooth=True)

        # Draw points
        for i, (px, py) in enumerate(sv_points):
            color = "yellow"
            is_glue = False
            if condition_data is not None and i < len(condition_data):
                if len(condition_data.shape) == 1:
                     if condition_data[i] > 0.5: is_glue = True
                elif condition_data.shape[1] > 6:
                     if condition_data[i, 6] > 0.5: is_glue = True
            
            if is_glue:
                color = "orange"
            self.canvas.create_oval(px-2, py-2, px+2, py+2, fill=color, outline=color)
        
        # Start/End/Control Side View
        sx, sy = self.transform_side_view(start[1], start[2])
        self.canvas.create_oval(sx-4, sy-4, sx+4, sy+4, fill="green")
        
        ex, ey = self.transform_side_view(end[1], end[2])
        self.canvas.create_line(ex-5, ey-5, ex+5, ey+5, fill="red", width=2)
        self.canvas.create_line(ex-5, ey+5, ex+5, ey-5, fill="red", width=2)
        
        cx, cy = self.transform_side_view(control[1], control[2])
        self.canvas.create_oval(cx-2, cy-2, cx+2, cy+2, fill="gray")

    def transform_top_down(self, x, y):
        # Map Field X/Y to Screen X/Y
        # Field X: -4096 to 4096 -> Screen X
        # Field Y: -6000 to 6000 -> Screen Y (Included nets)
        
        scale = 0.05  # Reduced Scale factor to fit everything
        cx, cy = 400, 300  # Center of Top-Down View
        
        screen_x = cx + x * scale
        screen_y = cy + y * scale  # +y is down in screen, +y is orange in game.
        return screen_x, screen_y

    def transform_side_view(self, y, z):
        # Map Field Y/Z to Screen X/Y
        # Field Y: -6000 to 6000 -> Screen X
        # Field Z: 0 to 2044 -> Screen Y
        
        scale = 0.05  # Reduced Scale factor to fit everything
        cx, cy = 400, 750 # Center of Side View (Y axis)
        
        screen_x = cx + y * scale
        screen_y = cy - z * scale # Invert Z because screen Y is down
        return screen_x, screen_y

    def draw_field_top_down(self):
        # Draw Title
        self.canvas.create_text(150, 20, text="Top-Down View (X vs Y)", font=("Arial", 14, "bold"))
        
        # Draw Field Boundary
        # Outline coordinates (CCW from Top Right)
        # Corners are cut: (SideX, BackY-Corner) -> (SideX-Corner, BackY)
        points = [
            (SIDE_WALL_X, BACK_WALL_Y - CORNER_CATHETUS_LENGTH), # Orange Right Corner Start
            (SIDE_WALL_X - CORNER_CATHETUS_LENGTH, BACK_WALL_Y), # Orange Right Corner End
            (-SIDE_WALL_X + CORNER_CATHETUS_LENGTH, BACK_WALL_Y), # Orange Left Corner Start
            (-SIDE_WALL_X, BACK_WALL_Y - CORNER_CATHETUS_LENGTH), # Orange Left Corner End
            (-SIDE_WALL_X, -BACK_WALL_Y + CORNER_CATHETUS_LENGTH), # Blue Left Corner Start
            (-SIDE_WALL_X + CORNER_CATHETUS_LENGTH, -BACK_WALL_Y), # Blue Left Corner End
            (SIDE_WALL_X - CORNER_CATHETUS_LENGTH, -BACK_WALL_Y), # Blue Right Corner Start
            (SIDE_WALL_X, -BACK_WALL_Y + CORNER_CATHETUS_LENGTH), # Blue Right Corner End
        ]
        
        screen_points = [self.transform_top_down(p[0], p[1]) for p in points]
        self.canvas.create_polygon(screen_points, outline="black", fill="#e0e0e0", width=2)
        
        # Draw Goals (Top Down)
        # Orange Goal (Top)
        og_x1, og_y1 = self.transform_top_down(-GOAL_CENTER_TO_POST, BACK_WALL_Y)
        og_x2, og_y2 = self.transform_top_down(GOAL_CENTER_TO_POST, BACK_NET_Y)
        self.canvas.create_rectangle(og_x1, og_y1, og_x2, og_y2, outline="orange", width=2)

        # Blue Goal (Bottom)
        bg_x1, bg_y1 = self.transform_top_down(-GOAL_CENTER_TO_POST, -BACK_WALL_Y)
        bg_x2, bg_y2 = self.transform_top_down(GOAL_CENTER_TO_POST, -BACK_NET_Y)
        self.canvas.create_rectangle(bg_x1, bg_y1, bg_x2, bg_y2, outline="blue", width=2)

    def draw_field_side_view(self):
        # Draw Title
        self.canvas.create_text(100, 580, text="Side View (Y vs Z)", font=("Arial", 14, "bold"))
        
        # Draw Floor
        f_start = self.transform_side_view(-BACK_WALL_Y, 0)
        f_end = self.transform_side_view(BACK_WALL_Y, 0)
        self.canvas.create_line(f_start[0], f_start[1], f_end[0], f_end[1], fill="black", width=2)
        
        # Draw Ceiling
        c_start = self.transform_side_view(-BACK_WALL_Y, CEILING_Z)
        c_end = self.transform_side_view(BACK_WALL_Y, CEILING_Z)
        self.canvas.create_line(c_start[0], c_start[1], c_end[0], c_end[1], fill="black", width=2)
        
        # Draw End Walls
        w_blue_start = self.transform_side_view(-BACK_WALL_Y, 0)
        w_blue_end = self.transform_side_view(-BACK_WALL_Y, CEILING_Z)
        self.canvas.create_line(w_blue_start[0], w_blue_start[1], w_blue_end[0], w_blue_end[1], fill="black", width=2)
        
        w_orange_start = self.transform_side_view(BACK_WALL_Y, 0)
        w_orange_end = self.transform_side_view(BACK_WALL_Y, CEILING_Z)
        self.canvas.create_line(w_orange_start[0], w_orange_start[1], w_orange_end[0], w_orange_end[1], fill="black", width=2)

        # Draw Goals (Side View)
        # Orange Goal
        og_y_start = BACK_WALL_Y
        og_y_end = BACK_NET_Y
        og_z_top = GOAL_HEIGHT
        
        p1 = self.transform_side_view(og_y_start, 0)
        p2 = self.transform_side_view(og_y_end, 0)
        p3 = self.transform_side_view(og_y_end, og_z_top)
        p4 = self.transform_side_view(og_y_start, og_z_top)
        self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="orange", width=2)
        self.canvas.create_line(p2[0], p2[1], p3[0], p3[1], fill="orange", width=2)
        self.canvas.create_line(p3[0], p3[1], p4[0], p4[1], fill="orange", width=2)
        
        # Blue Goal
        bg_y_start = -BACK_WALL_Y
        bg_y_end = -BACK_NET_Y
        bg_z_top = GOAL_HEIGHT
        
        p1 = self.transform_side_view(bg_y_start, 0)
        p2 = self.transform_side_view(bg_y_end, 0)
        p3 = self.transform_side_view(bg_y_end, bg_z_top)
        p4 = self.transform_side_view(bg_y_start, bg_z_top)
        self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="blue", width=2)
        self.canvas.create_line(p2[0], p2[1], p3[0], p3[1], fill="blue", width=2)
        self.canvas.create_line(p3[0], p3[1], p4[0], p4[1], fill="blue", width=2)


    def update_path(self):
        try:
            step_dist = float(self.step_dist_var.get())
        except ValueError:
            step_dist = 1000
            
        path, start, end, control, glue_conditions = generate_random_path(step_distance=step_dist)
        
        # In standalone, we only have glue conditions (1D array)
        # show_path handles 1D array as the glue column
        self.show_path(path, start, end, control, condition_data=glue_conditions)

    def update_ball(self, position):
        x, y, z = position
        
        # Top Down
        screen_x, screen_y = self.transform_top_down(x, y)
        r = 6 # Radius
        if self.canvas.find_withtag("ball_top_down"):
            self.canvas.coords("ball_top_down", screen_x-r, screen_y-r, screen_x+r, screen_y+r)
            self.canvas.tag_raise("ball_top_down")
        else:
            self.canvas.create_oval(screen_x-r, screen_y-r, screen_x+r, screen_y+r, fill="purple", outline="black", tags="ball_top_down")

        # Side View
        screen_x_side, screen_y_side = self.transform_side_view(y, z)
        if self.canvas.find_withtag("ball_side_view"):
            self.canvas.coords("ball_side_view", screen_x_side-r, screen_y_side-r, screen_x_side+r, screen_y_side+r)
            self.canvas.tag_raise("ball_side_view")
        else:
             self.canvas.create_oval(screen_x_side-r, screen_y_side-r, screen_x_side+r, screen_y_side+r, fill="purple", outline="black", tags="ball_side_view")

if __name__ == "__main__":
    viz = PathVisualizer()
