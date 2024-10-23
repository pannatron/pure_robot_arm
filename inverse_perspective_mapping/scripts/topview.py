#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String,Int32
from cv_bridge import CvBridge
import cv2
import numpy as np
import tkinter as tk
from PIL import Image as PILImage, ImageTk
from threading import Thread
from std_msgs.msg import Int32  # เพิ่มการ import Int32

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        # Subscriptions
        self.subscription = self.create_subscription(
            Image,
            '/v4l/camera/image_raw',
            self.image_callback,
            1)
        self.ai_detected_subscription = self.create_subscription(
            Image,
            '/ai_detected_image',
            self.ai_detected_callback,
            1)

        self.br = CvBridge()

        # Aruco parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()


        # Variables for UI and processing
        self.clicked_points = []
        self.clicked_point = None
        self.width_cm = 70.0
        self.height_cm = 50.0
        self.width_px = 600
        self.height_px = 600
        self.scale_x = self.width_cm / self.width_px
        self.scale_y = self.height_cm / self.height_px
        # Default delay values for each process
        self.move_to_delay = 5
        self.pick_delay = 5
        self.place_delay = 5
        # Default Z-axis value
        self.z_value = 120.0

        # Publishers for coordinates and commands
        self.command_pub = self.create_publisher(String, 'serial_commands', 10)
        self.ai_start_process = self.create_publisher(String, 'ai_start_process', 10)

        self.top_view_pub = self.create_publisher(Image, 'top_view_image', 10)
        self.tray_position_pub = self.create_publisher(String, 'tray_positions', 10)
 # Publishers for delay times
        self.move_to_delay_pub = self.create_publisher(Int32, 'move_to_delay', 10)
        self.pick_delay_pub = self.create_publisher(Int32, 'pick_delay', 10)
        self.place_delay_pub = self.create_publisher(Int32, 'place_delay', 10)
        self.z_value_pub = self.create_publisher(Int32, 'z_value', 10)

        # Mode flags
        self.topview_mode = False
        self.manual_mode = False
        self.ai_detect_mode = False
        self.display_topview = False
        self.tray_setting_mode = None
        # Variables for topview lock status
        self.topview_locked = False

        # Tray positions dictionary
        self.tray_positions = {
            "circle_large": None,
            "circle_medium": None,
            "circle_small": None,
            "square_large": None,
            "square_medium": None,
            "square_small": None,
            "triangle_large": None,
            "triangle_medium": None,
            "triangle_small": None,
        }

        # Flag to check if UI is ready
        self.ui_ready = False

        # Stored position to send command after clicking "Send Command"
        self.stored_point = None

        # Initialize UI
        self.init_ui()
        self.get_logger().info('Aruco Detector Node has been started.')

    def init_ui(self):
        # Create a separate thread for Tkinter UI
        self.ui_thread = Thread(target=self.create_ui)
        self.ui_thread.start()

    def wait_for_ui(self):
        # Wait until UI is ready
        while not self.ui_ready:
            pass
    def adjust_z(self, delta):
        """ Adjust the Z-axis value by delta (+1 or -1) """
        current_z = self.z_slider.get()
        new_z = current_z + delta
        # Ensure the new Z value is within the slider's range
        new_z = max(self.z_slider.cget("from"), min(self.z_slider.cget("to"), new_z))
        self.z_slider.set(new_z)
        self.z_entry.delete(0, tk.END)
        self.z_entry.insert(0, str(new_z))

    def create_ui(self):
        # Initialize the main window
        self.root = tk.Tk()
        self.root.title("Aruco Detector Control Panel")

        # Create the main frame that holds all other frames
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a frame for mode selection buttons
        mode_frame = tk.LabelFrame(main_frame, text="Mode Selection", padx=10, pady=10)
        mode_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Add buttons for different modes
        self.btn_topview = tk.Button(mode_frame, text="TOPVIEW", command=self.enable_topview_mode)
        self.btn_topview.pack(fill='x', pady=5)

        self.btn_ai_detect = tk.Button(mode_frame, text="AI Detect ON", command=self.publish_ai_detect)
        self.btn_ai_detect.pack(fill='x', pady=5)


        self.btn_ai_detect_OFF = tk.Button(mode_frame, text="AI Detect OFF", command=self.publish_ai_detect_OFF)
        self.btn_ai_detect_OFF.pack(fill='x', pady=5)


        self.btn_manual = tk.Button(mode_frame, text="Manual", command=self.enable_manual_mode)
        self.btn_manual.pack(fill='x', pady=5)
# Create a frame for Delay Time Control
        delay_frame = tk.LabelFrame(mode_frame, text="Delay Time Settings", padx=10, pady=10)
        delay_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Move to Delay
        tk.Label(delay_frame, text="Move To Delay (sec)").grid(row=0, column=0, padx=5, pady=5)
        self.move_to_delay_entry = tk.Entry(delay_frame, width=5)
        self.move_to_delay_entry.insert(0, str(self.move_to_delay))
        self.move_to_delay_entry.grid(row=0, column=1, padx=5, pady=5)

        # Pick Delay
        tk.Label(delay_frame, text="Pick Delay (sec)").grid(row=1, column=0, padx=5, pady=5)
        self.pick_delay_entry = tk.Entry(delay_frame, width=5)
        self.pick_delay_entry.insert(0, str(self.pick_delay))
        self.pick_delay_entry.grid(row=1, column=1, padx=5, pady=5)

        # Place Delay
        tk.Label(delay_frame, text="Place Delay (sec)").grid(row=2, column=0, padx=5, pady=5)
        self.place_delay_entry = tk.Entry(delay_frame, width=5)
        self.place_delay_entry.insert(0, str(self.place_delay))
        self.place_delay_entry.grid(row=2, column=1, padx=5, pady=5)

        # Update button to set the delay times
        self.update_delay_button = tk.Button(delay_frame, text="Update Delay", command=self.update_delays)
        self.update_delay_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Create a frame for robot control buttons
        control_frame = tk.LabelFrame(main_frame, text="Robot Control", padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        # Add buttons for robot control
        self.btn_sethome = tk.Button(control_frame, text="SETHOME", command=self.set_home)
        self.btn_sethome.pack(fill='x', pady=5)

        self.btn_pick = tk.Button(control_frame, text="PICK", command=self.pick_command)
        self.btn_pick.pack(fill='x', pady=5)

        self.btn_place = tk.Button(control_frame, text="PLACE", command=self.place_command)
        self.btn_place.pack(fill='x', pady=5)

        self.btn_reset = tk.Button(control_frame, text="RESET", command=self.reset_topview)
        self.btn_reset.pack(fill='x', pady=5)

        # Add button for sending command after marking the point
        self.btn_send_command = tk.Button(control_frame, text="SEND COMMAND", command=self.send_stored_command)
        self.btn_send_command.pack(fill='x', pady=5)

        # Label to display current command
        self.command_label = tk.Label(control_frame, text="Command: ", fg="blue")
        self.command_label.pack(fill='x', pady=20)

        # Label to display stored point information
        self.point_label = tk.Label(control_frame, text="Stored Point: None", fg="red")
        self.point_label.pack(fill='x', pady=20)
        # Create Z-axis control
        z_control_frame = tk.LabelFrame(control_frame, text="Z-Axis Control", padx=10, pady=10)
        z_control_frame.pack(fill='x', pady=5)

        # Create a frame to hold the buttons and the entry for Z-axis
        z_control_buttons_frame = tk.Frame(z_control_frame)
        z_control_buttons_frame.pack(fill='x', pady=5)

        # Button to decrease Z value by 1
        self.btn_z_decrease = tk.Button(z_control_buttons_frame, text="-", command=lambda: self.adjust_z(-1))
        self.btn_z_decrease.grid(row=0, column=0, padx=5)

        # Entry field for manual input of Z value
        self.z_entry = tk.Entry(z_control_buttons_frame, width=5)
        self.z_entry.insert(0, str(self.z_value))
        self.z_entry.grid(row=0, column=1, padx=5)

        # Button to increase Z value by 1
        self.btn_z_increase = tk.Button(z_control_buttons_frame, text="+", command=lambda: self.adjust_z(1))
        self.btn_z_increase.grid(row=0, column=2, padx=5)

        # Add a scale for Z-axis (optional if you still want to use the slider)
        self.z_slider = tk.Scale(z_control_frame, from_=0, to=300, orient=tk.HORIZONTAL, label="Z-Axis (cm)")
        self.z_slider.set(self.z_value)
        self.z_slider.pack(fill='x')

        # Bind slider and entry events for Z value synchronization
        self.z_slider.bind("<Motion>", self.update_z_entry)
        self.z_entry.bind("<Return>", self.update_z_slider)

        # Create a frame for X-Y Offset Controls
        offset_frame = tk.LabelFrame(control_frame, text="X-Y Offset Control", padx=10, pady=10)
        offset_frame.pack(fill='x', pady=5)

        # Entry for X offset with closer buttons
        tk.Label(offset_frame, text="X Offset").grid(row=0, column=0, padx=5)
        tk.Button(offset_frame, text="-", command=lambda: self.adjust_offset("x", -1)).grid(row=0, column=1, padx=2)
        self.x_offset_entry = tk.Entry(offset_frame, width=5)
        self.x_offset_entry.insert(0, "0")  # Default offset value
        self.x_offset_entry.grid(row=0, column=2, padx=2)
        tk.Button(offset_frame, text="+", command=lambda: self.adjust_offset("x", 1)).grid(row=0, column=3, padx=2)

        # Entry for Y offset with closer buttons
        tk.Label(offset_frame, text="Y Offset").grid(row=1, column=0, padx=5)
        tk.Button(offset_frame, text="-", command=lambda: self.adjust_offset("y", -1)).grid(row=1, column=1, padx=2)
        self.y_offset_entry = tk.Entry(offset_frame, width=5)
        self.y_offset_entry.insert(0, "0")  # Default offset value
        self.y_offset_entry.grid(row=1, column=2, padx=2)
        tk.Button(offset_frame, text="+", command=lambda: self.adjust_offset("y", 1)).grid(row=1, column=3, padx=2)
        # Create a frame for tray positions
        tray_frame = tk.LabelFrame(main_frame, text="Tray Position Setting", padx=10, pady=10)
        tray_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Add buttons for setting tray positions and input fields for X, Y positions
        tray_labels = ["circle_large", "circle_medium", "circle_small", 
                    "square_large", "square_medium", "square_small", 
                    "triangle_large", "triangle_medium", "triangle_small"]

        self.tray_labels = {}
        self.tray_entries = {}

        for label in tray_labels:
            # Create a button to manually set the tray mode
            button = tk.Button(tray_frame, text=label.replace('_', ' ').title(), command=lambda l=label: self.set_tray_mode(l))
            button.pack(fill='x', pady=2)

            # Create entry fields for X and Y values
            entry_frame = tk.Frame(tray_frame)
            entry_frame.pack(fill='x', pady=2)

            x_entry = tk.Entry(entry_frame, width=8)
            x_entry.insert(0, "X (mm)")
            x_entry.grid(row=0, column=0, padx=5)

            y_entry = tk.Entry(entry_frame, width=8)
            y_entry.insert(0, "Y (mm)")
            y_entry.grid(row=0, column=1, padx=5)

            # Save references to the labels and entries for later use
            self.tray_labels[label] = tk.Label(tray_frame, text=f"{label.replace('_', ' ').title()}: Not Set", fg="green")
            self.tray_labels[label].pack(fill='x', pady=2)
            self.tray_entries[label] = (x_entry, y_entry)

        # Add a button to update tray positions from the input fields
        self.btn_update_tray_positions = tk.Button(tray_frame, text="Update Tray Positions", command=self.update_tray_positions)
        self.btn_update_tray_positions.pack(fill='x', pady=5)

        # Add a button to lock all tray positions
        self.btn_lock_all_tray_positions = tk.Button(tray_frame, text="Lock All Tray Positions", command=self.lock_all_tray_positions)
        self.btn_lock_all_tray_positions.pack(fill='x', pady=5)

        # Create a labeled frame for the main/top-view image
        topview_frame = tk.LabelFrame(main_frame, text="Top-View/Main Camera", padx=10, pady=10)
        topview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a canvas for showing the main/top-view image
        self.image_canvas = tk.Canvas(topview_frame, width=self.width_px, height=self.height_px, bg="gray")
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.image_canvas.bind("<Button-1>", self.canvas_click)  # Re-bind the canvas click
        self.image_canvas.bind("<Motion>", self.canvas_hover)  # Bind the hover event

        # Label for displaying mouse hover coordinates
        self.hover_label = tk.Label(topview_frame, text="Hover Position: X=0, Y=0")
        self.hover_label.pack(fill='x', pady=5)

        # Create a labeled frame for the AI detected image
        ai_frame = tk.LabelFrame(main_frame, text="AI Detected Image", padx=10, pady=10)
        ai_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a canvas for showing the AI detected image (smaller size)
        self.ai_image_canvas = tk.Canvas(ai_frame, width=self.width_px // 2, height=self.height_px // 2, bg="gray")
        self.ai_image_canvas.pack(fill=tk.BOTH, expand=True)

        # UI is ready
        self.ui_ready = True

        self.root.mainloop()
    def update_delays(self):
            # Update the delay times based on the entries and publish them
            try:
                self.move_to_delay = int(self.move_to_delay_entry.get())
                self.pick_delay = int(self.pick_delay_entry.get())
                self.place_delay = int(self.place_delay_entry.get())

                # Publish the updated delay values
                self.move_to_delay_pub.publish(Int32(data=self.move_to_delay))
                self.pick_delay_pub.publish(Int32(data=self.pick_delay))
                self.place_delay_pub.publish(Int32(data=self.place_delay))

                self.get_logger().info(f"Published delays: Move To={self.move_to_delay}s, Pick={self.pick_delay}s, Place={self.place_delay}s")

            except ValueError:
                self.get_logger().warning("Invalid delay value entered. Please enter valid integers.")

    def adjust_offset(self, axis, value):
        if axis == "x":
            current_value = int(self.x_offset_entry.get())
            new_value = current_value + value
            self.x_offset_entry.delete(0, tk.END)
            self.x_offset_entry.insert(0, str(new_value))

            # Publish the X offset value as Int32
            self.x_offset_pub.publish(Int32(data=new_value))
            self.get_logger().info(f'Published X Offset: {new_value}')

        elif axis == "y":
            current_value = int(self.y_offset_entry.get())
            new_value = current_value + value
            self.y_offset_entry.delete(0, tk.END)
            self.y_offset_entry.insert(0, str(new_value))

            # Publish the Y offset value as Int32
            self.y_offset_pub.publish(Int32(data=new_value))
            self.get_logger().info(f'Published Y Offset: {new_value}')
    def canvas_hover(self, event):
        # Update the label with the current mouse position on the canvas
        cm_x = ((event.x - 300) * self.scale_x) * 10
        cm_y = (((event.y - 600) * self.scale_y) * -1) * 10
        self.hover_label.config(text=f"Hover Position: X={cm_x:.2f}, Y={cm_y:.2f}")

    def lock_all_tray_positions(self):
        # Update all tray positions from input fields and publish their values
        self.update_tray_positions()
        self.tray_setting_mode = None
        self.manual_mode = True  # Switch back to manual mode after locking tray positions
        self.get_logger().info('All tray positions are locked, switching to manual mode.')
        self.update_button_colors(self.btn_manual)

    def lock_topview(self):
        self.topview_locked = True
        self.update_button_colors(self.btn_lock_topview)
        self.get_logger().info('Topview is now locked.')

    def unlock_topview(self):
        self.topview_locked = False
        self.update_button_colors(self.btn_unlock_topview)
        self.get_logger().info('Topview is now unlocked.')

    def enable_topview_mode(self):
        self.topview_mode = True
        self.manual_mode = False
        self.ai_detect_mode = False
        self.display_topview = False
        self.tray_setting_mode = None

        self.update_button_colors(self.btn_topview)
        self.get_logger().info('TOPVIEW mode enabled. Please click on four points to create homography.')

    def publish_ai_detect_OFF(self):
        self.ai_detect_mode = False
        self.topview_mode = False
        self.manual_mode = False
        self.tray_setting_mode = None

        self.update_button_colors(self.btn_ai_detect_OFF)
        self.ai_start_process.publish(String(data="OFF"))
        self.get_logger().info('AI detection state OFF.')
    def publish_ai_detect(self):
        self.ai_detect_mode = True
        self.topview_mode = False
        self.manual_mode = False
        self.tray_setting_mode = None

        self.update_button_colors(self.btn_ai_detect)
        self.ai_start_process.publish(String(data="ON"))
        self.get_logger().info('AI detection state ON.')

    def enable_manual_mode(self):
        self.topview_mode = False
        self.manual_mode = True
        self.ai_detect_mode = False
        self.tray_setting_mode = None
        self.update_button_colors(self.btn_manual)
        self.get_logger().info('Manual mode enabled. Click on TOPVIEW to select move points.')

    def set_home(self):
        self.update_button_colors(self.btn_sethome)
        command_str = "set_home>\\n"
        self.command_pub.publish(String(data=command_str))
        self.get_logger().info('Set home command sent.')

        # Update the command label in the UI
        self.command_label.config(text="Command: Set Home")

    def pick_command(self):
        self.update_button_colors(self.btn_pick)
        command_str = "pick>\\n"
        self.command_pub.publish(String(data=command_str))
        self.get_logger().info('Pick command sent.')

        # Update the command label in the UI
        self.command_label.config(text="Command: Pick")

    def update_button_colors(self, active_button):
        # Reset all button colors to default
        buttons = [
            self.btn_topview,
            self.btn_ai_detect,
            self.btn_ai_detect_OFF,
            self.btn_manual,
            self.btn_sethome,
            self.btn_pick,
            self.btn_place,
            self.btn_reset,
            self.btn_send_command,
            self.btn_update_tray_positions,
            self.btn_lock_all_tray_positions,
        ]

        for btn in buttons:
            btn.config(bg="#f0f0f0")  # Default color for Tkinter buttons

        # Highlight the active button
        active_button.config(bg="lightgreen")

    def update_tray_positions(self):
        for label, entries in self.tray_entries.items():
            x_entry, y_entry = entries
            try:
                # Convert input to float and update tray_positions
                x_value = float(x_entry.get())
                y_value = float(y_entry.get())
                self.tray_positions[label] = (x_value, y_value)
                self.tray_labels[label].config(text=f"{label.replace('_', ' ').title()}: X={x_value:.2f} mm, Y={y_value:.2f} mm")

                # Publish the tray position
                position_str = f"{label}>{x_value:.2f},{y_value:.2f}\n"
                self.tray_position_pub.publish(String(data=position_str))
                self.get_logger().info(f'Updated position for {label}: ({x_value}, {y_value})')

            except ValueError:
                self.get_logger().warning(f"Invalid input for tray position '{label}'. Please enter valid numeric values.")

    def place_command(self):
        self.update_button_colors(self.btn_place)
        command_str = "place>\\n"
        self.command_pub.publish(String(data=command_str))
        self.get_logger().info('Place command sent.')

        # Update the command label in the UI
        self.command_label.config(text="Command: Place")

    def reset_topview(self):
        # Reset the homography matrix and all related variables to go back to the normal view
        self.clicked_points = []
        self.h_matrix = None
        self.display_topview = False
        self.topview_mode = False
        self.manual_mode = True  # Switch to manual mode after resetting top-view
        self.update_button_colors(self.btn_manual)
        self.get_logger().info('Topview reset. Showing main camera view.')

        # Clear all items on the canvas
        self.image_canvas.delete("all")

        # Reset command labels and stored points
        self.command_label.config(text="Command: ")  
        self.point_label.config(text="Stored Point: None")  

    def set_tray_mode(self, tray_label):
        # ตั้งค่าโหมดถาดและแสดงภาพ topview ต่อไป
        self.tray_setting_mode = tray_label
        self.get_logger().info(f'Tray setting mode enabled for: {tray_label.replace("_", " ").title()}')

        # ดึงค่า X และ Y จาก entry fields
        x_entry, y_entry = self.tray_entries[tray_label]
        try:
            # Convert input to float and update tray_positions
            x_value = float(x_entry.get())
            y_value = float(y_entry.get())
            self.tray_positions[tray_label] = (x_value, y_value)
            self.tray_labels[tray_label].config(text=f"{tray_label.replace('_', ' ').title()}: X={x_value:.2f} mm, Y={y_value:.2f} mm")

            # Publish the tray position
            position_str = f"{tray_label}>{x_value:.2f},{y_value:.2f}\n"
            self.tray_position_pub.publish(String(data=position_str))
            self.get_logger().info(f'Updated position for {tray_label}: ({x_value}, {y_value})')

        except ValueError:
            self.get_logger().warning(f"Invalid input for tray position '{tray_label}'. Please enter valid numeric values.")

    def canvas_click(self, event):
        if self.topview_mode:
            # ยังคงทำการคลิกเพื่อสร้าง homography เมื่อ topview_mode เปิดอยู่
            self.clicked_points.append((event.x, event.y))
            # วาดวงกลมสีแดงที่ตำแหน่งที่คลิก
            self.image_canvas.create_oval(
                event.x - 5, event.y - 5,
                event.x + 5, event.y + 5,
                fill='red', outline='red'
            )
            if len(self.clicked_points) == 4:
                self.process_homography()

        elif self.manual_mode:
            # Store the clicked point in manual mode for later use
            self.stored_point = (event.x, event.y)
            self.point_label.config(text=f"Stored Point: X={event.x}, Y={event.y} px")
            self.get_logger().info(f'Stored point for manual mode: ({event.x}, {event.y})')

        if self.tray_setting_mode:
            # ยังคงแสดงภาพ topview และบันทึกตำแหน่งถาดที่คลิก
            cm_x = ((event.x - 300) * self.scale_x) * 10
            cm_y = (((event.y - 600) * self.scale_y) * -1) * 10
            self.tray_positions[self.tray_setting_mode] = (cm_x, cm_y)
            self.tray_labels[self.tray_setting_mode].config(text=f"{self.tray_setting_mode.replace('_', ' ').title()}: X={cm_x:.2f} mm, Y={cm_y:.2f} mm")
            self.get_logger().info(f'Set position for {self.tray_setting_mode}: ({cm_x}, {cm_y})')

            # Publish the tray position
            position_str = f"{self.tray_setting_mode}>{cm_x:.2f},{cm_y:.2f}\n"
            self.tray_position_pub.publish(String(data=position_str))

    def process_homography(self):
        if len(self.clicked_points) == 4:
            if not self.topview_locked:
                pts_src = np.array(self.clicked_points, dtype="float32")
                pts_dst = np.array([
                    [0, 0],
                    [self.width_px - 1, 0],
                    [self.width_px - 1, self.height_px - 1],
                    [0, self.height_px - 1]
                ], dtype="float32")

                self.h_matrix, _ = cv2.findHomography(pts_src, pts_dst)
                self.clicked_points = []
                self.display_topview = True
                self.get_logger().info('Homography matrix created. Displaying top-view perspective.')

                # แสดงภาพ topview ทันทีหลังจากสร้าง homography
                if self.h_matrix is not None:
                    self.update_canvas(self.image_canvas, cv2.warpPerspective(self.latest_frame, self.h_matrix, (self.width_px, self.height_px)))
            else:
                self.get_logger().info('Homography update skipped as topview is locked.')

    def send_stored_command(self):
        if self.stored_point is not None:
            self.z_value = float(self.z_slider.get())
            cm_x = ((self.stored_point[0] - 300) * self.scale_x) * 10
            cm_y = (((self.stored_point[1] - 600) * self.scale_y) * -1) * 10
            
            # Apply the offset before sending the command
            offset_x = float(self.x_offset_entry.get())
            offset_y = float(self.y_offset_entry.get())
            cm_x += offset_x
            cm_y += offset_y

            command_str = f"move_to>{cm_x:.2f},{cm_y:.2f},{self.z_value:.2f},90.0>\\n"
            self.command_pub.publish(String(data=command_str))
            self.get_logger().info(f'Sent command: {command_str}')

            # Update the command label in the UI
            self.command_label.config(text=f"Command: Move to ({cm_x:.2f}, {cm_y:.2f}, {self.z_value:.2f}) mm")
            # Reset the stored point
            self.stored_point = None
            self.point_label.config(text="Stored Point: None")
        else:
            self.get_logger().warning('No point stored for manual command.')
    def update_z_entry(self, event):
        # Update the entry box when slider is moved
        self.z_entry.delete(0, tk.END)
        self.z_entry.insert(0, str(self.z_slider.get()))
        self.publish_z_value()  # Publish the updated Z value
    def update_z_slider(self, event):
        # Update the slider when entry box value changes
        try:
            value = float(self.z_entry.get())
            if 0 <= value <= 300:
                self.z_slider.set(value)
                self.publish_z_value()  # Publish the updated Z value
            else:
                self.get_logger().warning("Z value out of range. Must be between 0 and 300.")
        except ValueError:
            self.get_logger().warning("Invalid Z value entered.")
    def publish_z_value(self):
        """ Publish the current Z value """
        z_value_int = int(self.z_slider.get())
        self.z_value_pub.publish(Int32(data=z_value_int))
        self.get_logger().info(f'Published Z Value: {z_value_int}')
    def image_callback(self, msg):
        self.wait_for_ui()  # Ensure UI is ready before updating canvas
        frame = self.br.imgmsg_to_cv2(msg, 'bgr8')
        self.latest_frame = frame  # เก็บภาพล่าสุด

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.display_topview and hasattr(self, 'h_matrix'):
            top_view = cv2.warpPerspective(frame, self.h_matrix, (self.width_px, self.height_px))
            self.update_canvas(self.image_canvas, top_view)

            # Publish the top-view image
            top_view_msg = self.br.cv2_to_imgmsg(top_view, encoding="bgr8")
            self.top_view_pub.publish(top_view_msg)
        else:
            self.update_canvas(self.image_canvas, frame)

    def ai_detected_callback(self, msg):
        self.wait_for_ui()  # Ensure UI is ready before updating canvas
        # Convert ROS Image message to OpenCV format
        frame = self.br.imgmsg_to_cv2(msg, 'bgr8')
        self.update_canvas(self.ai_image_canvas, frame)

    def update_canvas(self, canvas, frame):
        # Update the canvas size dynamically based on frame dimensions
        height, width = frame.shape[:2]
        canvas.config(width=width, height=height)

        # Convert frame to PIL image, then to ImageTk
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = PILImage.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Update the canvas with the new image
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk  # Keep a reference to avoid garbage collection

def main(args=None):
    rclpy.init(args=args)
    aruco_detector = ArucoDetector()
    rclpy.spin(aruco_detector)

    aruco_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
