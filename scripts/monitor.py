"""
NitroGen Monitoring Dashboard
Run this on a second monitor to see:
- Live frame preview (what the AI sees)
- Controller state visualization (joysticks + buttons)
- Action history log
- Performance metrics (FPS, latency)

Usage: python scripts/monitor.py --port 5556
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import time
import json
from datetime import datetime
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import zmq
import pickle
import argparse


class ControllerVisualizer(tk.Canvas):
    """Visualizes controller state (joysticks and buttons)"""

    def __init__(self, parent, width=300, height=200):
        super().__init__(parent, width=width, height=height, bg='#2b2b2b', highlightthickness=0)
        self.width = width
        self.height = height

        # Button names for display
        self.button_names = [
            'A', 'B', 'X', 'Y', 'LB', 'RB', 'LT', 'RT',
            'Back', 'Start', 'L3', 'R3',
            'Up', 'Down', 'Left', 'Right', 'Guide'
        ]

        self.draw_empty()

    def draw_empty(self):
        """Draw empty controller state"""
        self.delete('all')

        # Left joystick background
        self.create_oval(20, 20, 120, 120, outline='#555', width=2)
        self.create_line(70, 20, 70, 120, fill='#444')
        self.create_line(20, 70, 120, 70, fill='#444')
        self.create_text(70, 135, text='Left Stick', fill='#888', font=('Consolas', 9))

        # Right joystick background
        self.create_oval(160, 20, 260, 120, outline='#555', width=2)
        self.create_line(210, 20, 210, 120, fill='#444')
        self.create_line(160, 70, 260, 70, fill='#444')
        self.create_text(210, 135, text='Right Stick', fill='#888', font=('Consolas', 9))

        # Buttons area
        self.create_text(150, 165, text='Buttons: None pressed', fill='#888', font=('Consolas', 9))

    def update_state(self, left_x, left_y, right_x, right_y, buttons):
        """Update controller visualization with current state"""
        self.delete('all')

        # Left joystick
        self.create_oval(20, 20, 120, 120, outline='#555', width=2)
        self.create_line(70, 20, 70, 120, fill='#444')
        self.create_line(20, 70, 120, 70, fill='#444')
        # Joystick position (convert -1..1 to pixel coords)
        lx = 70 + int(left_x * 45)
        ly = 70 + int(left_y * 45)
        self.create_oval(lx-8, ly-8, lx+8, ly+8, fill='#4CAF50', outline='#8BC34A', width=2)
        self.create_text(70, 135, text=f'L: ({left_x:.2f}, {left_y:.2f})', fill='#aaa', font=('Consolas', 8))

        # Right joystick
        self.create_oval(160, 20, 260, 120, outline='#555', width=2)
        self.create_line(210, 20, 210, 120, fill='#444')
        self.create_line(160, 70, 260, 70, fill='#444')
        rx = 210 + int(right_x * 45)
        ry = 70 + int(right_y * 45)
        self.create_oval(rx-8, ry-8, rx+8, ry+8, fill='#2196F3', outline='#64B5F6', width=2)
        self.create_text(210, 135, text=f'R: ({right_x:.2f}, {right_y:.2f})', fill='#aaa', font=('Consolas', 8))

        # Pressed buttons
        pressed = [self.button_names[i] for i, b in enumerate(buttons[:17]) if b > 0.5]
        btn_text = ', '.join(pressed) if pressed else 'None'
        self.create_text(150, 165, text=f'Buttons: {btn_text}', fill='#FFB74D' if pressed else '#888',
                        font=('Consolas', 9, 'bold' if pressed else 'normal'))


class FrameViewer(tk.Label):
    """Displays the current frame being processed"""

    def __init__(self, parent, width=256, height=256):
        super().__init__(parent, bg='#1e1e1e')
        self.width = width
        self.height = height
        self.current_image = None
        self.show_placeholder()

    def show_placeholder(self):
        """Show placeholder when no frame is available"""
        img = Image.new('RGB', (self.width, self.height), '#333')
        draw = ImageDraw.Draw(img)
        draw.text((self.width//2 - 50, self.height//2 - 10), 'Waiting for frame...', fill='#666')
        self.update_image(img)

    def update_image(self, pil_image):
        """Update displayed image"""
        resized = pil_image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        self.current_image = ImageTk.PhotoImage(resized)
        self.configure(image=self.current_image)


class MonitorDashboard:
    """Main dashboard window"""

    def __init__(self, port=5556):
        self.port = port
        self.running = True
        self.message_queue = queue.Queue()

        # Stats
        self.frame_count = 0
        self.start_time = time.time()
        self.last_update = time.time()
        self.fps = 0.0
        self.latency_ms = 0.0

        self.setup_ui()
        self.setup_subscriber()

    def setup_ui(self):
        """Create the UI"""
        self.root = tk.Tk()
        self.root.title('NitroGen Monitor')
        self.root.geometry('700x700')
        self.root.configure(bg='#1e1e1e')

        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TLabel', background='#1e1e1e', foreground='#e0e0e0')
        style.configure('Header.TLabel', font=('Consolas', 12, 'bold'))

        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(main_frame, text='NitroGen AI Monitor', style='Header.TLabel')
        title.pack(pady=(0, 10))

        # Top section: Frame + Controller side by side
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)

        # Frame viewer (left)
        frame_section = ttk.Frame(top_frame)
        frame_section.pack(side=tk.LEFT, padx=5)
        ttk.Label(frame_section, text='AI Vision (256x256)').pack()
        self.frame_viewer = FrameViewer(frame_section, 256, 256)
        self.frame_viewer.pack(pady=5)

        # Controller visualizer (right)
        ctrl_section = ttk.Frame(top_frame)
        ctrl_section.pack(side=tk.LEFT, padx=20)
        ttk.Label(ctrl_section, text='Controller State').pack()
        self.controller_viz = ControllerVisualizer(ctrl_section, 300, 200)
        self.controller_viz.pack(pady=5)

        # Stats section
        stats_frame = ttk.Frame(main_frame)
        stats_frame.pack(fill=tk.X, pady=10)

        self.stats_label = ttk.Label(stats_frame, text='FPS: 0.0 | Latency: 0ms | Frames: 0',
                                     font=('Consolas', 10))
        self.stats_label.pack()

        self.status_label = ttk.Label(stats_frame, text='Status: Connecting...',
                                      font=('Consolas', 10), foreground='#FFA726')
        self.status_label.pack()

        # Log section
        log_frame = ttk.Frame(main_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        ttk.Label(log_frame, text='Action Log').pack()

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, bg='#2b2b2b', fg='#e0e0e0',
                                                   font=('Consolas', 9), insertbackground='white')
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state='disabled')

        # Close handler
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

    def setup_subscriber(self):
        """Setup ZeroMQ subscriber in background thread"""
        self.zmq_thread = threading.Thread(target=self.zmq_subscriber_loop, daemon=True)
        self.zmq_thread.start()

    def zmq_subscriber_loop(self):
        """Background thread that receives monitoring data"""
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt_string(zmq.SUBSCRIBE, '')
        socket.setsockopt(zmq.RCVTIMEO, 1000)

        try:
            socket.connect(f'tcp://localhost:{self.port}')
            self.message_queue.put({'type': 'status', 'message': f'Connected to port {self.port}'})

            while self.running:
                try:
                    msg = socket.recv()
                    data = pickle.loads(msg)
                    self.message_queue.put(data)
                except zmq.error.Again:
                    pass  # Timeout, continue
                except Exception as e:
                    self.message_queue.put({'type': 'error', 'message': str(e)})
        finally:
            socket.close()
            context.term()

    def process_messages(self):
        """Process messages from the queue (runs in main thread)"""
        try:
            while True:
                msg = self.message_queue.get_nowait()
                self.handle_message(msg)
        except queue.Empty:
            pass

        # Update stats display
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        self.stats_label.configure(
            text=f'FPS: {self.fps:.1f} | Latency: {self.latency_ms:.0f}ms | Frames: {self.frame_count}'
        )

        if self.running:
            self.root.after(50, self.process_messages)

    def handle_message(self, msg):
        """Handle a single message"""
        msg_type = msg.get('type', 'unknown')

        if msg_type == 'status':
            self.status_label.configure(text=f"Status: {msg['message']}", foreground='#4CAF50')

        elif msg_type == 'error':
            self.status_label.configure(text=f"Error: {msg['message']}", foreground='#f44336')

        elif msg_type == 'frame':
            # Update frame display
            if 'image' in msg:
                try:
                    img_array = np.array(msg['image'], dtype=np.uint8)
                    pil_img = Image.fromarray(img_array)
                    self.frame_viewer.update_image(pil_img)
                except Exception as e:
                    self.log(f'Frame error: {e}')

            self.frame_count += 1
            self.latency_ms = msg.get('latency_ms', 0)

        elif msg_type == 'action':
            # Update controller visualization
            action = msg.get('action', {})
            left_x = action.get('left_x', 0)
            left_y = action.get('left_y', 0)
            right_x = action.get('right_x', 0)
            right_y = action.get('right_y', 0)
            buttons = action.get('buttons', [0] * 17)

            self.controller_viz.update_state(left_x, left_y, right_x, right_y, buttons)

            # Log significant actions
            if any(b > 0.5 for b in buttons) or abs(left_x) > 0.3 or abs(left_y) > 0.3:
                self.log(f"Action: L({left_x:.2f},{left_y:.2f}) R({right_x:.2f},{right_y:.2f})")

        elif msg_type == 'log':
            self.log(msg.get('message', ''))

    def log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, f'[{timestamp}] {message}\n')
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')

        # Keep log size manageable
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 500:
            self.log_text.configure(state='normal')
            self.log_text.delete('1.0', '100.0')
            self.log_text.configure(state='disabled')

    def on_close(self):
        """Handle window close"""
        self.running = False
        self.root.destroy()

    def run(self):
        """Start the dashboard"""
        self.log('Monitor dashboard started')
        self.log(f'Listening for data on port {self.port}')
        self.log('Run play.py with --monitor flag to enable monitoring')
        self.process_messages()
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description='NitroGen Monitoring Dashboard')
    parser.add_argument('--port', type=int, default=5556, help='Port to receive monitoring data')
    args = parser.parse_args()

    dashboard = MonitorDashboard(port=args.port)
    dashboard.run()


if __name__ == '__main__':
    main()
