import tkinter as tk
from pynput import mouse
import threading
import win32gui
import win32con
from PIL import Image, ImageTk
import keyboard
import pyperclip
import time

from logger import log_debug

class InputHookManager:
    """
    Manages the input field translation feature, including
    - Detecting clicks on text input fields in various applications.
    - Displaying a floating translation icon next to the detected input field.
    - Handling the click-to-translate and undo functionality.
    """

    def __init__(self, app):
        """
        Initializes the InputHookManager.

        Args:
            app: The main application instance.
        """
        self.app = app
        self.is_running = False
        self.mouse_listener = None
        self.icon_window = None
        self.original_text = None

    def start(self):
        """Starts the input hook listeners (mouse and keyboard)."""
        if self.is_running:
            return
        log_debug("Starting Input Hook Manager.")
        self.is_running = True
        self.mouse_listener = mouse.Listener(on_click=self._on_click)
        self.mouse_listener.start()

    def stop(self):
        """Stops the input hook listeners."""
        if not self.is_running:
            return
        log_debug("Stopping Input Hook Manager.")
        self.is_running = False
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None
        if self.icon_window:
            self.icon_window.destroy()
            self.icon_window = None

    def _on_click(self, x, y, button, pressed):
        """Callback for mouse click events."""
        if not self.is_running or not pressed or button != mouse.Button.left:
            return

        # Hide icon on any click
        if self.icon_window:
            self.icon_window.destroy()
            self.icon_window = None

        def check_and_show_icon():
            try:
                hwnd = win32gui.WindowFromPoint((x, y))
                focused_hwnd = win32gui.GetForegroundWindow()

                if hwnd != focused_hwnd:
                    return

                class_name = win32gui.GetClassName(hwnd)
                style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)

                # More robust check for edit controls
                is_edit_control = "EDIT" in class_name.upper() or \
                                  "RICHEDIT" in class_name.upper() or \
                                  (style & win32con.ES_WANTRETURN) or \
                                  (style & win32con.ES_MULTILINE)

                if is_edit_control:
                    log_debug(f"Input field detected at ({x}, {y}) with class '{class_name}'")
                    self.app.root.after(50, lambda: self._show_translation_icon(x, y))

            except Exception as e:
                log_debug(f"Error in _on_click thread: {e}")

        threading.Thread(target=check_and_show_icon, daemon=True).start()

    def _show_translation_icon(self, x, y):
        """Displays the floating translation icon at the given coordinates."""
        if self.icon_window:
            self.icon_window.destroy()

        self.icon_window = tk.Toplevel(self.app.root)
        self.icon_window.overrideredirect(True)
        self.icon_window.attributes("-topmost", True)
        self.icon_window.attributes("-transparentcolor", "white")
        self.icon_window.config(bg="white")
        self.icon_window.geometry(f"+{x+10}+{y-10}")

        # Base64 encoded icon (simple 'T')
        icon_data = b"iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAPklEQVQ4jWNgGAWjgA8wGNhewEwGKGcwrYDBqQE2pA7kJCQk/A8L5AJ2fP///w8DAwP/D/1/0HwGAJ03J0VqgQctAAAAAElFTkSuQmCC"
        import base64
        icon_image = tk.PhotoImage(data=base64.b64decode(icon_data))

        icon_label = tk.Label(self.icon_window, image=icon_image, bg="white", cursor="hand2")
        icon_label.image = icon_image
        icon_label.pack()
        icon_label.bind("<Button-1>", lambda e: self._translate_and_paste())

        # Auto-hide after 3 seconds
        self.icon_window.after(3000, self.icon_window.destroy)

    def _translate_and_paste(self):
        """Handles the core translation and paste logic."""
        if self.icon_window:
            self.icon_window.destroy()
            self.icon_window = None

        def translate_task():
            try:
                # 1. Copy text
                pyperclip.copy('') # Clear clipboard
                keyboard.send('ctrl+a') # Select all
                time.sleep(0.05)
                keyboard.send('ctrl+c') # Copy
                time.sleep(0.05)
                self.original_text = pyperclip.paste()

                if not self.original_text:
                    log_debug("No text to translate from input field.")
                    return

                # 2. Translate text
                translated_text = self.app.translation_handler.translate_text(self.original_text, is_hover=True)

                if translated_text and translated_text != self.original_text:
                    # 3. Paste translated text
                    pyperclip.copy(translated_text)
                    time.sleep(0.05)
                    keyboard.send('ctrl+v') # Paste
                    log_debug("Pasted translated text.")

                    # 4. Setup undo hotkey
                    self._setup_undo_hotkey()

            except Exception as e:
                log_debug(f"Error in translate_task: {e}")

        threading.Thread(target=translate_task, daemon=True).start()

    def _undo_translation(self):
        """Reverts the text in the input field to its original content."""
        if self.original_text:
            pyperclip.copy(self.original_text)
            time.sleep(0.05)
            keyboard.send('ctrl+v') # Paste
            log_debug("Undo translation: Restored original text.")
            self.original_text = None

    def _setup_undo_hotkey(self):
        """Temporarily registers a hotkey for undoing the translation."""
        try:
            keyboard.add_hotkey('ctrl+z', self._undo_translation, suppress=True)
            log_debug("Undo hotkey (ctrl+z) registered.")

            # Automatically remove the hotkey after 5 seconds
            threading.Timer(5.0, self._remove_undo_hotkey).start()
        except Exception as e:
            log_debug(f"Error setting up undo hotkey: {e}")

    def _remove_undo_hotkey(self):
        """Removes the undo hotkey."""
        try:
            keyboard.remove_hotkey('ctrl+z')
            log_debug("Undo hotkey (ctrl+z) removed.")
        except Exception as e:
            log_debug(f"Error removing undo hotkey: {e}")
