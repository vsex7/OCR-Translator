# handlers/input_hook_manager.py
import tkinter as tk
from pynput import mouse
from logger import log_debug
import threading
import sys
from PIL import Image, ImageTk
import pyperclip
import time
from pynput.keyboard import Key, Controller

if sys.platform == "win32":
    import win32gui
    import win32con

class InputHookManager:
    def __init__(self, app_logic):
        self.app_logic = app_logic
        self.listener = None
        self.listener_thread = None
        self.floating_icon = None
        self.icon_photo = None
        self.hide_timer = None
        self.original_text = None
        self.keyboard = Controller()

    def start_listener(self):
        if self.listener is None:
            log_debug("Starting input hook listener.")
            # We need to run the listener in a separate thread
            self.listener_thread = threading.Thread(target=self._run_listener, daemon=True)
            self.listener_thread.start()

    def _run_listener(self):
        with mouse.Listener(on_click=self.on_click) as self.listener:
            self.listener.join()

    def stop_listener(self):
        if self.listener is not None:
            log_debug("Stopping input hook listener.")
            self.listener.stop()
            if self.listener_thread is not None:
                self.listener_thread.join()
            self.listener = None
            self.listener_thread = None
            self.hide_floating_icon()

    def on_click(self, x, y, button, pressed):
        if pressed and button == mouse.Button.left:
            if self.is_text_cursor():
                self.app_logic.root.after(0, self.show_floating_icon, x, y)
            else:
                self.app_logic.root.after(0, self.hide_floating_icon)

    def is_text_cursor(self):
        if sys.platform != "win32":
            return False  # Not implemented for other platforms

        try:
            cursor_info = win32gui.GetCursorInfo()
            cursor_handle = cursor_info[1]
            # IDC_IBEAM is the text cursor
            return cursor_handle == win32gui.LoadCursor(0, win32con.IDC_IBEAM)
        except Exception as e:
            log_debug(f"Could not get cursor info: {e}")
            return False

    def show_floating_icon(self, x, y):
        if self.floating_icon is None:
            self.floating_icon = tk.Toplevel(self.app_logic.root)
            self.floating_icon.overrideredirect(True)
            self.floating_icon.wm_attributes("-topmost", True)

            try:
                # Load the icon using Pillow to handle PNG transparency
                image = Image.open("resources/translate_icon.png")
                self.icon_photo = ImageTk.PhotoImage(image)

                icon_label = tk.Label(self.floating_icon, image=self.icon_photo, bd=0, bg='white', cursor="hand2")
                icon_label.pack()
                icon_label.bind("<Button-1>", self.on_icon_click)

                # Make the window transparent where the image is transparent
                self.floating_icon.config(bg='white')
                self.floating_icon.wm_attributes('-transparentcolor', 'white')

            except Exception as e:
                log_debug(f"Could not load translate_icon.png: {e}")
                # Fallback to a simple text label
                fallback_label = tk.Label(self.floating_icon, text="T", padx=5, pady=5, cursor="hand2")
                fallback_label.pack()
                fallback_label.bind("<Button-1>", self.on_icon_click)

        self.floating_icon.geometry(f"+{x+10}+{y+10}")
        self.floating_icon.deiconify()

        # Cancel any existing hide timer
        if self.hide_timer is not None:
            self.app_logic.root.after_cancel(self.hide_timer)

        # Automatically hide the icon after 3 seconds
        self.hide_timer = self.app_logic.root.after(3000, self.hide_floating_icon)

    def hide_floating_icon(self):
        if self.floating_icon is not None:
            self.floating_icon.withdraw()
        if self.hide_timer is not None:
            self.app_logic.root.after_cancel(self.hide_timer)
            self.hide_timer = None

    def on_icon_click(self, event):
        self.hide_floating_icon()

        # Use a separate thread to avoid blocking the UI
        threading.Thread(target=self._translate_and_paste, daemon=True).start()

    def _translate_and_paste(self):
        # 1. Select all and copy text from the input field
        self.keyboard.press(Key.ctrl)
        self.keyboard.press('a')
        self.keyboard.release('a')
        self.keyboard.release(Key.ctrl)

        time.sleep(0.1)

        self.keyboard.press(Key.ctrl)
        self.keyboard.press('c')
        self.keyboard.release('c')
        self.keyboard.release(Key.ctrl)

        time.sleep(0.1) # Give clipboard time to update

        # 2. Get text from clipboard
        self.original_text = pyperclip.paste()
        if not self.original_text:
            log_debug("No text selected to translate.")
            return

        # 3. Translate the text
        translated_text = self.app_logic.translate_text(self.original_text)
        if not translated_text or translated_text == self.original_text:
            log_debug("Translation failed or returned original text.")
            return

        # 4. Paste the translated text
        pyperclip.copy(translated_text)
        time.sleep(0.1)

        self.keyboard.press(Key.ctrl)
        self.keyboard.press('v')
        self.keyboard.release('v')
        self.keyboard.release(Key.ctrl)

        # 5. Register the undo hotkey
        self.app_logic.hotkey_handler.register_undo_hotkey()

    def undo_translation(self):
        if self.original_text:
            log_debug("Undoing translation.")
            pyperclip.copy(self.original_text)
            self.original_text = None # Clear after undo

            time.sleep(0.1)

            self.keyboard.press(Key.ctrl)
            self.keyboard.press('v')
            self.keyboard.release('v')
            self.keyboard.release(Key.ctrl)

            # Unregister the undo hotkey after use
            self.app_logic.hotkey_handler.unregister_undo_hotkey()

    def toggle_listener(self):
        if self.app_logic.input_field_translation_enabled_var.get():
            self.start_listener()
        else:
            self.stop_listener()
