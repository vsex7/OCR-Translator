# input_hook_manager.py
import threading
import time
import win32gui
import win32process
import win32api
import pywintypes
import tkinter as tk
from PIL import Image, ImageTk
from logger import log_debug
from resource_handler import get_resource_path
import keyboard
import pyperclip

class InputHookManager:
    def __init__(self, app_logic):
        self.app_logic = app_logic
        self.is_running = False
        self.thread = None
        self.input_class_names = {"Edit", "RichEdit20W", "RichEdit20A"}
        self.icon_window = None
        self.icon_photo = None
        self.last_focused_hwnd = None
        self.original_text_history = []

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.app_logic.root.after(0, self._create_icon_window)
            self.thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.thread.start()
            log_debug("InputHookManager started.")

    def stop(self):
        if self.is_running:
            self.is_running = False
            if self.thread and self.thread.is_alive():
                self.thread.join()
            if self.icon_window:
                self.app_logic.root.after(0, self.icon_window.destroy)
            log_debug("InputHookManager stopped.")

    def _create_icon_window(self):
        self.icon_window = tk.Toplevel(self.app_logic.root)
        self.icon_window.overrideredirect(True)
        self.icon_window.wm_attributes("-topmost", True)
        self.icon_window.withdraw()

        try:
            icon_path = get_resource_path("resources/translate_icon.png")
            self.icon_photo = ImageTk.PhotoImage(Image.open(icon_path))
            icon_label = tk.Label(self.icon_window, image=self.icon_photo, bg='white', cursor="hand2")
            icon_label.pack()
            icon_label.bind("<Button-1>", self.on_icon_click)
            self.icon_window.lift()
        except Exception as e:
            log_debug(f"Failed to load translate icon: {e}")

    def _detection_loop(self):
        while self.is_running:
            try:
                focused_hwnd = win32gui.GetFocus()

                if focused_hwnd and focused_hwnd != self.last_focused_hwnd:
                    self.last_focused_hwnd = focused_hwnd
                    class_name = win32gui.GetClassName(focused_hwnd)
                    if class_name in self.input_class_names:
                        rect = win32gui.GetWindowRect(focused_hwnd)
                        self.app_logic.root.after(0, self._show_icon, rect)
                    else:
                        self.app_logic.root.after(0, self._hide_icon)
                elif not focused_hwnd:
                    self.last_focused_hwnd = None
                    self.app_logic.root.after(0, self._hide_icon)

            except pywintypes.error as e:
                self.last_focused_hwnd = None
                self.app_logic.root.after(0, self._hide_icon)
                log_debug(f"InputHookManager error: {e}")
            except Exception as e:
                self.last_focused_hwnd = None
                self.app_logic.root.after(0, self._hide_icon)
                log_debug(f"An unexpected error occurred in InputHookManager: {e}")

            time.sleep(0.2)

    def _show_icon(self, rect):
        if self.icon_window:
            x = rect[2] - 30
            y = rect[1] + 5
            self.icon_window.geometry(f"+{x}+{y}")
            self.icon_window.deiconify()

    def _hide_icon(self):
        if self.icon_window:
            self.icon_window.withdraw()

    def on_icon_click(self, event):
        self._hide_icon()
        # Use a small delay to ensure the focus is back on the input field
        self.app_logic.root.after(100, self._translate_and_replace)

    def _translate_and_replace(self):
        try:
            # Check if any text is selected
            original_clipboard = pyperclip.paste()
            pyperclip.copy('')
            keyboard.send('ctrl+c')
            time.sleep(0.1)
            selected_text = pyperclip.paste()

            if selected_text:
                text_to_translate = selected_text
            else:
                keyboard.send('ctrl+a')
                keyboard.send('ctrl+c')
                time.sleep(0.1)
                text_to_translate = pyperclip.paste()

            if text_to_translate:
                self.original_text_history.append(text_to_translate)
                if len(self.original_text_history) > 10:
                    self.original_text_history.pop(0)

                translated_text = self.app_logic.translate_text(text_to_translate)
                if translated_text:
                    pyperclip.copy(translated_text)
                    keyboard.send('ctrl+v')

            # Restore original clipboard content if it wasn't empty
            if original_clipboard:
                 pyperclip.copy(original_clipboard)

        except Exception as e:
            log_debug(f"Error during translate and replace: {e}")

    def undo_last_translation(self):
        if self.original_text_history:
            last_text = self.original_text_history.pop()
            pyperclip.copy(last_text)
            keyboard.send('ctrl+v')
            log_debug("Undid last translation.")
