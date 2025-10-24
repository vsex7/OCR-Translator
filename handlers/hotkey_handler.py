from logger import log_debug
import keyboard

class HotkeyHandler:
    """Handles keyboard shortcuts and hotkey registration"""
    
    def __init__(self, app):
        """Initialize with a reference to the main application
        
        Args:
            app: The main GameChangingTranslator application instance
        """
        self.app = app
        self.undo_hotkey = None
    
    def setup_hotkeys(self):
        """Registers all keyboard shortcuts if the keyboard module is available"""
        if self.app.KEYBOARD_AVAILABLE:
            try:
                # Add suppress=True to ensure hotkeys work when app is not in focus
                keyboard.add_hotkey('`', self.toggle_translation_hotkey, suppress=True)
                keyboard.add_hotkey('alt+1', self.toggle_source_visibility_hotkey, suppress=True)
                keyboard.add_hotkey('alt+2', self.toggle_target_visibility_hotkey, suppress=True)
                keyboard.add_hotkey('alt+s', self.save_settings_hotkey, suppress=True)
                keyboard.add_hotkey('alt+c', self.clear_cache_hotkey, suppress=True)
                keyboard.add_hotkey('alt+l', self.clear_debug_log_hotkey, suppress=True)
                log_debug("Keyboard shortcuts registered.")
            except Exception as e_hk: # Use a distinct variable name
                log_debug(f"Error setting up keyboard shortcuts: {e_hk}")
        else:
            log_debug("Keyboard library not available. Hotkeys disabled.")

    def register_undo_hotkey(self):
        if self.app.KEYBOARD_AVAILABLE and self.undo_hotkey is None:
            try:
                self.undo_hotkey = keyboard.add_hotkey('ctrl+z', self.undo_translation_hotkey, suppress=True)
                log_debug("Undo hotkey registered.")
            except Exception as e:
                log_debug(f"Error registering undo hotkey: {e}")

    def unregister_undo_hotkey(self):
        if self.app.KEYBOARD_AVAILABLE and self.undo_hotkey is not None:
            try:
                keyboard.remove_hotkey(self.undo_hotkey)
                self.undo_hotkey = None
                log_debug("Undo hotkey unregistered.")
            except Exception as e:
                log_debug(f"Error unregistering undo hotkey: {e}")
    
    def toggle_translation_hotkey(self):
        """Hotkey callback for toggling translation on/off"""
        try:
            if self.app.root.winfo_exists(): 
                self.app.root.after(0, self.app.toggle_translation)
        except Exception as e:
            log_debug(f"Error in toggle_translation_hotkey: {e}")
            
    def toggle_source_visibility_hotkey(self):
        """Hotkey callback for toggling source overlay visibility"""
        try:
            if self.app.root.winfo_exists(): 
                self.app.root.after(0, self.app.toggle_source_visibility) # Call the wrapper method
        except Exception as e:
            log_debug(f"Error in toggle_source_visibility_hotkey: {e}")
            
    def toggle_target_visibility_hotkey(self):
        """Hotkey callback for toggling target overlay visibility"""
        try:
            if self.app.root.winfo_exists(): 
                self.app.root.after(0, self.app.toggle_target_visibility) # Call the wrapper method
        except Exception as e:
            log_debug(f"Error in toggle_target_visibility_hotkey: {e}")
            
    def save_settings_hotkey(self):
        """Hotkey callback for saving settings"""
        try:
            if self.app.root.winfo_exists(): 
                self.app.root.after(0, self.app.save_settings)
        except Exception as e:
            log_debug(f"Error in save_settings_hotkey: {e}")
            
    def clear_cache_hotkey(self):
        """Hotkey callback for clearing translation cache"""
        try:
            if self.app.root.winfo_exists(): 
                self.app.root.after(0, self.app.clear_cache)
        except Exception as e:
            log_debug(f"Error in clear_cache_hotkey: {e}")
            
    def clear_debug_log_hotkey(self):
        """Hotkey callback for clearing the debug log"""
        try:
            if self.app.root.winfo_exists(): 
                self.app.root.after(0, self.app.clear_debug_log)
        except Exception as e:
            log_debug(f"Error in clear_debug_log_hotkey: {e}")

    def undo_translation_hotkey(self):
        """Hotkey callback for undoing the last input field translation"""
        try:
            if self.app.root.winfo_exists() and self.app.input_hook_manager:
                self.app.root.after(0, self.app.input_hook_manager.undo_translation)
        except Exception as e:
            log_debug(f"Error in undo_translation_hotkey: {e}")
