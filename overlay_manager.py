import tkinter as tk
from tkinter import messagebox
import time
from logger import log_debug
from ui_elements import ResizableMovableFrame
from pyside_overlay import get_pyside_manager, is_pyside_available

def _hex_to_rgba_om(hex_color, opacity):
    """Helper to convert #RRGGBB hex and opacity float to rgba(r,g,b,a) string."""
    try:
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {opacity})"
    except Exception:
        # Fallback if color format is invalid
        return hex_color

def _select_screen_area_interactive(app, prompt_text, is_target=False):
    selection_result = None
    try:
        sel_win = tk.Toplevel(app.root)
        sel_win.attributes("-fullscreen", True)
        sel_win.attributes("-alpha", 0.6)
        sel_win.configure(cursor="crosshair")
        
        sel_win.attributes("-topmost", True)
        sel_win.update_idletasks()
        sel_win.wait_visibility()
        sel_win.attributes("-topmost", True)
        sel_win.lift()
        sel_win.focus_force()
        sel_win.grab_set()
        sel_win.update()
        
        tk.Label(sel_win, text=prompt_text + "\n(Esc to cancel)", font=("Arial", 16, "bold"), bg="black", fg="white").place(relx=0.5, rely=0.1, anchor="center")
        canvas = tk.Canvas(sel_win, highlightthickness=0, bg=sel_win['bg'])
        canvas.pack(fill=tk.BOTH, expand=True)
        
        start_coords = {'x': None, 'y': None}
        rect_item = None

        def on_press(e):
            nonlocal start_coords, rect_item
            start_coords['x'], start_coords['y'] = e.x, e.y
            if rect_item:
                canvas.delete(rect_item)
            
            if is_target:
                rect_item = canvas.create_rectangle(
                    e.x, e.y, e.x, e.y, 
                    outline=app.target_colour_var.get(), 
                    width=3,
                    fill=app.target_colour_var.get(),
                    stipple="gray25"
                )
            else:
                rect_item = canvas.create_rectangle(
                    e.x, e.y, e.x, e.y, 
                    outline=app.source_colour_var.get(), 
                    width=3,
                    fill=app.source_colour_var.get(),
                    stipple="gray50"
                )

        def on_drag(e):
            nonlocal rect_item
            if start_coords['x'] is not None and rect_item:
                canvas.coords(rect_item, start_coords['x'], start_coords['y'], e.x, e.y)
                canvas.update_idletasks()

        def on_release(e):
            nonlocal selection_result, start_coords, rect_item
            if start_coords['x'] is None:
                sel_win.destroy()
                return
            x1,y1,x2,y2 = min(start_coords['x'],e.x), min(start_coords['y'],e.y), max(start_coords['x'],e.x), max(start_coords['y'],e.y)
            if abs(x2-x1) < 10 or abs(y2-y1) < 10:
                messagebox.showwarning("Selection Small", "Area too small. Drag again or Esc to cancel.", parent=sel_win)
                if rect_item:
                    canvas.delete(rect_item)
                rect_item = None
                start_coords['x'] = None
                return
            selection_result = [x1,y1,x2,y2]
            sel_win.destroy()

        def on_escape(e):
            nonlocal selection_result
            selection_result=None
            sel_win.destroy()

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        sel_win.bind("<Escape>", on_escape)
        sel_win.wait_window()
    except tk.TclError as e_sel:
        log_debug(f"OverlayManager: Screen selection error: {e_sel}")
        selection_result = None
    finally:
        if 'sel_win' in locals() and sel_win.winfo_exists():
            sel_win.grab_release()
    return selection_result

def select_source_area_om(app):
    app.root.lift()
    app.root.focus_force()
    app.root.update()
    time.sleep(0.1)
    
    app.root.withdraw()
    time.sleep(0.3)
    selected = _select_screen_area_interactive(app, "Select OCR Source Area (Click & Drag)", is_target=False)
    app.root.deiconify()
    app.root.lift()
    app.root.focus_force()
    if selected:
        # Clear existing overlays when a new manual selection is made
        for overlay in app.source_overlays.values():
            if overlay and overlay.winfo_exists():
                overlay.destroy()
        app.source_overlays.clear()
        app.source_areas.clear()

        # A manually selected area doesn't have a real hwnd, so we use 0 as a placeholder
        hwnd = 0
        app.source_areas[hwnd] = selected
        log_debug(f"OverlayManager: Source area selected: {app.source_areas[hwnd]}")
        messagebox.showinfo(
            app.ui_lang.get_label("dialog_area_selected_title", "Area Selected"), 
            f"{app.ui_lang.get_label('dialog_source_area_set_message', 'Source area set to:')}\n{app.source_areas[hwnd]}",
            parent=app.root
        )
        create_source_overlay_om(app, hwnd, app.source_areas[hwnd])
        
        if app.source_overlays.get(hwnd) and app.source_overlays[hwnd].winfo_exists() and app.source_overlays[hwnd].winfo_viewable():
            app.source_overlays[hwnd].hide()
            app.config['Settings']['source_area_visible'] = 'False'
            log_debug("OverlayManager: Source overlay hidden after initial selection")

def select_target_area_om(app):
    app.root.lift()
    app.root.focus_force()
    app.root.update()
    time.sleep(0.1)
    
    app.root.withdraw()
    time.sleep(0.3)
    selected = _select_screen_area_interactive(app, "Select Translation Display Area (Click & Drag)", is_target=True)
    app.root.deiconify()
    app.root.lift()
    app.root.focus_force()
    if selected:
        app.target_area = selected
        log_debug(f"OverlayManager: Target area selected: {app.target_area}")
        messagebox.showinfo(
            app.ui_lang.get_label("dialog_area_selected_title", "Area Selected"), 
            f"{app.ui_lang.get_label('dialog_target_area_set_message', 'Target display area set to:')}\n{app.target_area}", 
            parent=app.root
        )
        create_target_overlay_om(app, hwnd=None, area=app.target_area, skip_preservation=True)
        
        if app.target_overlay and app.target_overlay.winfo_exists() and app.target_overlay.winfo_viewable():
            app.target_overlay.hide()
            app.config['Settings']['target_area_visible'] = 'False'
            log_debug("OverlayManager: Target overlay hidden after initial selection")

def create_source_overlay_om(app, hwnd, area):
    if not area or len(area) != 4:
        log_debug(f"OverlayManager: Invalid area provided for overlay creation: {area}")
        return

    if hwnd in app.source_overlays and app.source_overlays[hwnd].winfo_exists():
        try:
            app.source_overlays[hwnd].destroy()
        except tk.TclError:
            log_debug(f"OverlayManager: Error destroying existing source overlay for hwnd {hwnd} (already gone?).")

    try:
        overlay = ResizableMovableFrame(app.root, area, bg_color=app.source_colour_var.get(), title="")
        overlay.attributes("-alpha", 0.7)
        overlay.update_color(app.source_colour_var.get())

        should_be_visible = app.config['Settings'].getboolean('source_area_visible', fallback=False)
        if not should_be_visible and overlay.winfo_viewable():
            overlay.hide()
        elif should_be_visible and not overlay.winfo_viewable():
            overlay.show()
        else:
            overlay.hide()
            app.config['Settings']['source_area_visible'] = 'False'
        
        app.source_overlays[hwnd] = overlay
        log_debug(f"OverlayManager: Created source overlay for hwnd {hwnd}. Visible: {overlay.winfo_viewable()}")
    except Exception as e_cso:
        log_debug(f"OverlayManager: Error creating source overlay for hwnd {hwnd}: {e_cso}")

def _preserve_overlay_position(app, hwnd=None):
    overlay = app.target_overlays.get(hwnd) if hwnd else app.target_overlay
    if not overlay:
        return
        
    try:
        current_geometry = None
        
        if hasattr(overlay, 'get_geometry'):
            current_geometry = overlay.get_geometry()
            log_debug(f"OverlayManager: Extracted PySide overlay geometry: {current_geometry}")
            
        elif hasattr(overlay, 'winfo_x') and hasattr(overlay, 'winfo_exists'):
            if overlay.winfo_exists():
                x = overlay.winfo_x()
                y = overlay.winfo_y()
                w = overlay.winfo_width()
                h = overlay.winfo_height()
                current_geometry = [x, y, x + w, y + h]
                log_debug(f"OverlayManager: Extracted tkinter overlay geometry: {current_geometry}")
        
        if current_geometry and len(current_geometry) == 4:
            if hwnd is None:
                app.target_area = current_geometry
            
            app.config['Settings'][f'target_area_x1_{hwnd}' if hwnd else 'target_area_x1'] = str(current_geometry[0])
            app.config['Settings'][f'target_area_y1_{hwnd}' if hwnd else 'target_area_y1'] = str(current_geometry[1])
            app.config['Settings'][f'target_area_x2_{hwnd}' if hwnd else 'target_area_x2'] = str(current_geometry[2])
            app.config['Settings'][f'target_area_y2_{hwnd}' if hwnd else 'target_area_y2'] = str(current_geometry[3])
            
            log_debug(f"OverlayManager: Preserved overlay position for hwnd {hwnd}: {current_geometry}")
        else:
            log_debug("OverlayManager: Could not extract valid overlay geometry")
            
    except Exception as e:
        log_debug(f"OverlayManager: Error preserving overlay position: {e}")

def create_target_overlay_om(app, hwnd=None, area=None, skip_preservation=False):
    if area is None:
        if not app.target_area or len(app.target_area) != 4:
            try:
                x1 = int(app.config['Settings'].get('target_area_x1', '200'))
                y1 = int(app.config['Settings'].get('target_area_y1', '200'))
                x2 = int(app.config['Settings'].get('target_area_x2', '500'))
                y2 = int(app.config['Settings'].get('target_area_y2', '400'))
                app.target_area = [x1, y1, x2, y2]
                area = app.target_area
            except (ValueError, KeyError) as e:
                log_debug(f"OverlayManager: Could not load target area from config for overlay creation: {e}")
                return
        else:
            area = app.target_area

    if hwnd is None:
        if app.target_overlay and hasattr(app.target_overlay, 'winfo_exists'):
            try:
                if app.target_overlay.winfo_exists():
                    if not skip_preservation:
                        _preserve_overlay_position(app)
                        log_debug("OverlayManager: Preserved overlay position before recreation")
                    else:
                        log_debug("OverlayManager: Skipped overlay position preservation (user area selection)")
                    app.target_overlay.destroy()
            except tk.TclError:
                log_debug("OverlayManager: Error destroying existing tkinter target overlay (already gone?).")
        elif app.target_overlay and hasattr(app.target_overlay, 'close'):
            try:
                if not skip_preservation:
                    _preserve_overlay_position(app)
                    log_debug("OverlayManager: Preserved overlay position before recreation")
                else:
                    log_debug("OverlayManager: Skipped overlay position preservation (user area selection)")
                app.target_overlay.close()
            except:
                log_debug("OverlayManager: Error destroying existing PySide target overlay (already gone?).")
        app.target_overlay = None
        app.translation_text = None
    else:
        if hwnd in app.target_overlays and app.target_overlays[hwnd].winfo_exists():
            try:
                app.target_overlays[hwnd].destroy()
            except tk.TclError:
                log_debug(f"OverlayManager: Error destroying existing target overlay for hwnd {hwnd} (already gone?).")
        app.target_overlays[hwnd] = None
        app.translation_texts[hwnd] = None

    try:
        target_color = app.target_colour_var.get()

        try:
            font_size = int(app.target_font_size_var.get())
        except Exception:
            font_size = int(app.config['Settings'].get('default_font_size', '14'))

        pad_x = int(app.config['Settings'].get('target_text_pad_x', '5'))
        pad_y = int(app.config['Settings'].get('target_text_pad_y', '5'))
        top_bar_height = int(app.config['Settings'].get('target_top_bar_height', '10'))
        border_px = int(app.config['Settings'].get('target_border_px', '0'))

        # Background opacity - prefer app variable over config
        try:
            opacity = app.target_opacity_var.get()
        except (AttributeError, tk.TclError):
            opacity = float(app.config['Settings'].get('target_opacity', '0.15'))
        
        # Text opacity - prefer app variable over config  
        try:
            text_opacity = app.target_text_opacity_var.get()
        except (AttributeError, tk.TclError):
            text_opacity = float(app.config['Settings'].get('target_text_opacity', '1.0'))

        target_overlay = None
        translation_text = None
        if is_pyside_available():
            log_debug("OverlayManager: PySide6 available, attempting to create PySide overlay")
            pyside_manager = get_pyside_manager()

<<<<<<< HEAD
            app.target_overlay = pyside_manager.create_overlay(
                app,
                app.target_area,
=======
            app.target_overlay = pyside_manager.create_overlay(
                app,
                app.target_area,
>>>>>>> origin/feat-overlay-display-modes
                target_color,
                title="Translation",
                top_bar_height=top_bar_height,
                text_padding=(pad_x, pad_y),
                font_size=font_size,
                font_family="Arial",
                border_px=border_px,
                opacity=opacity
            )

            if target_overlay:
                translation_text = target_overlay.text_widget
                target_overlay.update_color(target_color)

                # MODIFIED: Apply text color with its own opacity setting
                text_hex_color = app.target_text_colour_var.get()
                text_rgba_color = _hex_to_rgba_om(text_hex_color, text_opacity)
                target_overlay.update_text_color(text_rgba_color)
                log_debug(f"OverlayManager: Set PySide text color to {text_rgba_color}")

                log_debug("OverlayManager: PySide target overlay created successfully")
                if hasattr(translation_text, 'winfo_exists'):
                    log_debug(f"OverlayManager: Target overlay exists: {target_overlay.winfo_exists()}")
                    try:
                        log_debug(f"OverlayManager: Translation text exists: {translation_text.winfo_exists()}")
                    except Exception:
                        pass
            else:
                log_debug("OverlayManager: Failed to create PySide overlay, falling back to tkinter")
        else:
            log_debug("OverlayManager: PySide6 not available, will use tkinter overlay")

        if not target_overlay:
            log_debug("OverlayManager: Using tkinter target overlay as fallback")
            try:
                target_overlay = ResizableMovableFrame(app.root, area, bg_color=target_color, title="Translation")
                target_overlay.attributes("-alpha", opacity)
                target_overlay.update_color(target_color)

                target_lang_code = None
                try:
                    target_lang_name = app.target_lang_var.get()
                    if hasattr(app, 'language_manager') and app.language_manager:
                        current_model = app.translation_model_var.get()
                        if 'gemini' in current_model.lower():
                            target_lang_code = app.language_manager.get_code_from_name(target_lang_name, "gemini_api", "target")
                        elif current_model == "Google Translate API":
                            target_lang_code = app.language_manager.get_code_from_name(target_lang_name, "google_api", "target")
                        elif current_model == "DeepL API":
                            target_lang_code = app.language_manager.get_code_from_name(target_lang_name, "deepl_api", "target")
                        is_rtl = app.language_manager.is_rtl_language(target_lang_code) if target_lang_code else False
                        log_debug(f"OverlayManager: Target language '{target_lang_name}' (code: {target_lang_code}) RTL: {is_rtl}")
                    else:
                        is_rtl = False
                except Exception as e:
                    log_debug(f"OverlayManager: Error determining RTL status: {e}")
                    is_rtl = False

                text_justify = tk.RIGHT if is_rtl else tk.LEFT

                translation_text = tk.Text(
                    target_overlay.content_frame,
                    wrap=tk.WORD,
                    bg=target_color,
                    fg=app.target_text_colour_var.get(),
                    font=("Arial", font_size),
                    bd=border_px,
                    relief="flat",
                    padx=pad_x,
                    pady=pad_y,
                    state=tk.DISABLED
                )

                if is_rtl:
                    translation_text.tag_configure("rtl", justify=tk.RIGHT)
                    translation_text.is_rtl = True
                else:
                    translation_text.tag_configure("ltr", justify=tk.LEFT)
                    translation_text.is_rtl = False

        def show_context_menu(event, hwnd=None):
            context_menu = tk.Menu(target_overlay, tearoff=0)

            text_widget = app.translation_texts.get(hwnd) if hwnd is not None else app.translation_text
            current_text = ""
            if text_widget:
                if isinstance(text_widget, tk.Text):
                    current_text = text_widget.get("1.0", tk.END).strip()
                elif hasattr(text_widget, 'toPlainText'):
                    current_text = text_widget.toPlainText().strip()

            def copy_text():
                app.copy_translation_to_clipboard(hwnd)

            def read_aloud():
                app.read_translation_aloud(hwnd)

            if current_text:
                context_menu.add_command(label=app.ui_lang.get_label("context_menu_copy", "Copy"), command=copy_text)
                context_menu.add_command(label=app.ui_lang.get_label("context_menu_read_aloud", "Read Aloud"), command=read_aloud)
            else:
                context_menu.add_command(label=app.ui_lang.get_label("context_menu_no_text", "No text to interact with"), state="disabled")

            try:
                context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                context_menu.grab_release()

        translation_text.bind("<Button-3>", lambda event, h=hwnd: show_context_menu(event, h))
        target_overlay.content_frame.bind("<Button-3>", lambda event, h=hwnd: show_context_menu(event, h))

                translation_text.bind("<Configure>", app.display_manager.on_translation_widget_resize)
                translation_text.pack(fill=tk.BOTH, expand=True)

                log_debug("OverlayManager: tkinter target overlay created successfully (text opacity not supported)")
                log_debug(f"OverlayManager: Target overlay exists: {target_overlay.winfo_exists()}")
                log_debug(f"OverlayManager: Translation text exists: {translation_text.winfo_exists()}")

            except Exception as e_tkinter:
                log_debug(f"OverlayManager: Error creating tkinter target overlay: {e_tkinter}")
                raise e_tkinter

        if hwnd is None:
            app.target_overlay = target_overlay
            app.translation_text = translation_text
        else:
            app.target_overlays[hwnd] = target_overlay
            app.translation_texts[hwnd] = translation_text

        should_be_visible = app.config['Settings'].getboolean('target_area_visible', fallback=False)
        if not should_be_visible and target_overlay.winfo_viewable():
            target_overlay.hide()
        elif should_be_visible and not target_overlay.winfo_viewable():
            target_overlay.show()
        else:
            target_overlay.hide()
            app.config['Settings']['target_area_visible'] = 'False'

        overlay_type = "PySide" if hasattr(translation_text, 'set_rtl_text') else "tkinter"
        log_debug(f"OverlayManager: {overlay_type} target overlay created. Visible: {target_overlay.winfo_viewable()}")

    except Exception as e_cto:
        log_debug(f"OverlayManager: Error creating target overlay: {e_cto}")
        app.target_overlay = None
        app.translation_text = None


def toggle_source_visibility_om(app):
    if app.source_overlays:
        for overlay in app.source_overlays.values():
            if overlay and overlay.winfo_exists():
                overlay.toggle_visibility()
        # We assume all overlays have the same visibility state, so we check the first one.
        first_overlay = next(iter(app.source_overlays.values()), None)
        if first_overlay:
            action = "hidden" if not first_overlay.winfo_viewable() else "shown"
            log_debug(f"OverlayManager: Toggled source overlays {action} by user.")
    else:
        messagebox.showwarning("Warning", "Source area overlay window does not exist.\nPlease select the source area first.", parent=app.root)
        log_debug("OverlayManager: Toggle source visibility failed: Overlay does not exist.")

def toggle_target_visibility_om(app):
    toggled = False
    if app.target_overlay and app.target_overlay.winfo_exists():
        if hasattr(app.target_overlay, 'update_color'):
            app.target_overlay.update_color(app.target_colour_var.get())
        app.target_overlay.toggle_visibility()
        toggled = True

    for overlay in app.target_overlays.values():
        if overlay and overlay.winfo_exists():
            if hasattr(overlay, 'update_color'):
                overlay.update_color(app.target_colour_var.get())
            overlay.toggle_visibility()
            toggled = True

    if toggled:
        # We assume all overlays have the same visibility state, so we check the first one that exists.
        final_overlay = app.target_overlay or next(iter(app.target_overlays.values()), None)
        if final_overlay:
            action = "hidden" if not final_overlay.winfo_viewable() else "shown"
            log_debug(f"OverlayManager: Toggled target overlays {action} by user.")
    else:
        messagebox.showwarning("Warning", "Target area overlay window does not exist.\nPlease select the target area first.", parent=app.root)
        log_debug("OverlayManager: Toggle target visibility failed: Overlay does not exist.")

def load_areas_from_config_om(app):
    """Loads the source and target areas from saved config and creates overlays via OM functions."""
    try:
        # Load single source area for fallback
        source_x1 = int(app.config['Settings'].get('source_area_x1', '100'))
        source_y1 = int(app.config['Settings'].get('source_area_y1', '100'))
        source_x2 = int(app.config['Settings'].get('source_area_x2', '400'))
        source_y2 = int(app.config['Settings'].get('source_area_y2', '300'))
        source_area = [source_x1, source_y1, source_x2, source_y2]

        # A manually selected area doesn't have a real hwnd, so we use 0 as a placeholder
        hwnd = 0
        app.source_areas[hwnd] = source_area
        create_source_overlay_om(app, hwnd, source_area)
        log_debug(f"OverlayManager: Created source overlay from config: {source_area}")

        # Load single target area for fallback
        target_x1 = int(app.config['Settings'].get('target_area_x1', '200'))
        target_y1 = int(app.config['Settings'].get('target_area_y1', '200'))
        target_x2 = int(app.config['Settings'].get('target_area_x2', '500'))
        target_y2 = int(app.config['Settings'].get('target_area_y2', '400'))
        app.target_area = [target_x1, target_y1, target_x2, target_y2]

        if not app.target_overlay or not app.target_overlay.winfo_exists():
            create_target_overlay_om(app, area=app.target_area)
            log_debug(f"OverlayManager: Created target overlay from config: {app.target_area}")

        # Load multi-window target areas
        for key in app.config['Settings']:
            if key.startswith('target_area_x1_'):
                hwnd = int(key.split('_')[-1])
                x1 = int(app.config['Settings'][f'target_area_x1_{hwnd}'])
                y1 = int(app.config['Settings'][f'target_area_y1_{hwnd}'])
                x2 = int(app.config['Settings'][f'target_area_x2_{hwnd}'])
                y2 = int(app.config['Settings'][f'target_area_y2_{hwnd}'])
                area = [x1, y1, x2, y2]
                create_target_overlay_om(app, hwnd=hwnd, area=area)
                log_debug(f"OverlayManager: Created target overlay for hwnd {hwnd} from config: {area}")

    except (ValueError, KeyError) as e:
        log_debug(f"OverlayManager: Could not load areas from config: {e}")