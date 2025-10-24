#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySide6 Translation Overlay - updated to match tkinter fallback visuals.

Features / fixes included:
- Visual parity with tkinter: top bar height, text padding, font family/size,
  border thickness, opacity.
- Removed forced CSS line-height so Qt uses native font metrics.
- VisualTopBar handles window moving and keeps Qt.SizeAllCursor visible.
- nativeEvent WM_NCHITTEST no longer returns HTCAPTION (prevents Windows from
  overriding the cursor). Resizing hit-tests are still returned to allow native resize.
- create_overlay accepts forwarded visual kwargs (top_bar_height, text_padding, font_size, etc.)
- MODIFIED: Split opacity for background (85%) and text (100%) by using a
  semi-transparent central widget on a fully transparent window.
"""

import sys
import os
import re

# Attempt to import logger; if not present, provide a simple fallback
try:
    from logger import log_debug
except Exception:
    def log_debug(*args, **kwargs):
        try:
            print("DEBUG:", *args, **kwargs)
        except Exception:
            pass

# Try to import PySide6; if not available, mark accordingly.
try:
    from PySide6.QtWidgets import QApplication, QTextEdit, QVBoxLayout, QMainWindow, QWidget
    from PySide6.QtCore import Qt, QRect, QPoint
    from PySide6.QtGui import QFont, QTextCursor, QTextBlockFormat
    PYSIDE6_AVAILABLE = True
except Exception:
    PYSIDE6_AVAILABLE = False

# Windows native interaction (only used if running on Win32 and PySide available)
if sys.platform == "win32" and PYSIDE6_AVAILABLE:
    import ctypes
    from ctypes import wintypes

# Optional Arabic reshaper / bidi support (best-effort)
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    RESHAPER_AVAILABLE = True
except Exception:
    RESHAPER_AVAILABLE = False


# -----------------------
# PySide6-backed classes
# -----------------------
if PYSIDE6_AVAILABLE:

    class RTLTextDisplay(QTextEdit):
        """RTL-capable text display for translation overlay (QTextEdit wrapper)."""

        def __init__(self, parent=None):
            super().__init__(parent)
            # Initialize color storage attributes
            self._bg_color = "#2c3e50"
            self._fg_color = "#ecf0f1"
            self._current_text = ""
            self._current_source_text = ""
            self._current_language = None
            self._display_mode = "target_only"
            self.setup_widget()

        def setup_widget(self):
            """Configure widget for RTL display using HTML approach.

            Important: do NOT force a CSS line-height so Qt's native metrics determine spacing.
            Padding is left to the parent overlay to set so we can match tkinter's padx/pady.
            """
            # Default to RTL direction; actual text will control alignment when set.
            self.setLayoutDirection(Qt.RightToLeft)

            # Default font (can be overridden by parent overlay)
            font = QFont("Arial Unicode MS", 14)
            font.setFamilies(["Arial Unicode MS", "Segoe UI", "Tahoma"])
            self.setFont(font)

            self.setReadOnly(True)
            self.setLineWrapMode(QTextEdit.WidgetWidth)

            # Don't enforce line-height; keep margins/padding minimal here
            doc = self.document()
            doc.setDefaultStyleSheet("""
                div, p, table, tr, td {
                    margin: 0;
                    padding: 0;
                }
            """)

        def set_rtl_text(self, text: str, source_text: str, display_mode: str, language_code: str = None, bg_color: str = "#2c3e50", text_color: str = "#ecf0f1", font_size: int = 14):
            """Set text content while respecting RTL/LTR and applying inline HTML."""
            # Store current state for color updates and re-rendering
            self._current_text = text
            self._current_source_text = source_text
            self._current_language = language_code
            self._bg_color = bg_color
            self._display_mode = display_mode
            
            # Normalize whitespace while preserving intentional line breaks
            processed = ' '.join(text.replace('\r\n', '\n').replace('\r', '\n').split())
            processed_source = ' '.join(source_text.replace('\r\n', '\n').replace('\r', '\n').split())

            # Determine direction
            is_rtl = self._is_rtl_language(language_code) if language_code else self._detect_rtl_text(processed)

            # Optionally reshape Arabic
            if RESHAPER_AVAILABLE and is_rtl:
                try:
                    processed = arabic_reshaper.reshape(processed)
                    processed = get_display(processed)
                except Exception:
                    pass

            # Convert newlines to HTML line breaks
            html_processed = processed.replace('\n', '<br>')
            html_source = processed_source.replace('\n', '<br>')

            # Build the HTML content based on the display mode
            final_html = ""
            if display_mode == "source_target":
                final_html = f"""
                <table width="100%" style="font-family: '{self.font().family()}'; font-size: {font_size}pt; color: {text_color};">
                    <tr>
                        <td width="50%" style="vertical-align:top; padding-right: 10px; border-right: 1px solid #555;">{html_source}</td>
                        <td width="50%" style="vertical-align:top; padding-left: 10px;">{html_processed}</td>
                    </tr>
                </table>
                """
            elif display_mode == "overlay":
                final_html = f"""
                <div style="position: relative; font-family: '{self.font().family()}'; font-size: {font_size}pt;">
                    <div style="color: transparent;">{html_source}</div>
                    <div style="position: absolute; top: 0; left: 0; width: 100%; color: {text_color};">{html_processed}</div>
                </div>
                """
            else: # target_only
                align = "right" if is_rtl else "left"
                direction = "rtl" if is_rtl else "ltr"
                final_html = f"""<div style="text-align: {align}; direction: {direction}; font-family: '{self.font().family()}'; font-size: {font_size}pt; color: {text_color};">
                {html_processed}
                </div>"""

            self.setHtml(final_html)

            # Ensure block alignment matches direction for target_only mode
            if display_mode == "target_only":
                cursor = self.textCursor()
                cursor.select(QTextCursor.Document)
                block_fmt = QTextBlockFormat()
                block_fmt.setAlignment(Qt.AlignRight if is_rtl else Qt.AlignLeft)
                cursor.mergeBlockFormat(block_fmt)
                cursor.clearSelection()
                cursor.movePosition(QTextCursor.Start)
                self.setTextCursor(cursor)


        def _is_rtl_language(self, lang_code: str) -> bool:
            if not lang_code:
                return False
            rtl_languages = ['ar', 'he', 'fa', 'ur', 'yi', 'ku', 'ps', 'dv']
            return any(lang_code.lower().startswith(l) for l in rtl_languages)

        def _detect_rtl_text(self, text: str) -> bool:
            rtl_pattern = r'[\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
            return bool(re.search(rtl_pattern, text))

        # Simple compatibility wrappers mimicking tkinter Text behavior
        def winfo_exists(self):
            return True

        def winfo_viewable(self):
            return self.isVisible()

        def config(self, **kwargs):
            """Enhanced config method with full tkinter Text widget compatibility"""
            if 'state' in kwargs:
                self.setReadOnly('DISABLED' in str(kwargs['state']).upper())
            
            if 'bg' in kwargs:
                self._bg_color = kwargs['bg']
                self.setStyleSheet(f"background-color: transparent; border: none; padding: {self.parent()._text_padding[1]}px {self.parent()._text_padding[0]}px;")

            if 'fg' in kwargs:
                self._fg_color = kwargs['fg']
                self.set_rtl_text(
                    self._current_text, self._current_source_text, self._display_mode, self._current_language,
                    self._bg_color, self._fg_color, self.font().pointSize()
                )

            if 'font' in kwargs:
                font_spec = kwargs['font']
                if isinstance(font_spec, tuple) and len(font_spec) >= 2:
                    qfont = QFont(font_spec[0], int(font_spec[1]))
                    self.setFont(qfont)
                    self.set_rtl_text(
                        self._current_text, self._current_source_text, self._display_mode, self._current_language,
                        self._bg_color, self._fg_color, int(font_spec[1])
                    )

        def configure(self, **kwargs):
            return self.config(**kwargs)

        def get(self, start=None, end=None): return self.toPlainText()
        def delete(self, start=None, end=None): self.clear()
        def insert(self, index, text): self.setPlainText(text)
        def see(self, index): self.moveCursor(QTextCursor.Start)

    class VisualTopBar(QWidget):
        def __init__(self, parent=None, height: int = 10):
            super().__init__(parent)
            self.setFixedHeight(int(height))
            self.setCursor(Qt.SizeAllCursor)
            self.setMouseTracking(True)
            self._dragging = False
            self._drag_offset = QPoint()

        def _global_pos(self, event):
            try: return event.globalPosition().toPoint()
            except AttributeError: return event.globalPos()

        def enterEvent(self, event):
            self.setCursor(Qt.SizeAllCursor)
            super().enterEvent(event)

        def leaveEvent(self, event):
            try: self.window().unsetCursor()
            except Exception: pass
            super().leaveEvent(event)

        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                win = self.window().windowHandle()
                if win is not None and hasattr(win, 'startSystemMove'):
                    win.startSystemMove()
                else:
                    self._dragging = True
                    self._drag_offset = self._global_pos(event) - self.window().frameGeometry().topLeft()
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            self.setCursor(Qt.SizeAllCursor)
            if self._dragging and (event.buttons() & Qt.LeftButton):
                self.window().move(self._global_pos(event) - self._drag_offset)
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            self._dragging = False
            super().mouseReleaseEvent(event)


    class PySideTranslationOverlay(QMainWindow):
        def __init__(self, initial_geometry, bg_color, title="Translation",
                     top_bar_height: int = 10, text_padding: tuple = (5, 5),
                     font_size: int = 14, font_family: str = "Arial",
                     border_px: int = 0, opacity: float = 0.85, is_movable: bool = True, parent=None):
            super().__init__(parent)
            self.text_widget = None
            self.bg_color = bg_color
            self.parent = parent

            self._top_bar_height = int(top_bar_height)
            self._text_padding = (int(text_padding[0]), int(text_padding[1]))
            self._font_size = int(font_size)
            self._font_family = font_family
            self._border_px = int(border_px)
            self._opacity = float(opacity)
            self._is_movable = is_movable

            if sys.platform == "win32":
                self.HTLEFT, self.HTRIGHT, self.HTTOP, self.HTBOTTOM = 10, 11, 12, 15
                self.HTTOPLEFT, self.HTTOPRIGHT, self.HTBOTTOMLEFT, self.HTBOTTOMRIGHT = 13, 14, 16, 17

            self.setup_window(initial_geometry, bg_color, title)

        def setup_window(self, initial_geometry, bg_color, title):
            self.setWindowTitle(title)
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
            self.setStyleSheet("background-color: transparent;")

            try:
                x1, y1, x2, y2 = map(int, initial_geometry)
                self.setGeometry(x1, y1, max(x2 - x1, 100), max(y2 - y1, 50))
            except Exception: self.setGeometry(200, 200, 300, 200)

            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            border_css = f"border: {self._border_px}px solid {self._adjust_color_brightness(bg_color, -20)};" if self._border_px > 0 else "border: none;"
            central_widget.setStyleSheet(f"QWidget {{ background-color: {self._hex_to_rgba(bg_color, self._opacity)}; {border_css} }}")

            if self._is_movable:
                self.top_bar = VisualTopBar(self, height=self._top_bar_height)
                self.top_bar.setStyleSheet("background-color: transparent; border: none;")
                layout.addWidget(self.top_bar)

            self.text_widget = RTLTextDisplay(self)
            self.text_widget.setFont(QFont(self._font_family, self._font_size))
            self.text_widget.setStyleSheet(f"background-color: transparent; border: none; padding: {self._text_padding[1]}px {self._text_padding[0]}px;")
            layout.addWidget(self.text_widget)

        def _hex_to_rgba(self, hex, opacity):
            hex = hex.lstrip('#')
            r, g, b = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
            return f"rgba({r}, {g}, {b}, {opacity})"

        def _adjust_color_brightness(self, hex, adj):
            hex = hex.lstrip('#')
            r, g, b = (int(hex[i:i+2], 16) for i in (0, 2, 4))
            r, g, b = (max(0, min(255, c + adj)) for c in (r, g, b))
            return f"#{r:02x}{g:02x}{b:02x}"

        def show_translation(self, text: str, source_text: str, language_code: str = None, text_color: str = "#FFFFFF", font_size: int = None):
            if self.text_widget:
                display_mode = self.parent.overlay_display_mode_var.get()
                self.text_widget.set_rtl_text(text, source_text, display_mode, language_code, self.bg_color, self.text_widget._fg_color, font_size or self._font_size)

        def update_color(self, new_color, new_opacity=None):
            self.bg_color = new_color
            if new_opacity is not None: self._opacity = float(new_opacity)
            if self.centralWidget():
                border_css = f"border: {self._border_px}px solid {self._adjust_color_brightness(new_color, -20)};" if self._border_px > 0 else "border: none;"
                self.centralWidget().setStyleSheet(f"QWidget {{ background-color: {self._hex_to_rgba(new_color, self._opacity)}; {border_css} }}")
            if hasattr(self, 'top_bar'): self.top_bar.setStyleSheet("background-color: transparent; border: none;")
            if self.text_widget: self.text_widget.config(bg=new_color)

        def update_text_color(self, new_color):
            if self.text_widget: self.text_widget.config(fg=new_color)

        def get_geometry(self):
            return [self.x(), self.y(), self.x() + self.width(), self.y() + self.height()]

        def hide(self): super().hide()
        def show(self):
            super().show()
            self.raise_()
            self.activateWindow()
            self.update_color(self.bg_color)
        def toggle_visibility(self): self.setVisible(not self.isVisible())
        def winfo_exists(self): return True
        def winfo_viewable(self): return self.isVisible()
        def destroy(self): self.close()

        def mouseMoveEvent(self, event):
            if not self._is_movable: return super().mouseMoveEvent(event)
            if event.pos().y() <= self._top_bar_height: self.setCursor(Qt.SizeAllCursor)
            else:
                margin = 4
                x, y, w, h = event.pos().x(), event.pos().y(), self.width(), self.height()
                on_left, on_right, on_top, on_bottom = (x < margin), (x > w - margin), (y < margin), (y > h - margin)
                if (on_top and on_left) or (on_bottom and on_right): self.setCursor(Qt.SizeFDiagCursor)
                elif (on_top and on_right) or (on_bottom and on_left): self.setCursor(Qt.SizeBDiagCursor)
                elif on_left or on_right: self.setCursor(Qt.SizeHorCursor)
                elif on_top or on_bottom: self.setCursor(Qt.SizeVerCursor)
                else: self.setCursor(Qt.ArrowCursor)
            super().mouseMoveEvent(event)

        def nativeEvent(self, eventType, message):
            if not self._is_movable or sys.platform != "win32" or eventType != "windows_generic_MSG":
                return super().nativeEvent(eventType, message)
            msg = ctypes.wintypes.MSG.from_address(message.__int__())
            if msg.message == 0x0084: # WM_NCHITTEST
                x_phys, y_phys = ctypes.c_short(msg.lParam & 0xFFFF).value, ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
                scale = self.devicePixelRatioF() or 1.0
                logical_pt = self.mapFromGlobal(QPoint(int(round(x_phys / scale)), int(round(y_phys / scale))))

                if not self.rect().contains(logical_pt): return super().nativeEvent(eventType, message)

                margin = 4
                x, y, w, h = logical_pt.x(), logical_pt.y(), self.width(), self.height()
                on_left, on_right, on_top, on_bottom = (x < margin), (x > w - margin), (y < margin), (y > h - margin)

                if on_top and on_left: return True, self.HTTOPLEFT
                if on_top and on_right: return True, self.HTTOPRIGHT
                if on_bottom and on_left: return True, self.HTBOTTOMLEFT
                if on_bottom and on_right: return True, self.HTBOTTOMRIGHT
                if on_left: return True, self.HTLEFT
                if on_right: return True, self.HTRIGHT
                if on_top: return True, self.HTTOP
                if on_bottom: return True, self.HTBOTTOM
            return super().nativeEvent(eventType, message)
else:
    class RTLTextDisplay: raise ImportError("PySide6 not available")
    class PySideTranslationOverlay: raise ImportError("PySide6 not available")

def ensure_qapplication():
    if not PYSIDE6_AVAILABLE: return None
    if sys.platform == "win32":
        os.environ.setdefault('QT_AUTO_SCREEN_SCALE_FACTOR', '1')
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    return QApplication.instance() or QApplication(sys.argv)

class PySideOverlayManager:
    def __init__(self):
        self.overlay = None
        self.qapp = None
        self.available = PYSIDE6_AVAILABLE

    def create_overlay(self, app, initial_geometry, bg_color, title="Translation", **kwargs):
        if not self.available: return None
        self.qapp = self.qapp or ensure_qapplication()
        if self.overlay: self.overlay.close()
        self.overlay = PySideTranslationOverlay(initial_geometry, bg_color, title, parent=app, **kwargs)
        return self.overlay

    def close_overlay(self):
        if self.overlay: self.overlay.close()
        self.overlay = None

_pyside_manager = PySideOverlayManager()
def get_pyside_manager(): return _pyside_manager
def is_pyside_available(): return PYSIDE6_AVAILABLE
