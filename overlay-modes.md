# Overlay Display Modes

This document outlines the three overlay display modes for the translated text.

## Modes

### 1. Target Only

This is the default mode. It displays only the translated text in the overlay window.

**Implementation:**
The `set_rtl_text` function in `pyside_overlay.py` receives the translated text and displays it directly.

### 2. Source + Target

This mode displays both the original source text and the translated target text side-by-side in a two-column layout.

**Implementation:**
The `set_rtl_text` function uses an HTML `<table>` to structure the source and target text in two columns.

### 3. Overlay

This mode displays the translated text directly over the original text's position, simulating a replacement.

**Implementation:**
The `set_rtl_text` function uses a `<div>` with `position: relative` and another `<div>` with `position: absolute` to place the translated text on top of the source text. The source text is rendered with a transparent color to maintain the layout.

## RTL Support

Right-to-left (RTL) language support is handled in all three modes. The `set_rtl_text` function detects if the language is RTL and sets the text direction accordingly. This ensures that languages like Arabic and Hebrew are displayed correctly.
