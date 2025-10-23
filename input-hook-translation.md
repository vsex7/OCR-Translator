# Input Hook Translation Feature

This document outlines the functionality and usage of the Input Hook Translation feature.

## Overview

The Input Hook Translation feature allows you to translate text directly within any application's input field. When enabled, a small floating icon will appear next to the focused input field. Clicking this icon will translate the text in the input field to your selected target language.

## How to Use

1.  **Enable the Feature:** Go to the "Settings" tab and check the "Enable input field translation" checkbox.
2.  **Focus an Input Field:** Click on any text input field in any application (e.g., a chat window, a text editor).
3.  **Click the Icon:** A small translate icon will appear next to the input field. Click it.
4.  **Translation:**
    *   If you have selected text in the input field, only the selected text will be translated and replaced.
    *   If you have not selected any text, the entire content of the input field will be translated and replaced.
5.  **Undo:** To undo the translation and restore the original text, press `Ctrl+Z`. You can undo up to the last 10 translations.

## Limitations

*   **Compatibility:** This feature works best with standard Windows input controls (e.g., "Edit" and "RichEdit" classes). It may not work with custom or non-standard input fields in some applications.
*   **Permissions:** The application may require administrator privileges to detect input fields in some applications.
*   **Focus:** The floating icon will only appear when an input field is actively focused.

## Technical Details

*   **Detection:** The feature uses `pywin32` to detect the focused window and input control.
*   **Interaction:** The `keyboard` and `pyperclip` libraries are used to simulate `Ctrl+A`, `Ctrl+C`, and `Ctrl+V` to select, copy, and paste text.
*   **Icon:** The floating icon is a frameless `Tkinter` `Toplevel` window that is always on top.
