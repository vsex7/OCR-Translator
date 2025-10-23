import pyautogui
import time

# Give me 10 seconds to switch the language
time.sleep(10)

# Take a screenshot
screenshot = pyautogui.screenshot()
screenshot.save("jules-scratch/verification/verification.png")
