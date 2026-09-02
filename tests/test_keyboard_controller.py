import unittest
from unittest.mock import patch

from src.input import KeyBoardController as keyboard_controller


@unittest.skipIf(keyboard_controller.is_mac(), "Windows keyboard backend tests")
class WindowsKeyboardBackendTests(unittest.TestCase):
    def test_resolves_letter_with_scan_code(self):
        vk_code, scan_code, extended, modifiers = (
            keyboard_controller._resolve_windows_key("a")
        )

        self.assertEqual(vk_code, 0x41)
        self.assertEqual(scan_code, 0x1E)
        self.assertFalse(extended)
        self.assertEqual(modifiers, 0)

    def test_resolves_left_as_extended_key(self):
        self.assertEqual(
            keyboard_controller._resolve_windows_key("left"),
            (0x25, 0x4B, True, 0),
        )

    @patch.object(keyboard_controller, "_send_windows_key_event")
    def test_emits_left_key_down(self, send_event):
        keyboard_controller._emit_windows_key("left")

        send_event.assert_called_once_with(0x25, 0x4B, True)

    @patch.object(keyboard_controller, "_send_windows_key_event")
    def test_emits_left_key_up(self, send_event):
        keyboard_controller._emit_windows_key("left", key_up=True)

        send_event.assert_called_once_with(0x25, 0x4B, True, key_up=True)

    def test_rejects_unknown_named_key(self):
        with self.assertRaises(ValueError):
            keyboard_controller._resolve_windows_key("not-a-key")


if __name__ == "__main__":
    unittest.main()
