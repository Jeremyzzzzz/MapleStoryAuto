# Standard Import
import sys

# Pyside
from PySide6.QtWidgets import QApplication

# Load Import
from src.ui.ui import MainWindow
from src.ui.AutoBotController import AutoBotController

def main():
    '''
    Main Function
    Run: python -m ui.main
    '''
    app = QApplication(sys.argv)

    autoBotController = AutoBotController()
    ui = MainWindow(autoBotController)

    autoBotController.update_signal(ui)

    ui.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
