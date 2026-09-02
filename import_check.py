import sys
print("PYTHON:", sys.executable)
try:
    import tools.auto_combat
    print("IMPORT_OK")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("IMPORT_FAIL")
