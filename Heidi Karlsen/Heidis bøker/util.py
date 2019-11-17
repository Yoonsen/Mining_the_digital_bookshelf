
def find_dir():
    from tkinter.filedialog import askopenfilename, askdirectory
    from tkinter import Tk
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.after(1000, lambda: root.focus_force())
    name = askdirectory(parent=root)
    return name