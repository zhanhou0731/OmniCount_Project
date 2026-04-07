import tkinter as tk
from tkinter import ttk
from ui.tabs import MainCounterTab, HistoryTab

class OmniCountApp(tk.Tk):
    """The Main Application Manager."""
    def __init__(self):
        super().__init__()
        self.title("OmniCount: Smart Object Counting System")
        self.geometry("1000x650")
        self.minsize(800, 500)

        self.count_var = tk.StringVar(value="TOTAL COUNT: 0")
        self.setup_notebook()
        self.setup_statusbar()

    def setup_notebook(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_main = MainCounterTab(self.notebook, self.count_var)
        self.tab_history = HistoryTab(self.notebook)

        self.notebook.add(self.tab_main, text="   Main Counter   ")
        self.notebook.add(self.tab_history, text="   Session History & Reports   ")

    def setup_statusbar(self):
        status_frame = ttk.Frame(self, relief=tk.SUNKEN, padding=(10, 2))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = ttk.Label(status_frame, text="Status: Ready.")
        self.status_label.pack(side=tk.LEFT)

        self.count_label = tk.Label(status_frame, textvariable=self.count_var, font=("Arial", 12, "bold"), fg="green")
        self.count_label.pack(side=tk.RIGHT)
        