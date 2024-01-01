import os
import tkinter as tk
from tkinter import filedialog
import huggingface_hub as hf_hub

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Download Manager")
        self.geometry("600x400")  # Adjusted the size for a better layout
        self.resizable(True, True)

        self.url_entry = []
        self.quantized_model_file = []
        self.vars = []
        self.create_widgets()

    def populate_quant_list(self, model_path):
        ignore = ["config.json", "pytorch_model.bin", "readme.md", 
                  "tokenizer.json", "tokenizer_config.json", "vocab.txt", "notice", "license.txt"]
        repo = model_path
        model_quant_list = hf_hub.list_repo_files(repo)
        self.quantized_model_file = [x for x in model_quant_list 
                                     if x.lower() not in ignore and not x.startswith('.')]
        print("Quantization List: {}".format(self.quantized_model_file))
        self.create_checkboxes()

    def create_checkboxes(self):
        # Clear existing checkboxes
        for widget in self.container.winfo_children():
            widget.destroy()

        self.vars = []
        for item in self.quantized_model_file:
            var = tk.BooleanVar()
            chk = tk.Checkbutton(self.container, text=item, variable=var)
            chk.pack(anchor='w', padx=5, pady=2)  # Added padding
            self.vars.append(var)

    def update_download_list(self):
        self.files_to_download = []
        self.url_entry = []
        for item, var in zip(self.quantized_model_file, self.vars):
            if var.get():
                self.files_to_download.append(item)
                self.url_entry.append("https://huggingface.co/" + self.model_path.get() 
                                      + "/resolve/main/" + item)
        print("Selections: {}".format(self.files_to_download))

    def create_widgets(self):
        # Layout using grid with better alignment and padding
        tk.Label(self, text="Model Path:").grid(row=0, column=0, sticky='e', padx=10, pady=10)
        self.model_path = tk.Entry(self, width=50)
        self.model_path.grid(row=0, column=1, padx=10, pady=10)

        tk.Button(self, text="Get Quantization List", 
                  command=lambda: self.populate_quant_list(self.model_path.get())).grid(row=1, column=1, padx=10, pady=5, sticky='e')

        tk.Label(self, text="Quantization Method:").grid(row=2, column=0, sticky='e', padx=10, pady=10)

        # Scrollable frame for checkboxes
        self.container = tk.Frame(self)
        self.container.grid(row=3, column=0, columnspan=2, padx=10, pady=5)
        self.scrollbar = tk.Scrollbar(self.container, orient="vertical")
        self.scrollbar.pack(side="right", fill="y")

        self.canvas = tk.Canvas(self.container, yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.config(command=self.canvas.yview)

        self.checkbox_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.checkbox_frame, anchor="nw")

        tk.Label(self, text="Download Location:").grid(row=4, column=0, sticky='e', padx=10, pady=10)
        self.location_entry = tk.Entry(self, width=50)
        self.location_entry.grid(row=4, column=1, padx=10, pady=10)

        tk.Button(self, text="...", command=self.open_file_dialog).grid(row=5, column=1, padx=10, pady=5, sticky='e')
        tk.Button(self, text="Download", command=self.download).grid(row=6, column=1, padx=10, pady=10, sticky='e')

    def open_file_dialog(self):
        self.location_entry.delete(0, tk.END)
        self.location_entry.insert(0, filedialog.askdirectory())

    def download(self):
        self.update_download_list()
        location = self.location_entry.get()
        for url in self.url_entry:
            os.system(f"wget -P {location} {url.split('/')[-1]} {url}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
