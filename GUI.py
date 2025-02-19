import json
import os
import queue
import signal
import time
import platform
import requests
import zipfile
import tkinter as tk
import tkinter.messagebox
from multiprocessing import Process, Queue
from tkinter import ttk, filedialog, messagebox
from typing import List, Tuple, Optional
from shutil import copy

from PIL import Image, ImageTk

from src.src_pipeline import Pipeline, QC, CalciumZero
from src.utils import *


def test(queue, idx):
    msg = {'idx': idx, 'is_running': False, 'is_finished': False}
    # print(f"putting first {msg}")
    time.sleep(5)
    # instance_list[idx].is_running = True
    msg['is_running'] = True
    print(f"putting second {msg}")
    queue.put(msg)
    time.sleep(20)
    # instance_list[idx].is_finished = True
    msg['is_finished'] = True
    print(f"putting third {msg}")
    queue.put(msg)


class MyTk(tk.Tk):
    # @override
    def __init__(self, logger, *args, **kwargs):
        self.logger = logger
        self.screen_width = self.screen_height = 0
        super().__init__(*args, **kwargs)

    # @override
    def report_callback_exception(self, exc, val, tb):
        import traceback
        trace = ''.join(traceback.format_exception(exc, val, tb))
        self.logger.error(f"Exception: \n{trace}")
        super().report_callback_exception(exc, val, tb)


class GUI:
    __slots__ = [
        'logger',
        'project_path',
        'debug',
        # GUI fields
        'root',
        'TAB_MAX',
        # Pipeline fields
        'instance_list',
        'process_list',
        'tab_list',
        'status_list',
        'queue',
        'log_queue',
        'log_th'
    ]

    def center(self, window, width: int, height: int):
        """
        Function to center any window on the screen

        Params:
            window: window to be centered
            width: width of the window
            height: height of the window
        """
        if self.root.screen_width == 0 or self.root.screen_height == 0:
            raise Exception("Screen size is not initialized")
        # Find the center point
        center_x = (self.root.screen_width - width) // 2
        center_y = (self.root.screen_height - height) // 2

        # Set the position of the window to the center of the screen
        window.geometry(f'{width}x{height}+{center_x}+{center_y}')

    def log(self, msg):
        iprint(msg, self.logger)

    def __init__(self, debug=False):
        self.debug = debug
        # Set up logging
        self.project_path = Path(__file__).parent
        if self.project_path.is_dir():
            self.logger = setup_logger(self.project_path)
        else:
            self.project_path = Path.cwd()
            self.logger = setup_logger(self.project_path)
        self.log(f"Project path set to {self.project_path}")
        self.prepare_imagej()
        # TODO: Read Configuration files if needed.
        pass
        self.queue = Queue()
        self.log_queue = Queue()

        # Instance-related List
        self.TAB_MAX = 4
        self.instance_list: List[Optional[CalciumZero]] = list(None for _ in range(self.TAB_MAX))
        self.tab_list: List[Optional[tkinter.Frame]] = list(None for _ in range(self.TAB_MAX))
        self.process_list: List[Optional[Process]] = list(None for _ in range(self.TAB_MAX))
        self.status_list: List[Optional[str]] = list(None for _ in range(self.TAB_MAX))

        # Initialize Main Window
        self.root = MyTk(self.logger)
        self.root.title("CalciumZero")
        # Define the window size
        window_width = 1200
        window_height = 800

        # Get the screen dimension
        self.root.screen_width = self.root.winfo_screenwidth()
        self.root.screen_height = self.root.winfo_screenheight()

        # Set root to center
        self.center(self.root, window_width, window_height)

        # Create the Menu
        self.create_menu()

        # Create the Notebook
        self.root.notebook = ttk.Notebook(self.root)
        self.root.notebook.pack(expand=True, fill='both')

        # Set up the app
        self.setup()
        self.log('GUI setup finished.')

    def prepare_imagej(self):
        # Check if installed
        if Path(self.project_path).joinpath('Fiji.app').exists():
            self.log(f"ImageJ already installed in {self.project_path}.")
            return
        system = platform.system()
        # match system:
        #     case 'Windows':
        #         url = 'https://downloads.imagej.net/fiji/latest/fiji-win64.zip'
        #     case 'Linux':
        #         url = 'https://downloads.imagej.net/fiji/latest/fiji-linux64.zip'
        #     case 'Darwin':
        #         url = 'https://downloads.imagej.net/fiji/latest/fiji-macosx.zip'
        #     case _:
        #         raise ValueError(f"Unsupported system: {system}")
        if system == 'Windows':
            url = 'https://downloads.imagej.net/fiji/latest/fiji-win64.zip'
        elif system == 'Linux':
            url = 'https://downloads.imagej.net/fiji/latest/fiji-linux64.zip'
        elif system == 'Darwin':
            url = 'https://downloads.imagej.net/fiji/latest/fiji-macosx.zip'
        else:
            raise ValueError(f"Unsupported system: {system}")
        self.log(f"On {system}, downloading ImageJ from {url}")
        # Download and unzip
        r = requests.get(url)
        if r.status_code == 200:
            temp_path = Path(self.project_path).joinpath('fiji.zip')
            try:
                with open(temp_path, 'wb') as temp_file:
                    temp_file.write(r.content)
                with zipfile.ZipFile(str(temp_path), 'r') as zip_ref:
                    zip_ref.extractall(self.project_path)
            finally:
                temp_path.unlink()
        else:
            raise ConnectionError(f"Failed to download ImageJ from {url}. Status code: {r.status_code}")
        self.log(f"ImageJ installed in {self.project_path}.")
        try:
            copy(str(self.project_path.joinpath('resource').joinpath('Image_Stabilizer_Headless.class')),
                 str(self.project_path.joinpath('Fiji.app').joinpath('plugins').joinpath(
                     'Image_Stabilizer_Headless.class')))
        except Exception as e:
            self.log(f"Failed to copy Image_Stabilizer_Headless.class to ImageJ plugins folder. Error: {e}")
        return str(self.project_path.joinpath('Fiji.app'))

    def create_menu(self):
        # Create a Menu Bar
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
        # Add 'File' Menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="  File  ", menu=file_menu)

        # Add 'New' Menu Item
        # Function to create a parameter dialog
        file_menu.add_command(label="New Instance", command=self.new_run_dialog)
        file_menu.add_command(label="New QC", command=self.new_qc_dialog)

    def new_run_dialog(self):
        with open(self.project_path.joinpath("config.json"), "r") as f:
            param_dict = json.load(f)
        # Create a simple dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("New instance initialization")
        self.center(dialog, 800, 600)

        # Add fields for parameters (e.g., tab name)
        # TODO: Add more fields for other parameters
        # Add checkboxes
        def on_checkbox_change(idx, var):
            value = var.get()
            param_dict['run'][idx] = True if value else False

        checkbox_frame = tk.Frame(dialog)
        checkbox_frame.pack()
        run_1, run_2, run_3, run_4 = tk.IntVar(), tk.IntVar(), tk.IntVar(), tk.IntVar()
        tk.Label(checkbox_frame, text="Run:").pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(checkbox_frame, text="Crop", variable=run_1,
                       command=lambda n=0, v=run_1: on_checkbox_change(n, v)).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(checkbox_frame, text="Stabilizer", variable=run_2,
                       command=lambda n=1, v=run_2: on_checkbox_change(n, v)).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(checkbox_frame, text="CaImAn", variable=run_3,
                       command=lambda n=2, v=run_3: on_checkbox_change(n, v)).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(checkbox_frame, text="PeakCall", variable=run_4,
                       command=lambda n=3, v=run_4: on_checkbox_change(n, v)).pack(side=tk.LEFT, padx=5)

        # File/Folder selections
        ioselect_frame = tk.Frame(checkbox_frame)
        ioselect_frame.pack()

        # Input file selection
        def on_select_file():
            # Determine filetypes to use for file selection
            filetypes = [("TIF files", "*.tif"), ("All files", "*.*")]

            file_path = filedialog.askopenfilename(parent=dialog, title="Select a file", filetypes=filetypes)
            param_dict['input_path'] = file_path
            self.log(f"Selected input file: {file_path}")

        tk.Button(ioselect_frame, text="Select input File", command=on_select_file).pack(side=tk.LEFT, padx=5)

        # Output folder selection
        def on_select_folder():
            folder_path = filedialog.askdirectory(parent=dialog, title="Select a folder")
            param_dict['output_path'] = folder_path
            self.log(f"Selected output folder: {folder_path}")

        tk.Button(ioselect_frame, text="Select output Folder", command=on_select_folder).pack(side=tk.LEFT, padx=5)

        # Add param fields for each checkbox
        notebook = ttk.Notebook(dialog)
        notebook.pack(expand=True, fill='both')

        # Crop param tab
        tab_1 = ttk.Frame(notebook)
        tab_1.pack(expand=True)
        notebook.add(tab_1, text='Crop')

        def on_slider_change(event):
            param_dict['crop'].update(threshold=slider.get())

        # Create a container frame for the label and slider
        container_1 = ttk.Frame(tab_1)
        container_1.pack(expand=True)
        # Threshold Label
        label = ttk.Label(container_1, text=f"Threshold:")
        label.pack(side=tk.LEFT, padx=5)
        # Threshold Slider
        slider = tk.Scale(container_1, from_=0, to=500, orient='horizontal', resolution=10,
                          command=on_slider_change)
        slider.pack(side=tk.LEFT, padx=5)
        slider.set(param_dict['crop']['threshold'])

        # Validate input numbers
        def validate_numbers(num, type_d):
            if num == '':
                return (True, 0) if type_d is int else (True, 0.0)
            else:
                try:
                    return (True, int(num)) if type_d is int else (True, float(num))
                except ValueError:
                    return False, None

        # Stabilizer param tab
        tab_2 = ttk.Frame(notebook)
        tab_2.pack(expand=True)
        notebook.add(tab_2, text='Stabilizer')
        # Stabilizer containers
        for dname, d in param_dict['stabilizer'].items():
            def on_entry_change_stab(value, key):
                print(value, key)
                if key == 'Transformation':
                    param_dict['stabilizer'][key] = value
                else:
                    status, val = validate_numbers(value, type(d))
                    if not status:
                        messagebox.showerror("Invalid Input", "Please enter a valid numeric value.")
                    else:
                        param_dict['stabilizer'][key] = val

            container = ttk.Frame(tab_2)
            container.pack(expand=True)
            label = ttk.Label(container, text=dname)
            label.pack(padx=10, side=tk.LEFT)
            if type(d) is str:
                # A selection entry so user can select from a list
                var = tk.StringVar(value='Translation')
                trans_button = tk.Radiobutton(container, text="Translation", variable=var, value="Translation")
                trans_button.pack(padx=10, side=tk.LEFT)
                affine_button = tk.Radiobutton(container, text="Affine", variable=var, value="Affine")
                affine_button.pack(padx=10, side=tk.LEFT)
                trans_button.select()  # Set the Translation Radiobutton as the default selection
                var.trace_add("write",
                              lambda *args, text=var, key=dname: on_entry_change_stab(text.get(), key))
            else:
                # Text entry for numbers
                entry_text = tk.StringVar(value=d)
                entry = ttk.Entry(container, textvariable=entry_text)
                entry.pack(padx=10, side=tk.LEFT)
                entry_text.trace_add("write",
                                     lambda *args, text=entry_text, key=dname: on_entry_change_stab(text.get(), key))

        # CaImAn param tab
        tab_3 = ttk.Frame(notebook)
        tab_3.pack(expand=True)
        notebook.add(tab_3, text='CaImAn')
        # CaImAn containers
        for dname, d in param_dict['caiman'].items():

            container_outer = ttk.Frame(tab_3)
            container_outer.pack(side=tk.LEFT, expand=True, fill='both', padx=5, pady=5)
            for k, v in d.items():
                def on_entry_change(value, key, dname=dname):
                    if type(value) is str:
                        status, val = validate_numbers(value, type(v))
                        if not status:
                            if key not in ['border_nan', 'method_init', 'method_deconvolution']:
                                messagebox.showerror("Invalid Input", "Please enter a valid numeric value.")
                            else:
                                if value == "None":
                                    param_dict['caiman'][dname][key] = None
                                else:
                                    param_dict['caiman'][dname][key] = value
                        # Numbers
                        else:
                            param_dict['caiman'][dname][key] = val
                    elif type(value) is bool:
                        param_dict['caiman'][dname][key] = True
                    elif type(value) is list:
                        status, val = validate_numbers(value[0], type(v))
                        if not status:
                            messagebox.showerror("Invalid Input", "Please enter a valid numeric value.")
                        else:
                            param_dict['caiman'][key] = [int(vi) if vi != '' else 0 for vi in value]
                    else:
                        raise NotImplementedError(f"Unknown type {type(value)} for value {value}.")

                if k == "fnames":
                    continue
                # Create a container frame for the label and entry
                container = ttk.Frame(container_outer)
                container.pack(expand=True)
                # Label
                label = ttk.Label(container, text=f"{k}:")
                label.pack(side=tk.LEFT, padx=5)
                # Entry type 1: strings
                if type(v) is str:
                    if k == 'border_nan':
                        var = tk.StringVar(value=v)

                        container_inner = ttk.Frame(container)
                        container_inner.pack()

                        # Radiobutton for True
                        copy_radio = tk.Radiobutton(container_inner, text="copy", variable=var, value="copy")
                        copy_radio.pack(side=tk.LEFT, padx=5)

                        # Radiobutton for False
                        none_radio = tk.Radiobutton(container_inner, text="None", variable=var, value="None")
                        none_radio.pack(side=tk.LEFT, padx=5)

                        var.trace_add("write",
                                      lambda *args, text=var, key=k: on_entry_change(text.get(), key))
                    elif k == 'method_init':
                        pass
                    else:  # "method_deconvolution"
                        pass
                # Entry type 2: True / False
                elif type(v) is bool:
                    var = tk.StringVar(value="True" if v else "False")

                    container_inner = ttk.Frame(container)
                    container_inner.pack()

                    # Radiobutton for True
                    true_radio = tk.Radiobutton(container_inner, text="True", variable=var, value="True")
                    true_radio.pack(side=tk.LEFT, padx=5)

                    # Radiobutton for False
                    false_radio = tk.Radiobutton(container_inner, text="False", variable=var, value="False")
                    false_radio.pack(side=tk.LEFT, padx=5)

                    var.trace_add("write",
                                  lambda *args, text=var, key=k: on_entry_change(
                                      True if text.get() == "True" else False, key))
                # Entry type 3: list[int, int]
                elif type(v) is list:
                    var = tk.StringVar(value=v[0])
                    container_inner = ttk.Frame(container)
                    container_inner.pack()
                    entry_1 = ttk.Entry(container_inner, textvariable=var, width=10)
                    entry_1.pack(side=tk.LEFT, padx=5)
                    entry_2 = ttk.Entry(container_inner, textvariable=var, width=10)
                    entry_2.pack(side=tk.LEFT, padx=5)
                    var.trace_add("write",
                                  lambda *args, text=var, key=k: on_entry_change([text.get(), text.get()], key))
                # Entry type 4: numbers
                else:
                    entry_text = tk.StringVar(value=v)
                    entry = ttk.Entry(container, textvariable=entry_text)
                    entry.pack(side=tk.LEFT, padx=5)
                    entry_text.trace_add("write",
                                         lambda *args, text=entry_text, key=k: on_entry_change(text.get(), key))

        # Function to handle 'OK' button click
        def on_ok():
            if param_dict['run'][2]:
                # Handle caiman fnames
                fname = param_dict['input_path'][:-4]
                if param_dict['run'][0]:
                    fname += '_crop'
                if param_dict['run'][1]:
                    fname += '_stab'
                fname = fname + '.tif'
                self.log(f'Expect caiman fname: {fname}')
                param_dict['caiman']['mc_dict']['fnames'] = [fname]
            status, msg = self.create_instance(run_params=param_dict)
            if not status:
                self.log(f'Instance creation failed: {msg}')
                tkinter.messagebox.showerror("Error", f"Instance creation failed: {msg}", parent=dialog)
                return
            dialog.destroy()
            self.log(f'New instance created and added to position {msg}.')

        # OK and Cancel buttons
        tk.Button(dialog, text="OK", command=on_ok).pack(side="left", padx=10, pady=10)
        tk.Button(dialog, text="Cancel", command=dialog.destroy).pack(side="right", padx=10, pady=10)

    def new_qc_dialog(self):
        qc_path = ''
        # Create a simple dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("New qc initialization")
        self.center(dialog, 400, 300)

        frame_1 = tk.Frame(dialog)
        frame_1.pack()

        # Input file selection
        def on_select_file():
            nonlocal qc_path
            # Determine filetypes to use for file selection
            filetypes = [("cmobj", "*.cmobj"), ("cm_obj", "*.cm_obj"), ("All files", "*.*")]

            qc_path = filedialog.askopenfilename(parent=dialog, title="Select a file", filetypes=filetypes)
            self.log(f"Selected qc file: {qc_path}")

        tk.Button(frame_1, text="Select QC File", command=on_select_file).pack(side=tk.LEFT, padx=5)

        frame_2 = tk.Frame(dialog)
        frame_2.pack()

        # Function to handle 'OK' button click
        def on_ok():
            if qc_path is None or qc_path == '':
                self.log(f'No qc file selected.')
                tkinter.messagebox.showerror("Error", "No qc file selected.")
                return

            status, msg = self.create_instance(qc_param={'cm_obj': qc_path})
            if not status:
                self.log(f'Instance creation failed: {msg}')
                tkinter.messagebox.showerror("Error", f"Instance creation failed: {msg}")
                return
            dialog.destroy()
            self.log(f'New QC instance created and added to position {msg}.')

        # OK and Cancel buttons
        tk.Button(frame_2, text="OK", command=on_ok).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(frame_2, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=10, pady=10)

    def create_instance(self, run_params=None, qc_param=None, idx=None) -> Tuple[bool, str]:
        """
        Create a new instance or fill in an existing instance with run and qc parameters
        Args:
            run_params: Parameters for running the instance
            qc_param: Parameters for creating a QC instance
            idx: Instance index to append to

        Returns:
            status: True if successful, False otherwise
            msg: Error message if status is False
        """
        # Find corresponding idx to use
        if idx is None:
            # Find next available index
            try:
                idx = self.instance_list.index(None)
            except ValueError:
                return False, "Maximum number of instances reached."
            cur_instance = CalciumZero()
        else:
            cur_instance = self.instance_list[idx]
        # TODO: customized tab name
        cur_tab = self.create_tab(idx)
        self.tab_list[idx] = cur_tab

        if run_params:
            run_instance = Pipeline(
                queue=self.queue,
                queue_id=idx,
                log_queue=self.log_queue,
            )
            for item in ['input_path', 'output_path', 'run']:
                if item not in run_params:
                    return False, f"Missing parameter {item}."
            run_instance.update(params_dict=run_params)  # Setup parameters
            ijp = str(self.project_path.joinpath("Fiji.app"))
            self.log(f'ImageJ path set to {ijp}.')
            run_instance.update(ijp=ijp)
            cur_instance.run_instance = run_instance
            self.instance_list[idx] = cur_instance
            self.create_run_tab(idx)
        # QC can be created initially OR appended to an existing instance
        if qc_param:
            qc_instance = QC(qc_param['cm_obj'], debug=False)
            cur_instance.qc_instance = qc_instance
            self.instance_list[idx] = cur_instance
            self.create_qc_tab(idx)

        self.status_list[idx] = 'idle'
        self.process_list[idx] = None

        self.root.notebook.add(cur_tab, text='New Instance')
        if hasattr(self.root.notebook, "tabs_list"):
            self.root.notebook.tabs_list.append(cur_tab)
        else:
            self.root.notebook.tabs_list = [cur_tab]
        return True, idx

    def create_run_tab(self, idx: int) -> None:
        """
        Create a run tab for the instance at idx position
        Args:
            idx: Instance index

        Returns:

        """
        # 1. Get instance tab.
        instance_tab = self.create_tab(idx)
        cur_instance = self.instance_list[idx]
        # Create run tab
        run_tab = ttk.Frame()
        run_tab.pack(expand=True, fill='both')
        ss_button = ttk.Button(run_tab, text='Start', command=lambda id=instance_tab.id: self.run_instance(id))
        ss_button.pack(side='top', padx=5, pady=5)
        run_tab.ss_button = ss_button

        close_button = ttk.Button(run_tab, text='Close',
                                  command=lambda id=instance_tab.id: self.close_instance(id))
        close_button.pack(side='top', padx=5, pady=5)
        run_tab.close_button = close_button

        param_button = ttk.Button(run_tab, text='Show Params',
                                  command=lambda id=instance_tab.id: self.show_params(id))
        param_button.pack(side='top', padx=5, pady=5)
        instance_tab.notebook.add(run_tab, text='Run')
        instance_tab.notebook.run_tab = run_tab

    def create_qc_tab(self, idx: int) -> None:
        """
        Create a CaImAn QC tab for the instance at idx position
        Args:
            idx: Instance index

        Returns:

        """
        cur_instance = self.instance_list[idx]
        # 1. Get instance tab.
        instance_tab = self.create_tab(idx)
        # 2. Create QC tab
        qc_instance: QC = cur_instance.qc_instance
        qc_tab = ttk.Frame()
        qc_tab.pack(expand=True, fill='both')
        qc_tab.movie_idx = 0

        qc_instance.qc_tab = qc_tab

        qc_tab_container = ttk.Frame(qc_tab)
        qc_tab_container.pack()

        # Define a callback function to update the scrollbar and canvas
        def update_canvas_from_scrollbar(value):
            frame_idx = int(value)
            if qc_tab.roi_enabled:
                ROI_index = qc_tab.roi_idx.get()
            else:
                ROI_index = None
            cur_frame = qc_instance.show_frame(image_idx=qc_tab.movie_idx, frame_idx=frame_idx, ROI_idx=ROI_index)

            cur_frame = Image.fromarray(cur_frame)
            cur_frame = cur_frame.convert("L")  # Convert to grayscale mode
            cur_frame = cur_frame.resize((w_, h_), Image.Resampling.LANCZOS)
            cur_frame = ImageTk.PhotoImage(cur_frame)
            qc_tab.canvas.create_image(0, 0, anchor="nw", image=cur_frame)
            qc_tab.canvas.image = cur_frame  # Keep a reference to prevent garbage collection
            # Update the input field with the current frame number
            qc_tab.input_field.delete(0, tk.END)
            qc_tab.input_field.insert(0, str(frame_idx))

        # Define a callback function to update the scrollbar and canvas when the user types in the input field
        def update_canvas_from_input(event):
            try:
                frame_idx = int(qc_tab.input_field.get())
                if 0 <= frame_idx < qc_instance.image_shape(qc_tab.movie_idx)[0]:
                    qc_tab.scrollbar.set(frame_idx)
                    update_canvas_from_scrollbar(frame_idx)
                else:
                    raise ValueError
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "Please enter a valid frame number.")
                qc_tab.scrollbar.set(0)
                qc_tab.input_field.delete(0, tk.END)
                qc_tab.input_field.insert(0, "0")

        def on_listbox_select(event):
            # Get the index of the selected item in the Listbox
            selected_index = qc_tab.listbox.curselection()
            if selected_index:  # Ensure an item is selected
                qc_tab.movie_idx = int(selected_index[0])
                update_canvas_from_scrollbar(0)

        qc_tab.listbox = tk.Listbox(qc_tab_container, selectmode=tk.SINGLE, height=5)
        qc_tab.listbox.pack(side=tk.LEFT, fill='y')
        for i in range(len(qc_instance.movies)):
            qc_tab.listbox.insert(tk.END, f"Movie {i}")
        # qc_tab.listbox.insert(tk.END, "test")
        qc_tab.listbox.config(height=qc_tab.listbox.size())
        qc_tab.listbox.bind("<<ListboxSelect>>", on_listbox_select)

        max_h, max_w = 500, 800
        h_, w_ = qc_instance.image_shape(0)[1:]
        if h_ > max_h or w_ > max_w:
            h_, w_ = int(h_ * min(max_h / h_, max_w / w_)), int(w_ * min(max_h / h_, max_w / w_))
        qc_tab.canvas = tk.Canvas(qc_tab_container, width=w_, height=h_)
        qc_tab.canvas.pack(side=tk.LEFT)

        # A ROI checkbox to show the ROI
        def on_roi_en_change():
            qc_tab.roi_enabled = roi_en.get()
            if qc_tab.roi_enabled:
                qc_tab.roi_scrollbar.config(state=tk.NORMAL)
                qc_tab.roi_input.config(state=tk.NORMAL)
                update_roi_idx_from_scrollbar(event=None)
            else:
                qc_tab.roi_scrollbar.config(state=tk.DISABLED)
                qc_tab.roi_input.config(state=tk.DISABLED)
            update_roi_idx_from_scrollbar(event=None)

        qc_roi_container = ttk.Frame(qc_tab_container)
        qc_roi_container.pack(side=tk.LEFT)
        qc_tab.roi_enabled = False
        roi_en = tk.BooleanVar(value=False)
        qc_tab.roi_cb = tk.Checkbutton(qc_roi_container, text="Show ROI", variable=roi_en, command=on_roi_en_change)
        qc_tab.roi_cb.pack()
        ac_rj_container = ttk.Frame(qc_roi_container)
        ac_rj_container.pack(side=tk.BOTTOM)  # side=tk.LEFT)
        ac_var, rj_var = tk.IntVar(), tk.IntVar()

        def update_listbox(event=None):
            ac, rj = ac_var.get(), rj_var.get()
            qc_tab.lb.delete(0, tk.END)
            if ac and rj:
                idxs = range(qc_instance.n_ROIs)
            elif ac:
                idxs = qc_instance.data.estimates.idx_components
            elif rj:
                idxs = list(set(range(qc_instance.n_ROIs)) - set(qc_instance.data.estimates.idx_components))
            else:
                qc_tab.roi_scrollbar.config(state=tk.NORMAL)
                return
            qc_tab.roi_scrollbar.config(state=tk.DISABLED)
            # qc_tab.roi_input.config(state=tk.DISABLED)
            for id in idxs:
                qc_tab.lb.insert(tk.END, id)
        qc_tab.roi_acb = tk.Checkbutton(ac_rj_container, text="Accept", variable=ac_var, command=update_listbox)
        qc_tab.roi_acb.pack()
        qc_tab.roi_rjb = tk.Checkbutton(ac_rj_container, text="Reject", variable=rj_var, command=update_listbox)
        qc_tab.roi_rjb.pack()
        lb_container = ttk.Frame(qc_roi_container)
        lb_container.pack(side=tk.BOTTOM)
        qc_tab.lb = tk.Listbox(lb_container, selectmode=tk.SINGLE, height=10)

        def update_roi_idx_from_lb(event=None):
            idx = qc_tab.lb.get(qc_tab.lb.curselection())
            qc_tab.roi_input.delete(0, tk.END)
            qc_tab.roi_input.insert(0, idx)
            qc_tab.roi_idx.set(idx)
            update_canvas_from_input(event=event)
        qc_tab.lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        qc_tab.lb.bind("<<ListboxSelect>>", update_roi_idx_from_lb)

        # Create a scrollbar
        scrollbar = tk.Scrollbar(lb_container, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the listbox to work with the scrollbar
        qc_tab.lb.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=qc_tab.lb.yview)

        # ROI scrollbar
        def update_roi_idx_from_scrollbar(event):
            qc_tab.roi_input.delete(0, tk.END)
            qc_tab.roi_input.insert(0, str(qc_tab.roi_idx.get()))
            update_canvas_from_input(event=event)

        def update_roi_idx_from_input(event):
            try:
                roi_idx = int(qc_tab.roi_input.get())
                qc_tab.roi_scrollbar.set(roi_idx)
                assert qc_tab.roi_idx.get() == roi_idx
                update_roi_idx_from_scrollbar(event)
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "Please enter a valid ROI index number.")
                qc_tab.roi_scrollbar.set(0)
                qc_tab.roi_input.delete(0, tk.END)
                qc_tab.roi_input.insert(0, "0")
        qc_tab.roi_idx = tk.IntVar(value=0)
        qc_tab.roi_scrollbar = tk.Scale(qc_roi_container, from_=0, to=qc_instance.n_ROIs - 1, orient="horizontal",
                                        variable=qc_tab.roi_idx, showvalue=False, resolution=1,
                                        command=update_roi_idx_from_scrollbar)
        qc_tab.roi_scrollbar.config(state=tk.DISABLED)
        qc_tab.roi_scrollbar.pack(side=tk.LEFT, padx=5, pady=5)
        # Create an input field next to the scrollbar
        qc_tab.roi_input = tk.Entry(qc_roi_container, width=5)
        qc_tab.roi_input.pack(side=tk.LEFT, padx=5, pady=5)
        # Bind the input field to the update_canvas_from_input function
        qc_tab.roi_input.bind("<Return>", update_roi_idx_from_input)

        container = ttk.Frame(qc_tab)
        container.pack()

        container_1 = ttk.Frame(container)
        container_1.pack()

        container_2 = ttk.Frame(container)
        container_2.pack()

        frame_label = ttk.Label(container_2, text="Frame Index")
        frame_label.pack(side=tk.LEFT, padx=5, pady=5)
        qc_tab.scrollbar = tk.Scale(container_2, from_=0, to=qc_instance.image_shape(qc_tab.movie_idx)[0] - 1,
                                    orient="horizontal",
                                    command=update_canvas_from_scrollbar, showvalue=False)
        qc_tab.scrollbar.pack(side=tk.LEFT, padx=5, pady=5)

        # Create an input field next to the scrollbar
        qc_tab.input_field = tk.Entry(container_2, width=5)
        qc_tab.input_field.pack(side=tk.LEFT, padx=5, pady=5)

        # Bind the input field to the update_canvas_from_input function
        qc_tab.input_field.bind("<Return>", update_canvas_from_input)

        close_button = ttk.Button(qc_tab, text='Close',
                                  command=lambda id=instance_tab.id: self.close_instance(id))
        close_button.pack(side='top', padx=5, pady=5)
        qc_tab.close_button = close_button

        # Initializations go here
        update_canvas_from_scrollbar(0)

        instance_tab.notebook.add(qc_tab, text='QC')
        instance_tab.notebook.qc_tab = qc_tab

    def create_tab(self, idx):
        # Get instance tab. If not exist, create one
        if self.tab_list[idx]:
            return self.tab_list[idx]
        else:
            instance_tab = ttk.Frame(self.root.notebook)
            instance_tab.pack(expand=True, fill='both')
            instance_tab.id = idx
            instance_tab.notebook = ttk.Notebook(instance_tab)
            instance_tab.notebook.pack(expand=True, fill='both')
            return instance_tab

    def close_instance(self, idx: int):
        # Delete Pipeline instance and running Process
        assert idx < self.TAB_MAX, f"Invalid index={idx} to close."
        self.log(f'Closing instance at position {idx}.')
        try:
            self.process_list[idx].terminate()
            self.process_list[idx] = None
        except AttributeError:
            pass
        # del self.instance_list[idx]
        self.instance_list[idx] = None
        # Delete tab
        self.tab_list[idx].destroy()
        self.tab_list[idx] = None

    def instance_monitor(self):
        assert len(self.instance_list) == len(self.tab_list)
        try:
            while True:
                msg = self.queue.get_nowait()
                print(msg)
                # idx = msg.idx
                idx = msg['idx']
                if msg['is_finished']:
                    if self.status_list[idx] == 'running':
                        self.log(f'Instance @ {idx} finished.')
                        self.status_list[idx] = 'finished'
                        self.tab_list[idx].notebook.run_tab.ss_button['text'] = 'Finished'
                        self.tab_list[idx].notebook.run_tab.ss_button.config(state=tk.DISABLED)
                        if msg['cm']:
                            # Automatically create qc tab
                            # self.create_instance(
                            #     qc_param={'cm_obj': self.instance_list[idx].run_instance.cmobj_path},
                            #     idx=idx
                            # )
                            pass
                elif msg['is_running']:
                    if self.status_list[idx] == 'idle':
                        self.status_list[idx] = 'running'
                        print(f'Process {idx} starts to run.')
                        self.tab_list[idx].notebook.run_tab.ss_button['text'] = 'Stop'
                        self.tab_list[idx].notebook.run_tab.ss_button.config(state=tk.NORMAL)
                else:
                    raise NotImplementedError(f"Invalid message received: {msg}")
                    self.tab_list[idx].notebook.run_tab.ss_button['text'] = 'Start'
                    self.tab_list[idx].notebook.run_tab.ss_button.config(state=tk.NORMAL)
        except queue.Empty:
            pass
        finally:
            self.root.after(1000, self.instance_monitor)

    def log_monitor(self):
        """Log messages sent from other processes through log_queue"""
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log(msg)
        except queue.Empty:
            pass
        finally:
            self.root.after(1000, self.log_monitor)

    def run_instance(self, idx):
        # Stop process
        if self.status_list[idx] == 'running':
            # self.process_list[idx].kill()
            try:
                os.kill(self.process_list[idx].pid, signal.SIGINT)
                self.log(f'Terminating instance @ {idx}.')
                self.process_list[idx] = None
                self.status_list[idx] = 'idle'
                self.tab_list[idx].notebook.run_tab.ss_button['text'] = 'Start'
            except Exception as e:
                self.log(f"Received error: {e}")
            return
        if self.status_list[idx] == 'finished':
            self.log(f'Instance @ {idx} already finished.')
            return
        # check running ok
        assert self.status_list[idx] == 'idle'
        status, msg = self.instance_list[idx].run_instance.ready()
        if not status:
            self.log(f'Instance not ready for running, message: {msg}')
            raise Warning("Instance not ready")
        self.log(f'Running instance @ {idx}')
        p = Process(target=self.instance_list[idx].run_instance.run, args=())
        p.start()
        self.process_list[idx] = p

    def show_params(self, index: int):
        """
        Show a dialog to adjust parameters for the instance at index position
        Args:
            index: Instance index
        Returns:

        """
        param_dict = self.instance_list[index].run_instance.params_dict
        dialog = tk.Toplevel(self.root)
        dialog.title("Param Adjustment")
        self.center(dialog, 800, 600)

        # Add fields for parameters (e.g., tab name)
        # Add checkboxes
        def on_checkbox_change(idx, var):
            value = var.get()
            param_dict['run'][idx] = True if value else False

        checkbox_frame = tk.Frame(dialog)
        checkbox_frame.pack()
        run_1 = tk.BooleanVar(value=param_dict['run'][0])
        run_2 = tk.BooleanVar(value=param_dict['run'][1])
        run_3 = tk.BooleanVar(value=param_dict['run'][2])
        tk.Label(checkbox_frame, text="Run:").pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(checkbox_frame, text="Crop", variable=run_1,
                       command=lambda n=0, v=run_1: on_checkbox_change(n, v)).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(checkbox_frame, text="Stabilizer", variable=run_2,
                       command=lambda n=1, v=run_2: on_checkbox_change(n, v)).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(checkbox_frame, text="CaImAn", variable=run_3,
                       command=lambda n=2, v=run_3: on_checkbox_change(n, v)).pack(side=tk.LEFT, padx=5)

        # Add param fields for each checkbox
        notebook = ttk.Notebook(dialog)
        notebook.pack(expand=True, fill='both')

        # Crop param tab
        tab_1 = ttk.Frame(notebook)
        tab_1.pack(expand=True)
        notebook.add(tab_1, text='Crop')

        def on_slider_change(event):
            param_dict['crop'].update(threshold=slider.get())

        # Create a container frame for the label and slider
        container_1 = ttk.Frame(tab_1)
        container_1.pack(expand=True)
        # Threshold Label
        label = ttk.Label(container_1, text=f"Threshold:")
        label.pack(side=tk.LEFT, padx=5)
        # Threshold Slider
        slider = tk.Scale(container_1, from_=0, to=500, orient='horizontal', resolution=10,
                          command=on_slider_change)
        slider.set(param_dict['crop']['threshold'])
        slider.pack(side=tk.LEFT, padx=5)

        # Validate input numbers
        def validate_numbers(num, type_d):
            if num == '':
                return (True, 0) if type_d is int else (True, 0.0)
            else:
                try:
                    return (True, int(num)) if type_d is int else (True, float(num))
                except ValueError:
                    return False, None

        # Stabilizer param tab
        tab_2 = ttk.Frame(notebook)
        tab_2.pack(expand=True)
        notebook.add(tab_2, text='Stabilizer')
        # Stabilizer containers
        for dname, d in param_dict['stabilizer'].items():
            def on_entry_change(value, key):
                status, val = validate_numbers(value, type(d))
                if not status and key != 'Transformation':
                    messagebox.showerror("Invalid Input", "Please enter a valid numeric value.")
                else:
                    param_dict['stabilizer'][key] = val

            container = ttk.Frame(tab_2)
            container.pack(expand=True)
            label = ttk.Label(container, text=dname)
            label.pack(padx=10, side=tk.LEFT)
            if type(d) is str:
                # A selection entry so user can select from a list
                var = tk.StringVar(value='Translation')
                trans_button = tk.Radiobutton(container, text="Translation", variable=var, value="Translation")
                trans_button.pack(padx=10, side=tk.LEFT)
                affine_button = tk.Radiobutton(container, text="Affine", variable=var, value="Affine")
                affine_button.pack(padx=10, side=tk.LEFT)
                trans_button.select()  # Set the Translation Radiobutton as the default selection
                var.trace_add("write",
                              lambda *args, text=var, key=dname: on_entry_change(text.get(), key))
            else:
                # Text entry for numbers
                entry_text = tk.StringVar(value=d)
                entry = ttk.Entry(container, textvariable=entry_text)
                entry.pack(padx=10, side=tk.LEFT)
                entry_text.trace_add("write",
                                     lambda *args, text=entry_text, key=dname: on_entry_change(text.get(), key))

        # CaImAn param tab
        tab_3 = ttk.Frame(notebook)
        tab_3.pack(expand=True)
        notebook.add(tab_3, text='CaImAn')
        # CaImAn containers
        for dname, d in param_dict['caiman'].items():

            container_outer = ttk.Frame(tab_3)
            container_outer.pack(side=tk.LEFT, expand=True, fill='both', padx=5, pady=5)
            for k, v in d.items():
                def on_entry_change(value, key, dname=dname):
                    if type(value) is str:
                        status, val = validate_numbers(value, type(v))
                        if not status:
                            if key not in ['border_nan', 'method_init', 'method_deconvolution']:
                                messagebox.showerror("Invalid Input", "Please enter a valid numeric value.")
                            else:
                                if value == "None":
                                    param_dict['caiman'][dname][key] = None
                                else:
                                    param_dict['caiman'][dname][key] = value
                        # Numbers
                        else:
                            param_dict['caiman'][dname][key] = val
                    elif type(value) is bool:
                        param_dict['caiman'][dname][key] = True
                    elif type(value) is list:
                        status, val = validate_numbers(value[0], type(v))
                        if not status:
                            messagebox.showerror("Invalid Input", "Please enter a valid numeric value.")
                        else:
                            param_dict['caiman'][key] = [int(vi) if vi != '' else 0 for vi in value]
                    else:
                        raise NotImplementedError(f"Unknown type {type(value)} for value {value}.")

                if k == "fnames":
                    continue
                # Create a container frame for the label and entry
                container = ttk.Frame(container_outer)
                container.pack(expand=True)
                # Label
                label = ttk.Label(container, text=f"{k}:")
                label.pack(side=tk.LEFT, padx=5)
                # Entry type 1: strings
                if type(v) is str:
                    if k == 'border_nan':
                        var = tk.StringVar(value=v)

                        container_inner = ttk.Frame(container)
                        container_inner.pack()

                        # Radiobutton for True
                        copy_radio = tk.Radiobutton(container_inner, text="copy", variable=var, value="copy")
                        copy_radio.pack(side=tk.LEFT, padx=5)

                        # Radiobutton for False
                        none_radio = tk.Radiobutton(container_inner, text="None", variable=var, value="None")
                        none_radio.pack(side=tk.LEFT, padx=5)

                        var.trace_add("write",
                                      lambda *args, text=var, key=k: on_entry_change(text.get(), key))
                    elif k == 'method_init':
                        pass
                    else:  # "method_deconvolution"
                        pass
                # Entry type 2: True / False
                elif type(v) is bool:
                    var = tk.StringVar(value="True" if v else "False")

                    container_inner = ttk.Frame(container)
                    container_inner.pack()

                    # Radiobutton for True
                    true_radio = tk.Radiobutton(container_inner, text="True", variable=var, value="True")
                    true_radio.pack(side=tk.LEFT, padx=5)

                    # Radiobutton for False
                    false_radio = tk.Radiobutton(container_inner, text="False", variable=var, value="False")
                    false_radio.pack(side=tk.LEFT, padx=5)

                    var.trace_add("write",
                                  lambda *args, text=var, key=k: on_entry_change(
                                      True if text.get() == "True" else False, key))
                # Entry type 3: list[int, int]
                elif type(v) is list:
                    var = tk.StringVar(value=v[0])
                    container_inner = ttk.Frame(container)
                    container_inner.pack()
                    entry_1 = ttk.Entry(container_inner, textvariable=var, width=10)
                    entry_1.pack(side=tk.LEFT, padx=5)
                    entry_2 = ttk.Entry(container_inner, textvariable=var, width=10)
                    entry_2.pack(side=tk.LEFT, padx=5)
                    var.trace_add("write",
                                  lambda *args, text=var, key=k: on_entry_change([text.get(), text.get()], key))
                # Entry type 4: numbers
                else:
                    entry_text = tk.StringVar(value=v)
                    entry = ttk.Entry(container, textvariable=entry_text)
                    entry.pack(side=tk.LEFT, padx=5)
                    entry_text.trace_add("write",
                                         lambda *args, text=entry_text, key=k: on_entry_change(text.get(), key))

        # Function to handle 'OK' button click
        def on_ok():
            if param_dict['run'][2]:
                # Handle caiman fnames
                fname = param_dict['input_path'][:-4]
                if param_dict['run'][0]:
                    fname += '_crop'
                if param_dict['run'][1]:
                    fname += '_stab'
                fname = fname + '.tif'
                self.log(f'expect fname: {fname}')
                param_dict['caiman']['mc_dict']['fnames'] = [fname]
            status, msg = self.create_instance(run_params=param_dict)
            if not status:
                self.log(f'Instance creation failed: {msg}')
                tkinter.messagebox.showerror("Error", f"Instance creation failed: {msg}")
                return
            dialog.destroy()
            self.log(f'New instance created and added to position {msg}.')

        # OK and Cancel buttons
        tk.Button(dialog, text="OK", command=on_ok).pack(side="left", padx=10, pady=10)
        tk.Button(dialog, text="Cancel", command=dialog.destroy).pack(side="right", padx=10, pady=10)

    def setup(self):
        def on_close():
            if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
                self.root.destroy()
                self.log('GUI closed.')

        self.root.protocol("WM_DELETE_WINDOW", on_close)

    def gui(self):
        try:
            self.instance_monitor()
            self.log_monitor()
            self.root.mainloop()
        except Exception as e:
            self.logger.error(e)


if __name__ == '__main__':
    gui = GUI()
    gui.gui()
