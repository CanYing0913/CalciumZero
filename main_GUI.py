from pathlib import Path
import PySimpleGUI as sg
from multiprocessing import Process
from tifffile import imread
from cv2 import imwrite, resize
import src.src_pipeline as pipe

section_start = 'Crop'
start_0 = start_1 = start_2 = False


def load_config():
    SETTINGS_PATH = Path.cwd()
    settings = sg.UserSettings(
        path=str(SETTINGS_PATH), filename='config.ini', use_config_file=True, convert_bools_and_none=True
    )
    return settings


def init_sg(settings):
    win_width, win_height = sg.Window.get_screen_size()
    win_width = int(win_width * float(settings['GUI']['ratio']))
    win_height = int(win_height * float(settings['GUI']['ratio']))
    theme = settings['GUI']['theme']
    font_family, font_size = settings['GUI']['font_family'], int(settings['GUI']['font_size'])
    sg.theme(theme)

    row_meta = [
        [sg.Text('Welcome to CaImAn pipeline GUI', font='italic 16 bold', justification='center')],
        [sg.Text('Which section you want to run?', justification='center')],
        [
            sg.Checkbox('Crop', enable_events=True, key='-META-START-crop-'),
            sg.Checkbox('Stabilizer', enable_events=True, key='-META-START-stabilizer-'),
            sg.Checkbox('CaImAn', enable_events=True, key='-META-START-caiman-'),
            sg.Checkbox('Peak Caller', enable_events=True, key='-META-START-peakcaller-')
        ],
        [
            sg.Text('Input File(s):'),
            sg.Input(size=(10, 2), key='-META-FIN-', enable_events=True),
            sg.FilesBrowse('Choose input', file_types=[('TIFF', '*.tif'), ], key='-META-FIN-SELECT-')
        ],
        [
            sg.Text('Output Folder:'),
            sg.Input(size=(10, 2), key='-META-FOUT-', enable_events=True),
            sg.FolderBrowse('Choose output')
        ],
        [
            sg.Button('start', key='-META-start-', enable_events=True),
            sg.Button('stop', key='-META-stop-', enable_events=True),
            sg.Text('# of processes:'),
            sg.Input('2', key='-META-process-', enable_events=True)
        ]
    ]
    opts_s0 = [
        [sg.Text('Crop Image')],
        [sg.Text('Choose margin in px:')],
        [sg.Slider(range=(0, 1000), default_value=200, resolution=10, key='-OPTION-margin-', enable_events=True,
                   orientation='h')]
    ]
    opts_s1 = [
        [sg.Text('ImageJ Stabilizer')],
        [
            sg.Text('ImageJ path:'),
            sg.Input(size=(10, 2), default_text=f'{Path(__file__).parent.joinpath("Fiji.app")}',
                     key='-OPTION-ijp-', enable_events=True),
            sg.FolderBrowse('Browse')
        ],
        [
            sg.Text('Transformation'),
            sg.Listbox(('Translation', 'Affine'), default_values=['Translation'],
                       key='-OPTION-ij-transform-', enable_events=True)
        ],
        [
            sg.Text('MAX_Pyramid_level [0.0-1.0]'),
            sg.Input(default_text='1.0', key='-OPTION-ij-maxpl-', size=5, justification='center', enable_events=True)
        ],
        [
            sg.Text('update_coefficient [0.0-1.0]'),
            sg.Input(default_text='0.90', key='-OPTION-ij-upco-', size=5, justification='center', enable_events=True)
        ],
        [
            sg.Text('MAX_iteration [1-500]'),
            sg.Input(default_text='200', key='-OPTION-ij-maxiter-', size=5, justification='center', enable_events=True)
        ],
        [
            sg.Text('error_rolerance [1E-7]'),
            sg.Input(default_text='1E-7', key='-OPTION-ij-errtol-',
                     size=5, justification='center', disabled=True, enable_events=True)
        ]
    ]
    row1 = [
        sg.Column(row_meta, size=(400, 250), justification='center'),
        sg.VSeparator(),
        sg.Column(opts_s0, size=(275, 175), justification='center'),
        sg.VSeparator(),
        sg.Column(opts_s1, size=(325, 250), justification='center')
    ]
    QC_s0 = [
        sg.Column([
            [sg.Text('QC for crop')],
            [sg.Listbox([], key='-QC-S0-img-select-', enable_events=True)],
            [sg.Text('Which slice you want to examine?')],
            [sg.Slider((0, 200), key='-QC-S0-img-slider-', orientation='h', enable_events=True)]
        ]),
        sg.VSeparator(),
        sg.Column([[sg.Image(key='-QC-S0-img-raw-')]]),
        sg.VSeparator(),
        sg.Column([[sg.Image(key='-QC-S0-img-s0-')]])
    ]
    QC_s1 = [
        sg.Column([
            [sg.Text('QC for stabilizer')],
            [sg.Listbox([], key='-QC-S1-img-select-', enable_events=True)],
            [sg.Text('Which slice you want to examine?')],
            [sg.Slider((0, 200), key='-QC-S1-img-slider-', orientation='h', enable_events=True)]
        ]),
        sg.VSeparator(),
        sg.Column([[sg.Image(key='-QC-S1-img-raw-')]]),
        sg.VSeparator(),
        sg.Column([[sg.Image(key='-QC-S1-img-s0-')]])
    ]
    QC_tb = sg.TabGroup(
        [[
            sg.Tab('Crop', [QC_s0]),
            sg.Tab('Stabilizer', [QC_s1]),
        ]], key='-QC-window-'
    )
    row_QC = sg.Column([
        [sg.Text('Quality Checks')],
        [QC_tb],
    ])
    layout = [
        [row1],
        [row_QC]
    ]

    sg.set_options(font=(font_family, font_size))
    window = sg.Window(title='Pipeline', layout=layout, size=(win_width, win_height), font=(font_family, font_size),
                       text_justification='center', element_justification='center', enable_close_attempted_event=True)
    return window


def handle_events(pipe_obj, window, settings):
    # Get variables from config for user-input purpose
    run_th = Process(target=pipe_obj.run, args=())
    try:
        while True:
            event, values = window.read()
            # print(event)
            # handle exit
            if event == '-WINDOW CLOSE ATTEMPTED-':
                if sg.popup_yes_no('Do you want to exit?') == 'Yes':
                    break
                else:
                    continue
            if '-META-' in event:
                if event == '-META-FIN-' or event == '-META-FOUT-':
                    if event == '-META-FIN-':
                        if values['-META-FIN-']:
                            FIN = values['-META-FIN-'].split(';')
                            input_root = FIN[0][:-len(Path(FIN[0]).name)]
                            FIN = [Path(f).name for f in FIN]
                            window['-QC-S0-img-select-'].update(values=FIN)
                            pipe_obj.update(input_root=input_root, input_list=FIN)
                            # print(pipe_obj.input_root, pipe_obj.input_list)
                    else:  # -FOUT-
                        pipe_obj.update(work_dir=values['-META-FOUT-'])
                        pipe_obj.update(s1_root=values['-META-FOUT-'])  # for testing
                        # print(pipe_obj.work_dir)
                elif '-process-' in event:
                    pipe_obj.update(process=int(values[event]))
                # handle starting section
                elif '-META-START-' in event:
                    if 'crop' in event:
                        pipe_obj.update(do_s0=values[event])
                    elif 'stabilizer' in event:
                        pipe_obj.update(do_s1=values[event])
                    elif 'caiman' in event:
                        pipe_obj.update(do_s2=values[event])
                    elif 'peakcaller' in event:
                        pipe_obj.update(do_s3=values[event])
                    else:
                        sg.popup_error('NotImplementedError')
                    # at the end, sanitize input file types
                    do_s0, do_s1, do_s2, do_s3 = pipe_obj.do_s0, pipe_obj.do_s1, pipe_obj.do_s2, pipe_obj.do_s3
                    if do_s0 or do_s1 or do_s2:
                        window('-META-FIN-SELECT-').update(file_types=[('TIFF', '*.tif'), ])
                    elif do_s3:
                        window('-META-FIN-SELECT-').update(file_types=[('TIFF', '*.cmnobj'), ('HDF5 file', '*.hdf5')])

                elif event == '-META-start-':
                    status, msg = pipe_obj.ready()
                    if status and not run_th.is_alive():
                        run_th.start()
                        run_th = Process(target=pipe_obj.run, args=())
                    else:
                        sg.popup_error(msg + f". process live status: {str(run_th.is_alive())}")
                elif event == '-META-stop-':
                    if run_th.is_alive():
                        run_th.kill()
                    else:
                        sg.popup_error(f'Execution not running - Cannot stop')

                else:
                    sg.popup_error('NotImplementedError')
                    # raise NotImplementedError
            # handle options
            if '-OPTION-' in event:
                if '-margin-' in event:
                    pipe_obj.update(margin=values[event])
                if '-ijp-' in event:
                    pipe_obj.update(ijp=values[event])
                if '-ij-' in event:
                    if '-transform-' in event:
                        idx = 0
                    elif '-maxpl-' in event:
                        idx = 1
                    elif '-upco-' in event:
                        idx = 2
                    elif '-maxiter-' in event:
                        idx = 3
                    elif '-errtol-' in event:
                        idx = 4
                    else:
                        sg.popup_error('NotImplementedError')
                        raise NotImplementedError
                    pipe_obj.s1_params[idx] = values[event]
            if '-QC-' in event:
                if '-S0-img-select-' in event:
                    filename = values[event][0]
                    filename_raw = str(Path(pipe_obj.input_root).joinpath(filename))
                    if pipe_obj.work_dir == '':
                        sg.popup_error('Please select output folder first.')
                        continue
                    filename_s0 = str(Path(pipe_obj.work_dir).joinpath(Path(filename).stem + '_crop.tif'))
                    QCimage_raw = imread(filename_raw)
                    QCimage_s0 = imread(filename_s0)
                    raw_path = str(Path(pipe_obj.cache).joinpath(settings['QC']['s0_input']))
                    s0_path = str(Path(pipe_obj.cache).joinpath(settings['QC']['s0_output']))
                    pipe_obj.update(QCimage_s0_raw=QCimage_raw, QCimage_s0=QCimage_s0)
                    s, w, h = QCimage_raw.shape
                    s_, w_, h_ = QCimage_s0.shape
                    new_w = int(200 * h / w)
                    new_w_ = int(200 * h_ / w_)
                    QCimage_raw = resize(QCimage_raw[0], (new_w, 200))
                    QCimage_s0 = resize(QCimage_s0[0], (new_w_, 200))
                    imwrite(raw_path, QCimage_raw)
                    imwrite(s0_path, QCimage_s0)
                    window['-QC-S0-img-slider-'].update(range=(0, s - 1))
                    window['-QC-S0-img-raw-'].update(filename=raw_path)
                    window['-QC-S0-img-s0-'].update(filename=s0_path)

                elif '-S0-img-slider-' in event:
                    slice = int(values[event])
                    if pipe_obj.QCimage_s0_raw is not None:
                        raw_path = str(Path(pipe_obj.cache).joinpath(settings['QC']['s0_input']))
                        s0_path = str(Path(pipe_obj.cache).joinpath(settings['QC']['s0_output']))
                        s, w, h = pipe_obj.QCimage_s0_raw.shape
                        s_, w_, h_ = pipe_obj.QCimage_s0.shape
                        new_w = int(200 * h / w)
                        new_w_ = int(200 * h_ / w_)
                        QCimage_raw = resize(pipe_obj.QCimage_s0_raw[slice], (new_w, 200))
                        QCimage_s0 = resize(pipe_obj.QCimage_s0[slice], (new_w_, 200))
                        imwrite(raw_path, QCimage_raw)
                        imwrite(s0_path, QCimage_s0)
                        window['-QC-S0-img-raw-'].update(filename=raw_path)
                        window['-QC-S0-img-s0-'].update(filename=s0_path)
                elif event == '-QC-select-':
                    pass
                else:
                    sg.popup_error('NotImplementError')
    finally:
        window.close()


def main():
    # Initialize pipeline and GUI
    pipe_obj = pipe.Pipeline()
    settings = load_config()
    window = init_sg(settings)
    handle_events(pipe_obj, window, settings)


if __name__ == '__main__':
    main()
