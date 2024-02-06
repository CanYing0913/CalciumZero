from pathlib import Path
from os import path
import PySimpleGUI as sg
from PySimpleGUI import Column
from multiprocessing import Process
from threading import Thread
from tifffile import imread
from cv2 import imwrite, resize
from matplotlib import pyplot as plt
import numpy as np
from src.src_pipeline import Pipeline


def load_config():
    SETTINGS_PATH = path.dirname(path.abspath(__file__))
    settings = sg.UserSettings(
        path=SETTINGS_PATH, filename='config.ini', use_config_file=True, convert_bools_and_none=True
    )
    return settings


def init_sg(settings):
    # Set window and theme based on config and user display
    win_width, win_height = sg.Window.get_screen_size()
    scale_w = win_width / 1920
    scale_h = win_height / 1080
    win_width = int(win_width * float(settings['GUI']['ratio']))
    win_height = int(win_height * float(settings['GUI']['ratio']))
    theme = settings['GUI']['theme']
    font_family, font_size = settings['GUI']['font_family'], int(settings['GUI']['font_size'])
    font_size = int(scale_w * font_size)
    sg.theme(theme)

    # GUI components
    row_meta = [
        [sg.Text('Welcome to CalciumZero GUI', font='italic 16 bold', justification='center')],
        [
            sg.Checkbox('Crop', enable_events=True, key='-META-START-crop-'),
            sg.Checkbox('Stabilizer', enable_events=True, key='-META-START-stabilizer-'),
            sg.Checkbox('CaImAn', enable_events=True, key='-META-START-caiman-'),
            sg.Checkbox('Peak Caller', enable_events=True, key='-META-START-peakcaller-')
        ],
        [
            sg.Text('Input File(s):'),
            sg.Input(size=(int(10 * scale_w), 2), key='-META-FIN-', enable_events=True, readonly=True),
            sg.FilesBrowse('Choose input', file_types=[('TIFF', '*.tif'), ('cmobj', '*.cmobj')],
                           key='-META-FIN-SELECT-')
        ],
        [
            sg.Text('Output Folder:'),
            sg.Input(size=(int(10 * scale_w), 2), key='-META-FOUT-', enable_events=True, readonly=True),
            sg.FolderBrowse('Choose output')
        ],
        [
            sg.Button('start', key='-META-start-', enable_events=True),
            sg.Button('stop', key='-META-stop-', enable_events=True),
            sg.Button('QC', key='-META-QC-', enable_events=True),
            sg.Text('# of processes:'),
            sg.Input('2', key='-META-process-', enable_events=True, size=int(10 * scale_w))
        ]
    ]
    opts_s0 = [
        [sg.Text('Crop Image')],
        [sg.Text('Margin(pixel)')],
        [sg.Slider(range=(0, 1000), default_value=200, resolution=10, key='-OPTION-margin-', enable_events=True,
                   orientation='h', size=(int(10 * scale_w), int(20 * scale_h)))]
    ]
    opts_s1 = [
        [sg.Text('ImageJ Stabilizer')],
        [
            sg.Text('ImageJ:'),
            sg.Input(size=(int(10 * scale_w), 2), default_text=f'{Path(__file__).parent.joinpath("Fiji.app")}',
                     key='-OPTION-ijp-', enable_events=True, font=('Comic Sans MS', f'{int(font_size * scale_w)}')),
            sg.FolderBrowse('Locate')
        ],
        [
            sg.Text('Transform'),
            sg.Listbox(('Translation', 'Affine'), default_values=['Translation'],
                       key='-OPTION-ij-transform-', enable_events=True)
        ],
        [
            sg.Text('MAX_Pyramid_level [0.0-1.0]'),
            sg.Input(default_text='1.0', key='-OPTION-ij-maxpl-', size=int(5 * scale_w), justification='center',
                     enable_events=True)
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
    opts_tg = sg.TabGroup(
        [[
            sg.Tab('Crop', opts_s0),
            sg.Tab('Stabilizer', opts_s1)
        ]]
    )
    row1 = [
        Column(row_meta, justification='center', ),  # size=(400, 250), ),
        sg.VSeparator(),
        opts_tg
    ]
    QC_s0 = [
        Column([
            [sg.Text('QC for crop')],
            [sg.Listbox([], key='-QC-S0-img-select-', enable_events=True)],
            [sg.Text('Slice Index')],
            [sg.Slider((0, 200), key='-QC-S0-img-slider-', orientation='h', enable_events=True,
                       size=(int(10 * scale_w), int(20 * scale_h)))]
        ]),
        sg.VSeparator(),
        Column([[sg.Image(key='-QC-S0-img-raw-')]]),
        sg.VSeparator(),
        Column([[sg.Image(key='-QC-S0-img-s0-')]])
    ]
    QC_s1 = [
        Column([
            [sg.Text('QC for stabilizer')],
            [sg.Listbox([], key='-QC-S1-img-select-', enable_events=True)],
            [sg.Text('Which slice you want to examine?')],
            [sg.Slider((0, 200), key='-QC-S1-img-slider-', orientation='h', enable_events=True,
                       size=(int(10 * scale_w), int(20 * scale_h)))]
        ]),
        sg.VSeparator(),
        Column([[sg.Image(key='-QC-S1-img-raw-')]]),
        sg.VSeparator(),
        Column([[sg.Image(key='-QC-S1-img-s0-')]])
    ]
    QC_s2 = [
        sg.Text(),
        Column([
            [
                sg.Listbox([0], default_values=[0], enable_events=True, key='-QC-S2-lb-'),
                sg.Slider((0, 9), 0, resolution=1, orientation='h', enable_events=True, key='-QC-S2-img-slider-',
                          disable_number_display=True),
                sg.Input('0', enable_events=True, key='-QC-S2-slice-', size=5),
                sg.Button('Movie', enable_events=True, key='-QC-S2-movie-')
            ],
            [
                sg.Text('ROI index:'),
                sg.Slider((0, 9), 0, 1, orientation='h', disable_number_display=True, enable_events=True,
                          key='-QC-S2-ROI-slider-'),
                sg.Input('0', enable_events=True, key='-QC-S2-ROI-idx-', size=5),
                sg.Listbox([], enable_events=True, key='-QC-S2-AC-LB-', visible=False),
                sg.Checkbox('AC', enable_events=True, key='-QC-S2-AC-')
            ],
            [sg.Image(key='-QC-S2-img-')]
        ])
    ]
    QC_tb = sg.TabGroup(
        [[
            sg.Tab('Crop', [QC_s0]),
            sg.Tab('Stabilizer', [QC_s1]),
            sg.Tab('CaImAn', [QC_s2]),
        ]], key='-QC-window-'
    )
    row_QC = Column([
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
    qc_s2_movie = Thread(target=pipe_obj.qc_caiman_movie, args=())
    prev_s2qc_fig = plt.figure()
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
                    try:
                        pipe_obj.update(process=int(values[event]))
                    except:
                        sg.popup_error(f'You should type in a number. Aborting.')
                        window['-META-process-'].update(values='')
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
                        window['-META-FIN-SELECT-'].FileTypes = [('TIFF', '*.tif'), ]
                    elif do_s3:
                        window['-META-FIN-SELECT-'].FileTypes = [('caiman obj', '*'), ('HDF5 file', '*.hdf5'), ]
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
                elif event == '-META-QC-':
                    if pipe_obj.caiman_obj is not None:
                        # Display 0-th frame and update slider
                        if len(pipe_obj.caiman_obj.input_files) == 1:
                            window['-QC-S2-img-slider-'].update(range=(0, len(pipe_obj.caiman_obj.estimates.F_dff) - 1))
                        else:
                            num_in = len(pipe_obj.caiman_obj.input_files)
                            slice_per_file = len(pipe_obj.caiman_obj.input_files[0])
                            window['-QC-S2-img-slider-'].update(range=(0, slice_per_file - 1))
                            window['-QC-S2-lb-'].update(values=[x for x in range(num_in)])
                            # window['-QC-S2-lb-'].update(default_values=[0])
                        window['-QC-S2-ROI-slider-'].update(range=(0, pipe_obj.caiman_obj.estimates.A.shape[1]-1))
                        img = pipe_obj.qc_s2(0, 0)
                        imwrite(str(pipe_obj.QCimage_s2), img)
                        window['-QC-S2-img-'].update(filename=pipe_obj.QCimage_s2)
                        window['-QC-S2-AC-LB-'].update(values=pipe_obj.caiman_obj.estimates.idx_components)
                    else:
                        sg.popup_error(f'You need to set input to your SINGLE cmn_obj file!')
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
                    try:
                        QCimage_raw = imread(filename_raw)
                        QCimage_s0 = imread(filename_s0)
                        QCimage_raw = np.array(QCimage_raw, dtype=np.uint8)
                        QCimage_s0 = np.array(QCimage_s0, dtype=np.uint8)
                    except FileNotFoundError:
                        sg.popup_error('No crop file detected.')
                        continue
                    raw_path = str(Path(pipe_obj.cache).joinpath(settings['QC']['s0_input']))
                    s0_path = str(Path(pipe_obj.cache).joinpath(settings['QC']['s0_output']))
                    pipe_obj.update(QCimage_s0_raw=QCimage_raw, QCimage_s0=QCimage_s0)
                    s, w, h = QCimage_raw.shape
                    s_, w_, h_ = QCimage_s0.shape
                    new_w = int(300 * h / w)
                    new_w_ = int(300 * h_ / w_)
                    QCimage_raw = resize(QCimage_raw[0], (new_w, 300))
                    QCimage_s0 = resize(QCimage_s0[0], (new_w_, 300))
                    imwrite(raw_path, QCimage_raw)
                    imwrite(s0_path, QCimage_s0)
                    window['-QC-S0-img-slider-'].update(range=(0, s - 1))
                    window['-QC-S0-img-raw-'].update(filename=raw_path)
                    window['-QC-S0-img-s0-'].update(filename=s0_path)
                    continue
                if '-S0-img-slider-' in event:
                    slice = int(values[event])
                    if pipe_obj.QCimage_s0_raw is not None:
                        raw_path = str(Path(pipe_obj.cache).joinpath(settings['QC']['s0_input']))
                        s0_path = str(Path(pipe_obj.cache).joinpath(settings['QC']['s0_output']))
                        s, w, h = pipe_obj.QCimage_s0_raw.shape
                        s_, w_, h_ = pipe_obj.QCimage_s0.shape
                        new_w = int(300 * h / w)
                        new_w_ = int(300 * h_ / w_)
                        QCimage_raw = resize(pipe_obj.QCimage_s0_raw[slice], (new_w, 300))
                        QCimage_s0 = resize(pipe_obj.QCimage_s0[slice], (new_w_, 300))
                        # if np.max(QCimage_s0) < 127:
                        #     QCimage_s0 = QCimage_s0 * 255 // np.max(QCimage_s0)
                        imwrite(raw_path, QCimage_raw)
                        imwrite(s0_path, QCimage_s0)
                        window['-QC-S0-img-raw-'].update(filename=raw_path)
                        window['-QC-S0-img-s0-'].update(filename=s0_path)
                    continue
                # CaImAn QC
                if '-QC-S2-' in event:
                    if event == '-QC-S2-movie-':
                        if qc_s2_movie.is_alive():
                            sg.popup_quick_message('Movie is already running. Aborting.')
                            continue
                        qc_s2_movie = Thread(target=pipe_obj.qc_caiman_movie, args=())
                        qc_s2_movie.start()
                        continue
                    if event == '-QC-S2-img-slider-' or event == '-QC-S2-slice-':
                        try:
                            slice = int(values[event])
                        except:
                            sg.popup_error(f'You need to type in a number. Aborting.')
                            window['-QC-S2-slice-'].update('')
                            continue
                    else:
                        slice = int(values['-QC-S2-img-slider-'])
                    if event == '-QC-S2-ROI-slice-' or event == '-QC-S2-ROI-slider-':
                        try:
                            ROI_idx = int(values[event])
                            if plt.fignum_exists(prev_s2qc_fig.number):
                                plt.close(prev_s2qc_fig)
                                prev_s2qc_fig = plt.figure()
                            plt.plot(pipe_obj.caiman_obj.estimates.C[ROI_idx])
                            plt.show(block=False)
                            plt.pause(0.001)
                        except:
                            sg.popup_error(f'You need to type in a number. Aborting.')
                            window['-QC-S2-slice-'].update('')
                            continue
                    else:
                        ROI_idx = int(values['-QC-S2-ROI-slider-'])

                    # accepted components
                    if event == '-QC-S2-AC-':
                        if values['-QC-S2-AC-']:
                            window['-QC-S2-ROI-slider-'].update(visible=False)
                            window['-QC-S2-ROI-idx-'].update(visible=False)
                            window['-QC-S2-AC-LB-'].update(visible=True)
                            ROI_idx = 0 if len(values['-QC-S2-AC-LB-']) == 0 else values['-QC-S2-AC-LB-'][0]
                        else:
                            window['-QC-S2-ROI-slider-'].update(visible=True)
                            window['-QC-S2-ROI-idx-'].update(visible=True)
                            window['-QC-S2-AC-LB-'].update(visible=False)
                        sg.popup_error(f'Please select slice index.')
                        continue
                    elif event == '-QC-S2-AC-LB-':
                        ROI_idx = 0 if len(values['-QC-S2-AC-LB-']) == 0 else values['-QC-S2-AC-LB-'][0]
                    if pipe_obj.caiman_obj is not None:
                        if slice >= pipe_obj.caiman_obj.estimates.F_dff.shape[1]:
                            sg.popup_error(f'{slice} is greater than possible. Index back to 0.')
                            slice = 0
                        if not values['-QC-S2-lb-']:
                            sg.popup_error('Select image index first!')
                            continue
                        slice_idx = slice + values['-QC-S2-lb-'][0] * len(pipe_obj.caiman_obj.input_files[0])
                        img = pipe_obj.qc_s2(slice_idx, ROI_idx)
                        imwrite(str(pipe_obj.QCimage_s2), img)
                        window['-QC-S2-img-'].update(filename=pipe_obj.QCimage_s2)
                    window['-QC-S2-slice-'].update(str(slice))
                    window['-QC-S2-img-slider-'].update(slice)
                    window['-QC-S2-ROI-idx-'].update(str(ROI_idx))
                    window['-QC-S2-ROI-slider-'].update(ROI_idx)

    finally:
        window.close()


def main():
    # Initialize pipeline and GUI
    settings = load_config()
    window = init_sg(settings)
    pipe_obj = Pipeline()
    pipe_obj.load_setting(settings)
    handle_events(pipe_obj, window, settings)


if __name__ == '__main__':
    main()
