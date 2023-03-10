from pathlib import Path
import PySimpleGUI as sg
from threading import Thread
from PIL import Image, ImageSequence
import src.src_pipeline as pipe

section_start = 'Crop'
start_0 = start_1 = start_2 = False


def load_config():
    SETTINGS_PATH = Path.cwd()
    settings = sg.UserSettings(
        path=SETTINGS_PATH, filename='config.ini', use_config_file=True, convert_bools_and_none=True
    )
    return settings


def init_sg(settings):
    theme = settings['GUI']['theme']
    font_family, font_size = settings['GUI']['font_family'], int(settings['GUI']['font_size'])
    sg.theme(theme)

    row_meta = [
        [sg.Text('Welcome to CaImAn pipeline GUI', font='italic 16 bold', justification='center')],
        [
            sg.Text('Input File(s):'),
            sg.Input(size=(10, 2), key='-META-FIN-', enable_events=True),
            sg.FilesBrowse('Choose input', file_types=[('TIFF', '*.tif'), ])
        ],
        [
            sg.Text('Output Folder:'),
            sg.Input(size=(10, 2), key='-META-FOUT-', enable_events=True),
            sg.FolderBrowse('Choose output')
        ],
        [
            sg.Text('Which section you want to start with?', justification='center'),
            sg.Listbox(
                ['Crop', 'Stabilizer', 'CaImAn', 'Peak caller'],
                default_values=['Crop'], enable_events=True, key='-META-SECTION-START-LIST-'
            )
        ],
        [sg.Button('start', key='-META-start', enable_events=True)]
        # [sg.Text(f"You choose to start with {section_start}", key='-META-start-msg-', justification='center')]
    ]
    opts_s0 = [
        [sg.Text('Crop Image')],
        [sg.Text('Choose margin in px:')],
        [sg.Slider(range=(0, 1000), default_value=200, resolution=10, key='-OPTION-margin-', enable_events=True,
                   orientation='h')]
    ]
    opts_s1 = [
        [
            sg.Text('ImageJ path:'),
            sg.Input(size=(10, 2), key='-OPTION-ijp-', enable_events=True),
            sg.FolderBrowse('Browse')
        ],
        [sg.Text('ImageJ Stabilizer')],
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
    QC_s0 = [
        # [sg.Text('Crop Image via histogram segmentation')],
        # [sg.Slider(range=(0, 1000), default_value=200, resolution=10, orientation='h', key='-S0-margin-')],
        [sg.Text('QC for crop & segmentation')],
        [sg.Listbox([], key='-S0-img-select-', enable_events=True)],
        [sg.Text('Which slice you want to examine?')],
        [sg.Slider((0, 200), key='-S0-img-slider-', orientation='h', enable_events=True)],
        # [sg.Image(filename_QC0_1)] if filename_QC0_1 else [],
        # [sg.Image(filename_QC0_2)] if filename_QC0_2 else []
    ]
    QC_s1 = [

    ]
    layout = [
        [sg.Column(layout=row_meta, size=(400, 250), justification='center')],
        [
            sg.Column(opts_s0, size=(275, 175), justification='center'),
            sg.VSeparator(),
            sg.Column(opts_s1, size=(325, 225), justification='center'),
        ],
        [
            sg.Column(QC_s0),
            sg.VSeparator(),
            sg.Column(QC_s1)
        ]
    ]

    sg.set_options(font=(font_family, font_size))
    window = sg.Window(title='Pipeline', layout=layout, size=(1280, 720), font=(font_family, font_size),
                       text_justification='center', element_justification='center')
    return window


def main():
    # Initialize pipeline and GUI
    pipe_obj = pipe.Pipeline()
    pipe_obj.update(do_s0=True, do_s1=True)  # testing purpose
    settings = load_config()
    window = init_sg(settings)

    # Get variables from config for user-input purpose
    s1_param = [settings['ImageJ']['transform'],
                float(settings['ImageJ']['maxpl']),
                float(settings['ImageJ']['upco']),
                int(settings['ImageJ']['maxiter']),
                float(settings['ImageJ']['errtol'])]
    try:
        while True:
            event, values = window.read()
            # print(event)
            # handle exit
            if event == sg.WIN_CLOSED:
                # TODO add a confirmation window
                break
            # handle options
            if '-META-' in event:
                if event == '-META-FIN-' or event == '-META-FOUT-':
                    if event == '-META-FIN-':
                        if values['-META-FIN-']:
                            FIN = values['-META-FIN-'].split(';')
                            input_root = FIN[0][:-len(Path(FIN[0]).name)]
                            FIN = [Path(f).name for f in FIN]
                            window['-S0-img-select-'].update(values=FIN)
                            pipe_obj.update(input_root=input_root, input_list=FIN)
                            # print(pipe_obj.input_root, pipe_obj.input_list)
                    else:  # -FOUT-
                        pipe_obj.update(work_dir=values['-META-FOUT-'])
                        pipe_obj.update(s1_root=values['-META-FOUT-'])  # for testing
                        # print(pipe_obj.work_dir)
                # handle starting section
                elif event == '-META-SECTION-START-LIST-':
                    # window['-META-start-msg-'].update(value=f"You choose to start with {values[event][0]}")
                    section_start = values[event][0]
                    # TODO update to pipeline.
                elif event == '-META-start':
                    status, msg = pipe_obj.ready()
                    if status:
                        pipe_obj.run()
                    else:
                        sg.popup_error(msg)
                else:
                    raise NotImplementedError
            # s0 related
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
                        raise NotImplementedError
                    s1_param[idx] = values[event]
                    pipe_obj.update(s1_params=s1_param)
    finally:
        window.close()


if __name__ == '__main__':
    main()
