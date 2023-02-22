import streamlit as st
import numpy as np
import pandas as pd
import time
import src.src_pipeline as pipe


def main():
    pipeline = pipe.Pipeline()
    s1_visible = s2_visible = s3_visible = s4_visible = True
    st.title("Pipeline GUI Interface")

    with st.container():
        st.subheader('Parameter Box')
        # Enable selection of starting section
        section_list = ['Crop', 'Stabilizer', 'CaImAn', 'peak caller']
        select_label = 'Which part you want to start with? Once it is started it will not pause between sections.'
        section_start = st.selectbox(select_label, section_list)
        if section_start == 'peak caller':
            s1_visible = s2_visible = s3_visible = False
        elif section_start == 'CaImAn':
            s1_visible = s2_visible = False
        elif section_start == 'Stabilizer':
            s1_visible = False

        # TODO: add an input file selector
        pass

        # TODO: add stabilizer parameter inputs
        ijp = ''  # TODO
        col1, col2, col3 = st.columns(3)
        with col1:
            s1_transform = st.selectbox('Stabilizer Transformation', ['Translation', 'Affine'])
        with col2:
            s1_maxpl = st.number_input('ImageJ stabilizer parameter - MAX_Pyramid_level. You have to specify '
                                       '-ij_param to use it. Default to be 1.0.', 0.01, 1.0, 1.0, 0.05)
        with col3:
            s1_upco = st.number_input('ImageJ stabilizer parameter - update_coefficient. You have to specify '
                                      '-ij_param to use it. Default to 0.90.', 0.01, 1.0, 0.9, 0.05)
        col1, col2 = st.columns(2)
        with col1:
            s1_maxiter = st.number_input('ImageJ stabilizer parameter - MAX_iteration. You have to specify -ij_param '
                                         'to use it. Default to 200.', 1, 500, 200, 10)
        with col2:
            s1_errtol = st.number_input('ImageJ stabilizer parameter - error_rolerance. You have to specify -ij_param '
                                        'to use it. Default to 1E-7.', 1E-9, 1E-5, value=1E-7, format='%f')
        s1_params = [s1_transform, s1_maxpl, s1_upco, s1_maxiter, s1_errtol]
        pipeline.update(s1_params=s1_params)
        # TODO: add output selector (a folder to place everything, aka work_dir)
        pass
        # TODO: add update to pipeline's arguments.
        pass
    # section 1
    s1 = st.empty()
    if s1_visible:
        with s1.container():
            st.header('Auto crop')
            # TODO: add progress bar
            # st.progress(, 'Cropping running')
            crop_idx = st.slider('Which slice you want to examine?', 0, 100)  # TODO: modify max index by input file
            # TODO: a possible future improvement: for multiple input files, select which to QC
            # TODO: add QC windows
            col1, col2 = st.columns(2)
            with col1:
                st.image()
            with col2:
                st.image()

    # section 2: ImageJ Stabilizer
    s2 = st.empty()
    if s2_visible:
        with s2.container():
            st.header('ImageJ Stabilizer')
            # TODO: add QC windows
            pass

    # section 3: CaImAn
    s3 = st.empty()
    if s3_visible:
        with s3.container():
            st.header('CaImAn')
            # TODO: add QC control window

    # section 4: peak caller
    pass


if __name__ == '__main__':
    main()
