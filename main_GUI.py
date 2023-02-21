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
        # TODO: enable selection
        section_list = ['Crop', 'Stabilizer', 'CaImAn', 'peak caller']
        select_label = 'Which part you want to start with? Once it is started it will not pause between sections.'
        section_start = st.selectbox(select_label, section_list, key='start')
        if section_start == 'peak caller':
            s1_visible = s2_visible = s3_visible = False
        elif section_start == 'CaImAn':
            s1_visible = s2_visible = False
        elif section_start == 'Stabilizer':
            s1_visible = False
        else:
            pass
        # TODO: add a file selector
        pass
        # TODO: add stabilizer parameter inputs
        pass
        # TODO: add output selector
        pass
        # TODO: add update to pipeline's arguments.
        pass
    # section 1
    s1 = st.empty()
    if s1_visible:
        with s1.container():
            st.header('Auto crop')
            # TODO: add progress bar
            pass
            # TODO: modify max index by input file
            crop_slice_idx = st.slider('crop_idx', 0, 100)
            # TODO: add QC windows
            pass

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
