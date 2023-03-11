# import streamlit as st
# import numpy as np
# import pandas as pd
# import time
# import os
# import src.src_pipeline as pipe
# import tifffile
#
#
# def main():
#     pipeline = pipe.Pipeline()
#     s0_visible = s1_visible = s2_visible = s3_visible = True
#     st.title("Pipeline GUI Interface")
#
#     with st.container():
#         st.subheader('Parameter Box')
#         # Enable selection of starting section
#         section_list = ['Crop', 'Stabilizer', 'CaImAn', 'peak caller']
#         select_label = 'Which part you want to start with? Once it is started it will not pause between sections.'
#         section_start = st.selectbox(select_label, section_list)
#         if section_start == 'peak caller':
#             s0_visible = s1_visible = s2_visible = False
#         elif section_start == 'CaImAn':
#             s0_visible = s1_visible = False
#         elif section_start == 'Stabilizer':
#             s0_visible = False
#
#         # TODO: add an input file selector
#         pass
#         raw_input = st.text_input('Please provide path to your input. If you have multiple images to process at the '
#                                   'same time, please provide path to their folder (they should reside in the same '
#                                   'folder. All .tif files will be selected.)',
#                                   os.path.abspath(__file__)[:-len(os.path.basename(__file__))])
#         if not os.path.exists(raw_input):
#             st.error('path not exist.')
#         if os.path.isdir(raw_input):
#             # TODO: add input support for other format, such as .obj pickle file
#             input_list = [f for f in os.listdir(raw_input) if f[-4:] == '.tif']
#             pipeline.update(input_root=raw_input, input_list=input_list)
#         else:
#             pipeline.update(input_list=[raw_input])
#         # TODO: add output selector (a folder to place everything, aka work_dir)
#         wd_dir = st.text_input('Please provide path to your output folder.')
#         pipeline.update(work_dir=wd_dir)
#
#         # Crop Parameter Container
#         crop_container = st.empty()
#         if s0_visible:
#             with crop_container.container():
#                 margin = st.slider('Margin in px to crop the image:', 0, 500, 200, 25)
#                 pipeline.update(margin=margin)
#
#         # ImageJ Parameter Container
#         ij_container = st.empty()
#         if s1_visible:
#             with ij_container.container():
#                 # TODO: add stabilizer parameter inputs
#                 ijp = os.path.join(os.path.abspath(__file__)[:-len(os.path.basename(__file__))], 'Fiji.app')
#                 pipeline.update(ijp=ijp)
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     s1_transform = st.selectbox('Stabilizer Transformation', ['Translation', 'Affine'])
#                 with col2:
#                     s1_maxpl = st.number_input('ImageJ stabilizer parameter - MAX_Pyramid_level. Default to be 1.0.',
#                                                0.01, 1.0, 1.0, 0.05)
#                 with col3:
#                     s1_upco = st.number_input('ImageJ stabilizer parameter - update_coefficient. Default to 0.90.',
#                                               0.01, 1.0, 0.9, 0.05)
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     s1_maxiter = st.number_input('ImageJ stabilizer parameter - MAX_iteration. Default to 200.', 1, 500,
#                                                  200, 10)
#                 with col2:
#                     s1_errtol = st.number_input('ImageJ stabilizer parameter - error_rolerance. Default to 1E-7.', 1E-9,
#                                                 1E-5, value=1E-7, format='%f')
#                 pipeline.update(s1_params=[s1_transform, s1_maxpl, s1_upco, s1_maxiter, s1_errtol])
#
#     # section 0 - Segmentation and Cropping
#     s0 = st.empty()
#     if s0_visible and not pipeline.done_s0:
#         with s0.container():
#             st.header('Auto crop')
#             # TODO: a possible future improvement: for multiple input files, select which to QC
#             s0_idx = st.selectbox('Which image you want to examine?', pipeline.input_list, key='s0_idx')
#             # TODO: add QC windows
#             try:
#                 QC_in_0 = os.path.join(pipeline.input_root, s0_idx)
#                 QC_crop = os.path.join(wd_dir, s0_idx.removesuffix('.tif')+'_crop.tif')
#             except TypeError:
#                 st.error(f'Please specify input and output folders first.')
#                 QC_in_0 = 'NULL'
#                 QC_crop = 'NULL'
#             try:
#                 f_in_0 = tifffile.imread(QC_in_0)
#                 f_crop = tifffile.imread(QC_crop)
#                 crop_idx = st.slider('Which slice you want to examine?', 0, f_in_0.shape[0])
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.image(f_in_0[crop_idx])
#                 with col2:
#                     st.image(f_crop[crop_idx])
#             except Exception as e:
#                 st.error(f'path {QC_in_0} or {QC_crop} not exist.')
#
#     # section 2: ImageJ Stabilizer
#     s1 = st.empty()
#     if s1_visible:
#         with s1.container():
#             st.header('ImageJ Stabilizer')
#             # TODO: add QC windows
#             # TODO: a possible future improvement: for multiple input files, select which to QC
#             if section_start == 'Crop':
#                 s1_idx = st.selectbox('Which image you want to examine?', pipeline.imm1_list, key='s1_idx')
#             else:
#                 s1_idx = st.selectbox('Which image you want to examine?', pipeline.input_list, key='s1_idx')
#             # TODO: add QC windows
#             try:
#                 if section_start == 'Crop':
#                     QC_in_1 = os.path.join(pipeline.work_dir, s1_idx)
#                     QC_stab = os.path.join(wd_dir, s1_idx.removesuffix('.tif') + '_stab.tif')
#                 else:
#                     QC_in_1 = os.path.join(pipeline.input_root, s1_idx)
#                     QC_stab = os.path.join(wd_dir, s1_idx.removesuffix('.tif') + '_stab.tif')
#             except TypeError:
#                 st.error(f'Please specify input and output folders first.')
#                 QC_in_1 = 'NULL'
#                 QC_stab = 'NULL'
#             try:
#                 f_in_1 = tifffile.imread(QC_in_1)
#                 f_stab = tifffile.imread(QC_stab)
#                 stab_idx = st.slider('Which slice you want to examine?', 0, f_in_1.shape[0])
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.image(f_in_1[stab_idx])
#                 with col2:
#                     st.image(f_stab[stab_idx])
#             except Exception as e:
#                 st.error(f'path {QC_in_1} or {QC_stab} not exist.')
#
#     # section 3: CaImAn
#     s2 = st.empty()
#     if s2_visible:
#         with s2.container():
#             st.header('CaImAn')
#             # TODO: add QC control window
#
#     # section 4: peak caller
#     pass
#
#
# if __name__ == '__main__':
#     main()
