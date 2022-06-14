import os
import json
import subprocess
import dicom2nifti
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import nibabel as nib
from nipype.interfaces.fsl import BET
from skimage.metrics import structural_similarity as ssim


def get_dicom_paths(in_path, dicoms, subjects):
    if os.path.isfile(dicoms):
        with open(dicoms) as f:
            dicom_paths = json.load(f)
    else:
        overview = pd.read_csv(subjects)
        overview = set(overview['Individual'].tolist())

        mprage = ['MP-RAGE', 'MPRAGE']
        scantypes = glob(f'{in_path}/*/*/*/*/')

        dicom_paths = [x for x in scantypes if [
            ele for ele in mprage if(ele in x.upper())]]
        dicom_paths = [x for x in dicom_paths if x.split('/')[4] in overview]

        with open(dicoms, 'w+') as f:
            json.dump(dicom_paths, f)

    return dicom_paths


def dicom_to_nifti(dicoms_15T, dicoms_3T, output_folder, subjects):
    overview = pd.read_csv(subjects)
    overview = overview[['Individual', 'Field Strength', 'Scan date']]
    overview['Scan date'] = pd.to_datetime(
        overview['Scan date']).dt.strftime('%Y-%m-%d')

    common_id = 0
    invalid_dicoms = []

    for i in tqdm(range(len(overview.index)//2 - 1)):
        info_15T = overview.loc[2*i, :]
        info_3T = overview.loc[2*i+1, :]

        matching_patient_15T = [x for x in dicoms_15T if x.split(
            '/')[4] == info_15T['Individual']]
        matching_date_15T = [
            x for x in matching_patient_15T if info_15T['Scan date'] in x.split('/')[6]]

        matching_patient_3T = [x for x in dicoms_3T if x.split(
            '/')[4] == info_3T['Individual']]
        matching_date_3T = [
            x for x in matching_patient_3T if info_3T['Scan date'] in x.split('/')[6]]

        if len(matching_date_15T) > 0 and len(matching_date_3T) > 0:
            sorted_15 = sorted(matching_date_15T, key=lambda x: x.split(
                '/')[5].replace('_', '').replace('-', ''))
            sorted_3 = sorted(matching_date_3T, key=lambda x: x.split(
                '/')[5].replace('_', '').replace('-', ''))

            for j in range(min(len(sorted_15), len(sorted_3))):
                subject_15, scantype_15, date_15 = sorted_15[j].split('/')[4:7]
                scantype_15.replace('_', '').replace('-', '')
                date_15 = date_15.split('_')[0]

                subject_3, scantype_3, date_3 = sorted_3[j].split('/')[4:7]
                scantype_3.replace('_', '').replace('-', '')
                date_3 = date_3.split('_')[0]

                try:
                    dicom2nifti.dicom_series_to_nifti(
                        sorted_15[j],
                        f'{output_folder}/{common_id}_{subject_15}_{scantype_15}_{date_15}_15T',
                        reorient_nifti=True)

                    dicom2nifti.dicom_series_to_nifti(
                        sorted_3[j],
                        f'{output_folder}/{common_id}_{subject_3}_{scantype_3}_{date_3}_3T',
                        reorient_nifti=True)

                    common_id += 1

                except Exception:
                    invalid_dicoms.append(sorted_15[j])
                    invalid_dicoms.append(sorted_3[j])

    if len(invalid_dicoms) > 0:
        print(f'\nInvalid dicoms:')
        [print(x) for x in invalid_dicoms]


def ensure_3d(nifti_folder):
    paths = glob(f'{nifti_folder}/*')
    cnt = 0

    for path in tqdm(paths):
        img = nib.load(path)
        if len(img.shape) > 3:
            img = img.slicer[..., 0]
            nib.save(img, path)
            cnt += 1

    print(f'{cnt} nifti files was altered')


def extract_brain(folder_in, folder_out):
    paths = [x.split('/')[-1] for x in glob(f'{folder_in}/*')]

    for path in tqdm(paths):
        BET(
            in_file=f'{folder_in}/{path}',
            out_file=f'{folder_out}/{path}',
            robust=True,
            frac=0.6
        ).run()


def bias_field_correction(folder_in, folder_out):
    paths = [x.split('/')[-1] for x in glob(f'{folder_in}/*')]
    configuration = '-t 1 -n 3 -H 0.1 -I 4 -l 20.0 --nopve -B -o'

    for path in tqdm(paths):
        bashCommand = f'fast {configuration} {folder_out}/{path} {folder_in}/{path}'

        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        process.communicate()

        os.remove(f'{folder_out}/{path[:-7]}_seg.nii.gz')
        os.rename(f'{folder_out}/{path[:-7]}_restore.nii.gz', f'{folder_out}/{path}')


def remove_bad_imagepairs(folder_in):
    paths = sorted(glob(f'{folder_in}/*.nii.gz'))
    err_list = []

    for i in range(0, len(paths), 2):
        if not paths[i].split('_')[5] == paths[i+1].split('_')[5]:
            err_list.extend([paths[i], paths[i+1]])

        else:
            lowresimg = nib.load(paths[i]).get_fdata().astype(np.float32)
            highresimg = nib.load(paths[i+1]).get_fdata().astype(np.float32)

            lowresimg = lowresimg.flatten()
            highresimg = highresimg.flatten()

            iou = ssim(highresimg, lowresimg)

            if iou < 0.9:
                err_list.extend([paths[i], paths[i+1]])

    for item in err_list:
        if os.path.exists(item):
            os.remove(item)


def spatial_normalization_pairwise(folder_in, folder_out):
    paths = sorted([x.split('/')[-1] for x in glob(f'{folder_in}/*')])[::-1]

    pbar = tqdm(range(0, len(paths), 2))
    for i in pbar:
        highres, lowres = paths[i], paths[i+1]

        if '15T' in highres:
            highres, lowres = lowres, highres

        spatial_normalization(f'{folder_in}/{lowres}',
                              f'{folder_in}/{highres}', f'{folder_out}/{lowres}')
        img = nib.load(f'{folder_in}/{highres}')
        nib.save(img, f'{folder_out}/{highres}')


def spatial_normalization(lowres, highres, output):
    properties = '-bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear'
    bashCommand = f'flirt -in {lowres} -ref {highres} -out {output} {properties}'
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    process.communicate()


def spatial_normalization_to_standard(folder_in, folder_out):
    paths = [x for x in glob(f'{folder_in}/*')]

    for path in tqdm(paths):
        filename = path.split('/')[-1]
        bashCommand = f'flirt -in {path} -ref /usr/local/fsl/data/standard/MNI152_T1_1mm_brain -out {folder_out}/{filename} -bins 640 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear'
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        process.communicate()


def finalize_dataset(folder_in, folder_15T, folder_3T, status, subjects):
    paths = sorted(glob(f'{folder_in}/*'))

    overview = pd.read_csv(subjects)
    overview = overview[['Individual', 'Scan date', 'Timepoint']]
    overview['Scan date'] = pd.to_datetime(
        overview['Scan date']).dt.strftime('%Y-%m-%d')

    status = pd.read_csv(status, low_memory=False)
    status = status[['PTID', 'VISCODE', 'DX']]
    status.loc[status['VISCODE'] == 'bl', 'VISCODE'] = 'm00'

    for path in tqdm(paths):
        img = nib.load(path)

        info = path.split('/')[-1].split('_')
        number = info[0]
        subject = '_'.join(info[1:4])
        date = info[-2]
        tesla = info[-1]

        overview_subject = overview.loc[(overview['Individual'] == subject)
                                        & (overview['Scan date'] == date)]
        month = overview_subject.iloc[0]['Timepoint']

        status_subject = status.loc[(status['PTID'] == subject) & (status['VISCODE'] == month)]
        condition = status_subject.iloc[0]['DX']

        if '3T' in tesla:
            nib.save(img, f'{folder_3T}/{number}_{subject}_{month}_{date}_{condition}_{tesla}')
        else:
            nib.save(img, f'{folder_15T}/{number}_{subject}_{month}_{date}_{condition}_{tesla}')
