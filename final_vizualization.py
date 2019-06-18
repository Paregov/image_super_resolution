import gc
from enum import Enum
from utils import check_path_exists
from data_loader import load_images_from_folder_with_names
from visualizations import compare_images
from predictor import Predictor
from models import generator_no_residual, generator_with_residual, discriminator


class DataSet(Enum):
    MSCOCO = 1
    CELEBA = 2


dataset = DataSet.CELEBA
predict_first = False

folders_list = []
if DataSet.CELEBA:
    folders_list.append('./test_results/final_compare/CELEBA/Image_01/')
    folders_list.append('./test_results/final_compare/CELEBA/Image_02/')
    folders_list.append('./test_results/final_compare/CELEBA/Image_03/')
    folders_list.append('./test_results/final_compare/CELEBA/Image_04/')
    folders_list.append('./test_results/final_compare/CELEBA/Image_05/')
    folders_list.append('./test_results/final_compare/CELEBA/Image_06/')
    folders_list.append('./test_results/final_compare/CELEBA/Image_07/')
    folders_list.append('./test_results/final_compare/CELEBA/Image_08/')
    folders_list.append('./test_results/final_compare/CELEBA/Image_09/')
    folders_list.append('./test_results/final_compare/CELEBA/Image_10/')
elif DataSet.MSCOCO:
    folders_list.append('./test_results/final_compare/MSCOCO/Image_01/')
    folders_list.append('./test_results/final_compare/MSCOCO/Image_02/')
    folders_list.append('./test_results/final_compare/MSCOCO/Image_03/')
    folders_list.append('./test_results/final_compare/MSCOCO/Image_04/')
    folders_list.append('./test_results/final_compare/MSCOCO/Image_05/')
    folders_list.append('./test_results/final_compare/MSCOCO/test/')

if predict_first:
    predictors_data = []
    if DataSet.CELEBA:
        predictors_data.append({'weights': './saved_models/weights.best.train.celeba.pt.hdf5', 'name': "PT"})
        predictors_data.append({'weights': './saved_models/weights.best.train.celeba.pt16.hdf5', 'name': "PT16"})
    if DataSet.MSCOCO:
        predictors_data.append({'weights': './saved_models/weights.best.train.mscoco.pt.hdf5', 'name': "PT"})
        predictors_data.append({'weights': './saved_models/weights.best.train.mscoco.pt16.hdf5', 'name': "PT16"})

if predict_first:
    for p_data in predictors_data:
        predictor = Predictor(p_data['weights'])
        for folder in folders_list:
            predictor.predict_from_file_and_save("{0}Input.jpg".format(folder), "{0}{1}.jpg".format(folder, p_data['name']))

for folder in folders_list:
    print(folder)
    data = load_images_from_folder_with_names("{0}*".format(folder))
    compare_images(data, multiply=False)

