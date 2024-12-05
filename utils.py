# Core
import tensorflow as tf
import keras
import sys
import math
from tensorflow.python.client import device_lib
import cv2 as cv
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
import os
import glob
from tqdm import tqdm

# Image manipulation
from generator import DataGen, getIds, preprocess_image
from tensorflow.keras.utils import img_to_array
import keras.backend as kb

# Statistics
from sewar import full_ref as fr
import xlsxwriter
import numpy as np

import time

seed = 2019
np.random.seed = seed

# Image size
MODEL_SIZE = 512
INPUT_SHAPE = (MODEL_SIZE, MODEL_SIZE, 1)
BATCH_SIZE = 5

## LOSS FUNCTIONS

# MAE
def mae(y_true, y_pred):
    mae = tf.keras.losses.MeanAbsoluteError()
    return mae(y_true, y_pred)
# MSE
def mse(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred)

# PSNR
def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

# SSIM
def ssim(y_true, y_pred):
    _ssim = tf.math.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1))
    return _ssim

def ssim_loss(y_true, y_pred):
    _ssim_loss = tf.image.ssim(y_true, y_pred, max_val=1)
    _ssim_loss = tf.math.subtract(tf.constant(1.0), _ssim_loss)
    _ssim_loss = tf.reduce_mean(_ssim_loss)

    return _ssim_loss

## USED LOSS FUNCTION
def l1_ssim(y_true, y_pred):
    ALPHA = 0.84
    DELTA = 1 - ALPHA
    
    _ssim_loss = ssim_loss(y_true, y_pred)
    _ssim_loss = tf.math.multiply(_ssim_loss, ALPHA)
    
    _mae = mae(y_true, y_pred)
    _mae = tf.math.multiply(_mae, DELTA)
    
    _loss = tf.math.add(_mae, _ssim_loss)
    
    return _loss

# MS-SSIM
def ms_ssim(y_true, y_pred):
    _mssim = tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=1))
    return _mssim

    
## Check capabilities
def check_capabilities():
    ## GPU Check

    print(device_lib.list_local_devices())

    print("Num of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    tf.test.is_built_with_cuda()

    sys.version
    print(tf.__version__)
    print(keras.__version__)


## Saving the model and weights
def save_model(model, model_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join('saved_models', model_name), "w") as json_file:
        json_file.write(model_json)

    print("Saved model to disk")

def save_weights(model, file_name):
    model.save_weights(os.path.join('saved_models', file_name))

    print("Saved weights to disk")

def load_model(file_name):
    ## Loading the model
    # Load json and create model

    json_file = open(os.path.join('saved_models', file_name), 'r')
    model_json = json_file.read()
    json_file.close()

    model = tf.keras.models.model_from_json(model_json)

    print("Loaded model from disk")

    return model

def load_weights(model, file_name):
    # Load weights into new model
    model.load_weights(file_name)

    print("Loaded weights from disk")

## INTERNAL TEST SET
def get_test_data(t_path, RGB=False):

    test_path = os.path.join(t_path, 'JSRT')

    test_ids = getIds(test_path)
    print(f'Number of test images: {len(test_ids)}')

    test_gen = DataGen(test_ids, t_path, image_size=MODEL_SIZE, batch_size=BATCH_SIZE, RGB=RGB)
    return test_gen, test_ids

def get_flops(model, model_inputs) -> float:
        """
        Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
        in inference mode. It uses tf.compat.v1.profiler under the hood.
        Reference: https://github.com/wandb/wandb/blob/latest/wandb/integration/keras/keras.py#L1025-L1073
        """

        if not isinstance(
            model, (tf.keras.models.Sequential, tf.keras.models.Model)
        ):
            raise ValueError(
                "Calculating FLOPS is only supported for "
                "`tf.keras.Model` and `tf.keras.Sequential` instances."
            )

        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph,
        )

        # Compute FLOPs for one sample
        batch_size = 1
        inputs = [
            tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
            for inp in model_inputs
        ]

        # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
        real_model = tf.function(model).get_concrete_function(inputs)
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

        # Calculate FLOPs with tf.profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = (
            tf.compat.v1.profiler.ProfileOptionBuilder(
                tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
            )
            .with_empty_output()
            .build()
        )

        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
        )

        tf.compat.v1.reset_default_graph()

        # convert to GFLOPs
        return (flops.total_float_ops / 1e9)/2

## EXTERNAL TEST SET
def get_unseen_data(t_path, random=False, RGB=False):
    ids = getIds(t_path)
    print(f'Number of test images: {len(ids)}')
    test_ids = []
    
    # Store original image sizes
    original_sizes = []
    
    if (random):
        for i in range(0, 10):
            num = np.random.randint(0, len(ids))
            test_ids.append(ids[num])
    else:
        test_ids = ids.copy()
    
    test_data = []
    for id in tqdm(test_ids, desc='Loading test data'):
        data = os.path.join(t_path, id)

        if RGB:
            img = cv.imread(data)
            original_sizes.append(img.shape[:2])  # Store original height, width
            if (img.shape[:2] != (MODEL_SIZE, MODEL_SIZE)):
                img = cv.resize(img, (MODEL_SIZE, MODEL_SIZE), interpolation=cv.INTER_LANCZOS4)
        else:
            img = cv.imread(data, 0)
            original_sizes.append(img.shape[:2])  # Store original height, width
            if (img.shape != (MODEL_SIZE, MODEL_SIZE)):
                img = cv.resize(img, (MODEL_SIZE, MODEL_SIZE), interpolation=cv.INTER_LANCZOS4)
            img = np.expand_dims(img, axis = -1)
        
        img = preprocess_image(img)
        img = img_to_array(img)
        # Convert to 0--1 interval  
        img /= 255.0
        
        test_data.append(img)
    
    test_data = np.array(test_data)
    
    return test_data, test_ids, original_sizes

def test_model(model, t_path, RGB=False, random=False):
    results = []
    times = []
    data, ids, original_sizes = get_unseen_data(t_path, random=random, RGB=RGB)
    
    total_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(total_batches), desc='Performing inference'):
        batch_data = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        start = time.time()
        result = model(batch_data, training=False)
        end = time.time()
        results.extend(result)
        times.append(end-start)
    
    print(f'Mean inference time for a batch of size {BATCH_SIZE}: {np.mean(times) * 1000:.2f} ms')
    print(f'Median inference time for a batch of size {BATCH_SIZE}: {np.median(times) * 1000:.2f} ms')
    print(f'Min inference time for a batch of size {BATCH_SIZE}: {np.min(times) * 1000:.2f} ms')
    print(f'Max inference time for a batch of size {BATCH_SIZE}: {np.max(times) * 1000:.2f} ms')
    print(f'Std inference time for a batch of size {BATCH_SIZE}: {np.std(times) * 1000:.2f} ms')

    results = np.array(results)
    
    # Resize results back to original sizes and optionally extract bone images
    resized_suppression = []
    bone_only = [] 
    bone_boosted = []
    
    for idx, result in enumerate(results):
        original_h, original_w = original_sizes[idx]
        # 현재 처리중인 원본 이미지 가져오기
        original_img = data[idx]
        
        # Reshape to remove batch dimension if necessary
        result = result.reshape(MODEL_SIZE, MODEL_SIZE, 1)
        # Resize back to original dimensions
        resized = cv.resize(result, (original_w, original_h), interpolation=cv.INTER_LANCZOS4)
        # Add channel dimension back if needed
        resized = np.expand_dims(resized, axis=-1)
        
        #if extract_bones:
        # 원본 이미지도 동일한 크기로 리사이즈
        original_resized = cv.resize(original_img, (original_w, original_h), interpolation=cv.INTER_LANCZOS4)
        if len(original_resized.shape) == 2:
            original_resized = np.expand_dims(original_resized, axis=-1)
        
        # bone-only image
        bone_img = extract_bone_image(original_resized, resized)
        bone_only.append(bone_img)
        
        # boost bone contrast image
        boost_img = boost_bone_contrast(original_resized, resized)
        bone_boosted.append(boost_img)
          
        resized_suppression.append(resized)
    
    return resized_suppression, bone_only, bone_boosted, ids

# Extract bone only image
def extract_bone_image(original_image, suppressed_image, enhance=True, threshold=None):
    """
    원본 X-ray 이미지와 bone suppression된 이미지로부터 뼈 구조만을 추출합니다.
    
    Parameters:
    -----------
    original_image : numpy.ndarray
        원본 X-ray 이미지 (0-255 범위의 uint8 또는 0-1 범위의 float)
    suppressed_image : numpy.ndarray
        Bone이 제거된 이미지 (0-255 범위의 uint8 또는 0-1 범위의 float)
    enhance : bool, optional
        대비 향상 적용 여부 (default: True)
    threshold : float, optional
        뼈 구조 추출을 위한 임계값 (0-1 사이, default: None)
    
    Returns:
    --------
    numpy.ndarray
        추출된 뼈 구조 이미지
    """
    # 입력 이미지들을 float32 형식으로 변환하고 0-1 범위로 정규화
    if original_image.dtype == np.uint8:
        original_image = original_image.astype(np.float32) / 255.0
    if suppressed_image.dtype == np.uint8:
        suppressed_image = suppressed_image.astype(np.float32) / 255.0
    
    # 뼈 구조 추출을 위한 차영상 계산
    bone_image = original_image - suppressed_image
    
    # 음수값 제거 (뼈는 항상 밝은 부분이어야 함)
    bone_image = np.clip(bone_image, 0, 1)
    
    if enhance:
        # CLAHE(Contrast Limited Adaptive Histogram Equalization) 적용
        if len(bone_image.shape) == 3:
            bone_image = bone_image[:,:,0]  # 3채널인 경우 첫 번째 채널만 사용
            
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        bone_image = clahe.apply((bone_image * 255).astype(np.uint8)) / 255.0
    
    if threshold is not None:
        # 임계값 기반 이진화로 뼈 구조를 더 명확하게 구분
        bone_image = np.where(bone_image > threshold, bone_image, 0)
    
    # 노이즈 제거를 위한 가우시안 블러 적용
    bone_image = cv.GaussianBlur(bone_image, (3,3), 0)
    
    # 채널 차원 유지를 위해 shape 확인 후 차원 추가
    if len(original_image.shape) == 3 and len(bone_image.shape) == 2:
        bone_image = np.expand_dims(bone_image, axis=-1)
    
    return bone_image

# Extract boosted bone image
def boost_bone_contrast(original_image, suppressed_image, alpha=1.5, beta=0.5, gamma=0.8):
    """
    X-ray 이미지에서 뼈 구조를 강조하는 함수입니다.
    
    이 함수는 다음과 같은 단계로 뼈를 강조합니다:
    1. 원본과 suppressed 이미지의 차이로 뼈 마스크를 생성
    2. 적응형 히스토그램 평활화로 대비를 향상
    3. 언샤프 마스킹으로 엣지를 강화
    4. 원본 이미지와 강조된 뼈 구조를 블렌딩
    
    Parameters:
    -----------
    original_image : numpy.ndarray
        원본 X-ray 이미지 (0-1 범위의 float 또는 0-255 범위의 uint8)
    suppressed_image : numpy.ndarray
        Bone이 제거된 이미지 (0-1 범위의 float 또는 0-255 범위의 uint8)
    alpha : float
        뼈 구조 강조 강도 (더 큰 값 = 더 강한 강조, 기본값 1.5)
    beta : float
        엣지 강화 강도 (더 큰 값 = 더 선명한 엣지, 기본값 0.5)
    gamma : float
        최종 이미지 감마 보정 값 (1보다 작으면 어두운 부분 강조, 기본값 0.8)
    
    Returns:
    --------
    numpy.ndarray
        뼈 구조가 강조된 이미지
    """
    # 입력 이미지들을 float32 형식으로 변환하고 0-1 범위로 정규화
    if original_image.dtype == np.uint8:
        original_image = original_image.astype(np.float32) / 255.0
    if suppressed_image.dtype == np.uint8:
        suppressed_image = suppressed_image.astype(np.float32) / 255.0
        
    # 채널 차원 처리
    if len(original_image.shape) == 3:
        original_image = original_image[:,:,0]
    if len(suppressed_image.shape) == 3:
        suppressed_image = suppressed_image[:,:,0]
    
    # 1. 뼈 구조 마스크 생성
    bone_mask = original_image - suppressed_image
    bone_mask = np.clip(bone_mask, 0, 1)
    
    # 2. 적응형 히스토그램 평활화 (CLAHE) 적용
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply((original_image * 255).astype(np.uint8)) / 255.0
    
    # 3. 언샤프 마스킹으로 엣지 강화
    blur = cv.GaussianBlur(enhanced, (0,0), 3)
    unsharp_mask = enhanced - blur
    sharpened = enhanced + beta * unsharp_mask
    
    # 4. 뼈 구조 강조
    # bone_mask를 이용해 뼈 영역에서만 강조 효과 적용
    boosted = sharpened + alpha * bone_mask
    
    # 5. 감마 보정으로 대비 조정
    boosted = np.power(boosted, gamma)
    
    # 6. 값 범위 정규화
    boosted = np.clip(boosted, 0, 1)
    
    # 7. 채널 차원 추가 (if needed)
    if len(original_image.shape) == 3 or len(suppressed_image.shape) == 3:
        boosted = np.expand_dims(boosted, axis=-1)
    
    return boosted

def eval_results_bone_image(results, bone_images, boosted_images, ids, model_name):
    os.makedirs(os.path.join("outputs", "external", model_name), exist_ok=True)
    
    for i in tqdm(range(0, len(results)), desc='Saving predictions'):
        cv.imwrite(os.path.join("outputs", "external", model_name, f"{ids[i].split('.')[0]}_predicted.png"), results[i]*255)
        
    for i in tqdm(range(0, len(bone_images)), desc='Saving bone extracted images'):
        cv.imwrite(os.path.join("outputs", "external", model_name, f"{ids[i].split('.')[0]}_bone_extracted.png"), bone_images[i]*255)
        
    for i in tqdm(range(0, len(boosted_images)), desc='Saving boost bone contrast images'):
        cv.imwrite(os.path.join("outputs", "external", model_name, f"{ids[i].split('.')[0]}_bone_boosted.png"), boosted_images[i]*255)

def eval_results(results, ids, model_name):
    os.makedirs(os.path.join("outputs", "external", model_name), exist_ok=True)
    
    for i in tqdm(range(0, len(results)), desc='Saving predictions'):
        cv.imwrite(os.path.join("outputs", "external", model_name, f"{ids[i].split('.')[0]}_predicted.png"), results[i]*255)

def eval_test_results(model, model_name, t_path, RGB=False):
    test_gen, test_ids = get_test_data(t_path, RGB=RGB)
    print(f'Test batches: {len(test_gen)}')
    result = model.predict(test_gen)

    metrics = ["IMAGE", "SSIM", "MS-SSIM", "MSE", "MAE", "PSNR", "UQI", "CORRELATION", "INTERSECTION", "CHI_SQUARED", "BHATTACHARYYA"]

    predicted_ssim = []
    predicted_mssim = []
    predicted_mse = []
    predicted_mae = []
    predicted_psnr = []
    predicted_uqi = []
    predicted_corr = []
    predicted_inter = []
    predicted_chisq = []
    predicted_bhatta = []
    
    out_path = os.path.join("outputs", "internal", model_name)
    os.makedirs(out_path, exist_ok=True)
    
    workbook = xlsxwriter.Workbook(os.path.join(out_path, f"{model_name}_predictions_eval.xlsx"))
    f = workbook.add_worksheet()

    for col_num, data in enumerate(metrics):
        f.write(0, col_num, data)

    total_batches = (len(test_gen) + BATCH_SIZE - 1) // BATCH_SIZE  # Adjust for partial batches

    for i in tqdm(range(total_batches), desc='Evaluating results'):
        source, target = test_gen.__getitem__(i)
        target = np.array(target).astype('float32')
        temp_result = result[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
        temp_result = np.array(temp_result.astype('float32'))
        
        for j in range(temp_result.shape[0]):  # Handle partial batches
            temp_ssim = ssim(target[j], temp_result[j]).numpy() 
            temp_mssim = ms_ssim(target[j], temp_result[j]).numpy()  
            temp_mse = mse(target[j], temp_result[j]).numpy()
            temp_mae = mae(target[j], temp_result[j]).numpy()
            temp_psnr = psnr(target[j], temp_result[j]).numpy()  
            temp_uqi = fr.uqi(target[j], temp_result[j])  
            
            ## Convert to grayscale
            if target[j].shape[2] == 3:
                target[j] = cv.cvtColor(target[j], cv.COLOR_BGR2GRAY)
            if temp_result[j].shape[2] == 3:
                temp_result[j] = cv.cvtColor(temp_result[j], cv.COLOR_BGR2GRAY)
            
            img_g = target[j] * 255
            img_p = temp_result[j] * 255
            
            ## Calculate histograms
            hist_g = cv.calcHist([img_g], [0], None, [256], [0, 256])
            hist_p = cv.calcHist([img_p], [0], None, [256], [0, 256])
            hist_gn = cv.normalize(hist_g, hist_g).flatten()
            hist_pn = cv.normalize(hist_p, hist_p).flatten()
            
            ## Comparison
            temp_corr = cv.compareHist(hist_gn, hist_pn, cv.HISTCMP_CORREL)
            temp_inter = cv.compareHist(hist_gn, hist_pn, cv.HISTCMP_INTERSECT)
            temp_chisq = cv.compareHist(hist_gn, hist_pn, cv.HISTCMP_CHISQR)
            temp_bhatta = cv.compareHist(hist_gn, hist_pn, cv.HISTCMP_BHATTACHARYYA)
            
            predicted_ssim.append(temp_ssim)
            predicted_mssim.append(temp_mssim)
            predicted_mse.append(temp_mse)
            predicted_mae.append(temp_mae)
            predicted_psnr.append(temp_psnr)
            predicted_uqi.append(temp_uqi)
            predicted_corr.append(temp_corr)
            predicted_inter.append(temp_inter)
            predicted_chisq.append(temp_chisq)
            predicted_bhatta.append(temp_bhatta)

            name = os.path.splitext(test_ids[i*BATCH_SIZE+j])[0]
            vals = [name, temp_ssim, temp_mssim, temp_mse, temp_mae, temp_psnr, temp_uqi, temp_corr, temp_inter, temp_chisq, temp_bhatta]
            
            for col_num, data in enumerate(vals):
                f.write(i*BATCH_SIZE+j+1, col_num, data)

            ## Save results
            cv.imwrite(os.path.join(out_path, f"{name}_pred.png"), temp_result[j] * 255)
    workbook.close()

## FOR DEBONET ENSEMBLE, MATLAB SCRIPT FOR GENERATING THE COMBINED OUTPUT IS AVAILABLE AT: https://github.com/sivaramakrishnan-rajaraman/Bone-Suppresion-Ensemble/blob/main/bone_suppression_ensemble.py    
def eval_test_results_woPred(pred_path, target_path, model_name):
    pred_ids = sorted(glob.glob(pred_path + "*.png"))
    target_ids = sorted(glob.glob(target_path + "*.png"))

    metrics = ["IMAGE", "SSIM", "MS-SSIM", "MSE", "MAE", "PSNR", "UQI", "CORRELATION", "INTERSECTION", "CHI_SQUARED", "BHATTACHARYYA"]

    predicted_ssim = []
    predicted_mssim = []
    predicted_mse = []
    predicted_mae = []
    predicted_psnr = []
    predicted_uqi = []
    predicted_corr = []
    predicted_inter = []
    predicted_chisq = []
    predicted_bhatta = []
    
    out_path = os.path.join("outputs", "internal", model_name)
    os.makedirs(out_path, exist_ok=True)
    
    workbook = xlsxwriter.Workbook(
        os.path.join(out_path, f"{model_name}_predictions_eval.xlsx"))
    f = workbook.add_worksheet()

    for col_num, data in enumerate(metrics):
        f.write(0, col_num, data)

    for i in tqdm(range(0, len(target_ids)), desc='Evaluating results'):

        pred = cv.imread(pred_ids[i], 0)
        pred = np.expand_dims(pred, axis = -1)
        img_p = pred
        pred = np.array(pred).astype('float32')
        pred /= 255.0
        
        target = cv.imread(target_ids[i], 0)
        target = np.expand_dims(target, axis = -1)
        img_g = target
        target = np.array(target).astype('float32')
        target /= 255.0
        
        temp_ssim = ssim(target, pred).numpy() 
        temp_mssim = ms_ssim(target, pred).numpy()  
        temp_mse = mse(target, pred).numpy()
        temp_mae = mae(target, pred).numpy()
        temp_psnr = psnr(target, pred).numpy()  
        temp_uqi = fr.uqi(target, pred)  
        
        hist_g = cv.calcHist([img_g],[0],None,[256],[0,256])
        hist_p = cv.calcHist([img_p],[0],None,[256],[0,256])
        hist_gn = cv.normalize(hist_g, hist_g).flatten()
        hist_pn = cv.normalize(hist_p, hist_p).flatten()
        
        temp_corr = cv.compareHist(hist_gn, hist_pn, cv.HISTCMP_CORREL)
        temp_inter = cv.compareHist(hist_gn, hist_pn, cv.HISTCMP_INTERSECT)
        temp_chisq = cv.compareHist(hist_gn, hist_pn, cv.HISTCMP_CHISQR)
        temp_bhatta = cv.compareHist(hist_gn, hist_pn, cv.HISTCMP_BHATTACHARYYA)
        
        predicted_ssim.append(temp_ssim)
        predicted_mssim.append(temp_mssim)
        predicted_mse.append(temp_mse)
        predicted_mae.append(temp_mae)
        predicted_psnr.append(temp_psnr)
        predicted_uqi.append(temp_uqi)
        predicted_corr.append(temp_corr)
        predicted_inter.append(temp_inter)
        predicted_chisq.append(temp_chisq)
        predicted_bhatta.append(temp_bhatta)

        vals = [os.path.splitext(target_ids[i])[0], temp_ssim, temp_mssim, temp_mse, temp_mae, temp_psnr, temp_uqi, temp_corr, temp_inter, temp_chisq, temp_bhatta]
            
        for col_num, data in enumerate(vals):
            f.write(i+1, col_num, data)

    workbook.close()

initial_lr = 0.001
epochs = 100
decay = initial_lr / epochs

def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)

def compile_model(model):
    ## COMPILE

    model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr),
    loss = [l1_ssim],
    metrics = [ms_ssim, ssim, mse, mae, psnr]
    )
    model.summary()
    
    return model

def train_model(model, path, model_name):
    ## TRAINING
    
    # System paths
    source_path = os.path.join(path, "train")
    valid_path = os.path.join(path, "val")
    
    ## Validation / Training data
    #val_data_size = 720

    train_ids = getIds(os.path.join(source_path, "JSRT"))
    valid_ids = getIds(os.path.join(valid_path, "JSRT"))
    
    #valid_ids = train_ids[:val_data_size] 
    print(f"Validation: {len(valid_ids)}")

    #train_ids = train_ids[val_data_size:]
    print(f"Training: {len(train_ids)}")

    ## Training data generation
    train_gen = DataGen(train_ids, source_path, image_size=MODEL_SIZE, batch_size=BATCH_SIZE)

    ## Validation data generation
    valid_gen = DataGen(valid_ids, valid_path, image_size=MODEL_SIZE, batch_size=BATCH_SIZE)


    train_steps = len(train_ids) // BATCH_SIZE
    valid_steps = len(valid_ids) // BATCH_SIZE
    print(f'Training steps: {train_steps}')
    print(f'Validation steps: {valid_steps}')
    
    os.makedirs(os.path.join(".tf_checkpoints", model_name), exist_ok=True)
    filepath = os.path.join(".tf_checkpoints", model_name, f"{model_name}_b{BATCH_SIZE}_best_weights_{{epoch:02d}}.hdf5")
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_weights_only=True, monitor='val_loss', save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_time_based_decay, verbose=1)
    earlyStopping = EarlyStopping(monitor='val_loss', 
                                patience=10, 
                                verbose=1, 
                                mode='min')
    
    callbacks_list = [checkpoint, lr_scheduler, earlyStopping]
    
    history = model.fit(train_gen,
                        epochs = epochs,
                        validation_data=valid_gen,
                        steps_per_epoch=train_steps,
                        validation_steps=valid_steps,
                        callbacks=callbacks_list,
                        verbose=1)
    return history

def train_debonet(model, path, model_name):
    ## TRAINING OF THE DeBoNet ENSEMBLE FROM
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265691
    
    # System paths
    source_path = os.path.join(path, "train")
    valid_path = os.path.join(path, "val")

    train_ids = getIds(os.path.join(source_path, "JSRT"))
    valid_ids = getIds(os.path.join(valid_path, "JSRT"))
    
    print(f"Validation: {len(valid_ids)}")

    print(f"Training: {len(train_ids)}")

    ## Training data generation
    # The backbones require RGB input
    train_gen = DataGen(train_ids, source_path, image_size=MODEL_SIZE, batch_size=BATCH_SIZE, RGB=True)

    ## Validation data generation
    valid_gen = DataGen(valid_ids, valid_path, image_size=MODEL_SIZE, batch_size=BATCH_SIZE, RGB=True)

    ## STEPS
    train_steps = len(train_ids) // BATCH_SIZE
    valid_steps = len(valid_ids) // BATCH_SIZE
    print(f'Training steps: {train_steps}')
    print(f'Validation steps: {valid_steps}')
    
    ## NAMES: UNET_RES18, FPN_RES18, FPN_EF0
    os.makedirs(os.path.join(".tf_checkpoints", "DEBONET", model_name), exist_ok=True)
    filepath = os.path.join(".tf_checkpoints", "DEBONET", model_name, f"{model_name}_b{BATCH_SIZE}_best_weights_{{epoch:02d}}.hdf5")
    
    ## SETUP
    checkpoint = ModelCheckpoint(filepath, 
                                monitor='val_loss', 
                                verbose=1, 
                                save_weights_only=True,
                                save_best_only=True, 
                                mode='min') 
    earlyStopping = EarlyStopping(monitor='val_loss', 
                                patience=10, 
                                verbose=1, 
                                mode='min')
    tensor_board = TensorBoard(log_dir='.logs/', histogram_freq=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                factor=0.5, 
                                patience=10,
                                verbose=1, 
                                mode='min', 
                                min_lr=0.00001)
    
    callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]
    
    ## TRAINING
    history = model.fit(train_gen,
                        epochs = epochs,
                        validation_data=valid_gen,
                        steps_per_epoch=train_steps,
                        validation_steps=valid_steps,
                        callbacks=callbacks_list,
                        verbose=1)
    return history

## LR SCHEDULER FOR KALISZ MARCZYK MODEL
def scheduler(epoch, lr):
    if epoch <= 100:
        return lr
    else:
        initial_lrate = 0.001
        drop = 0.5
        epochs_drop = 50.0
        lrate = initial_lrate * math.pow(drop, math.floor((epoch-100)/epochs_drop))
        return lrate

def train_kalisz(model, path, epochs=300, model_name="KALISZ_AE"):
    ## TRAINING OF THE KALISZ MARCZYK AUTOENCODER FROM
    # https://ieeexplore.ieee.org/abstract/document/9635451
    
    # System paths
    source_path = os.path.join(path, "train")
    valid_path = os.path.join(path, "val")
    
    ## Validation / Training data
    train_ids = getIds(os.path.join(source_path, "JSRT"))
    valid_ids = getIds(os.path.join(valid_path, "JSRT"))
    
    print(f"Validation: {len(valid_ids)}")

    print(f"Training: {len(train_ids)}")

    ## Training data generation
    train_gen = DataGen(train_ids, source_path, image_size=MODEL_SIZE, batch_size=BATCH_SIZE)

    ## Validation data generation
    valid_gen = DataGen(valid_ids, valid_path, image_size=MODEL_SIZE, batch_size=BATCH_SIZE)

    ## Steps
    train_steps = len(train_ids) // BATCH_SIZE
    valid_steps = len(valid_ids) // BATCH_SIZE
    print(f'Training steps: {train_steps}')
    print(f'Validation steps: {valid_steps}')
    
    ## SETUP
    os.makedirs(os.path.join(".tf_checkpoints", model_name), exist_ok=True)
    filepath = os.path.join(".tf_checkpoints", model_name, f"{model_name}_b{BATCH_SIZE}_best_weights_{{epoch:02d}}.hdf5")
    checkpoint = ModelCheckpoint(filepath, 
                                monitor='val_loss', 
                                verbose=1, 
                                save_weights_only=True,
                                save_best_only=True, 
                                mode='min') 
    lr_scheduler = LearningRateScheduler(scheduler)
    
    callbacks_list = [checkpoint, lr_scheduler]
    
    ## TRAINING
    history = model.fit(train_gen,
                        epochs = epochs,
                        validation_data=valid_gen,
                        steps_per_epoch=train_steps,
                        validation_steps=valid_steps,
                        callbacks=callbacks_list,
                        verbose=1)
    
    ## RETURN RESULTS
    return history