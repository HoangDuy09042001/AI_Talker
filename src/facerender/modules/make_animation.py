from scipy.spatial import ConvexHull
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
import cv2
from PIL import Image
import numpy as np
import mediapipe as mp
import imageio
import time
import json
import torch
import pickle
from skimage import img_as_ubyte
import os
from src.facerender.modules.landmark import get_landmark

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).type_as(pred).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, 1) * 3 - 99
    return degree

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat

def keypoint_transformation(kp_canonical, he, wo_exp=False):
    kp = kp_canonical['value']    # (bs, k, 3) 
    yaw, pitch, roll= he['yaw'], he['pitch'], he['roll']      
    yaw = headpose_pred_to_degree(yaw) 
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    if 'yaw_in' in he:
        yaw = he['yaw_in']
    if 'pitch_in' in he:
        pitch = he['pitch_in']
    if 'roll_in' in he:
        roll = he['roll_in']

    rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)

    t, exp = he['t'], he['exp']
    if wo_exp:
        exp =  exp*0  
    
    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    # keypoint translation
    t[:, 0] = t[:, 0]*0
    t[:, 2] = t[:, 2]*0
    t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    return {'value': kp_transformed}


def convert_genoutput(predictions, frame_num):
  arrays = []
  images = []
  predictions_ts = torch.stack(predictions, dim=1)
  predictions_video = predictions_ts
  predictions_video = predictions_video.reshape((-1,)+predictions_video.shape[2:])
  predictions_video = predictions_video[:frame_num]
  # print("make_ani predicted frame: ",predictions_video)
  print("make_ani predicted frame shape: ",predictions_video.shape)
  for idx in range(predictions_video.shape[0]):
      image = predictions_video[idx]
      image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
      print("make_ani image array shape: ", image.shape)
      # Convert to uint8
      sub_array, sub_image = get_landmark(image, idx)
      arrays.append(sub_array)
      images.append(sub_image)
  return arrays, images

import json
import numpy as np

def append_data_to_json(existing_json_file, kp_driving, landmarks, image):
    kp_driving_np = kp_driving.numpy()
    
    try:
        # Try to open existing JSON file
        with open(existing_json_file, 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        # If file doesn't exist, create an empty list
        existing_data = []

    # Append new data to existing data
    existing_data.append({
        "kp_driving": kp_driving_np.tolist(),
        "landmarks": landmarks,
        "image": image
    })

    # Write the updated data back to the file
    with open(existing_json_file, 'w') as file:
        json.dump(existing_data, file)

    print("New data appended to existing JSON file.")



def make_animation(source_image, source_semantics, target_semantics,
                            generator, kp_detector, he_estimator, mapping, 
                            yaw_c_seq=None, pitch_c_seq=None, roll_c_seq=None,
                            use_exp=True, use_half=False):
    print("=========================MAKE ANIMATION=========================")
    with torch.no_grad():
        predictions = []
        source_image_start_time = time.time()
        kp_canonical = kp_detector(source_image)
        he_source = mapping(source_semantics)
        kp_source = keypoint_transformation(kp_canonical, he_source)
        print("make_ani kp_source time: ", time.time()-source_image_start_time)
        # for frame_idx in tqdm(range(2), 'Face Renderer:'):
        for frame_idx in tqdm(range(target_semantics.shape[1]), 'Face Renderer:'):
            subpredictions = []
            # still check the dimension
            print("make_ani target_semantics and source_semantics",target_semantics.shape, source_semantics.shape)
            target_semantics_frame = target_semantics[:, frame_idx]
            print(f"make_ani target_semantics_frame {frame_idx}: ",target_semantics_frame.shape)
            kp_drive_start = time.time()
            he_driving = mapping(target_semantics_frame)
            if yaw_c_seq is not None:
                he_driving['yaw_in'] = yaw_c_seq[:, frame_idx]
            if pitch_c_seq is not None:
                he_driving['pitch_in'] = pitch_c_seq[:, frame_idx] 
            if roll_c_seq is not None:
                he_driving['roll_in'] = roll_c_seq[:, frame_idx] 
            
            kp_driving = keypoint_transformation(kp_canonical, he_driving)
                
            kp_norm = kp_driving
            print("make_ani kp_drive time: ", time.time()-kp_drive_start)
            out = generator(source_image, kp_source=kp_source, kp_driving=kp_norm)
            '''
            source_image_new = out['prediction'].squeeze(1)
            kp_canonical_new =  kp_detector(source_image_new)
            he_source_new = he_estimator(source_image_new) 
            kp_source_new = keypoint_transformation(kp_canonical_new, he_source_new, wo_exp=True)
            kp_driving_new = keypoint_transformation(kp_canonical_new, he_driving, wo_exp=True)
            out = generator(source_image_new, kp_source=kp_source_new, kp_driving=kp_driving_new)
            '''
            predictions.append(out['prediction'])
            subpredictions.append(out['prediction'])
            landmarks, images = convert_genoutput(subpredictions, 2)
            print("make_ani sublandmarks: ", landmarks, len(landmarks))
            print("make_ani subimages: ", images, len(images))
            print("type of save: ", type(kp_driving["value"]), type(landmarks), type(images))

            # Create a dictionary to store all the data
            data_dict = {
                'kp_driving_value': kp_driving["value"],
                'landmarks': landmarks,
                'images': images
            }

            # Load existing data from data_dict.pt if it exists
            name_pt = "test.pt"
            if os.path.exists(name_pt):
                # Load the existing data_dict
                data_dict_array = torch.load(name_pt)
            else:
                # Create a new empty array if it doesn't exist
                data_dict_array = []

            # Append data_dict to the array
            data_dict_array.append(data_dict)

            # Save the updated array back to data_dict.pt
            torch.save(data_dict_array, name_pt)

            print("make_ani prediction ", len(predictions))
        predictions_ts = torch.stack(predictions, dim=1)
        # print('Results: ', results.multi_face_landmarks)
        ### the generated video is 256x256, so we keep the aspect ratio, 
    return predictions_ts

class AnimateModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, kp_extractor, mapping):
        super(AnimateModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.mapping = mapping

        self.kp_extractor.eval()
        self.generator.eval()
        self.mapping.eval()

    def forward(self, x):
        
        source_image = x['source_image']
        source_semantics = x['source_semantics']
        target_semantics = x['target_semantics']
        yaw_c_seq = x['yaw_c_seq']
        pitch_c_seq = x['pitch_c_seq']
        roll_c_seq = x['roll_c_seq']

        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                        self.generator, self.kp_extractor,
                                        self.mapping, use_exp = True,
                                        yaw_c_seq=yaw_c_seq, pitch_c_seq=pitch_c_seq, roll_c_seq=roll_c_seq)
        
        return predictions_video