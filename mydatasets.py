import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms
from tqdm import tqdm

class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames_per_sample, stride):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, "input")
        self.label_dir = os.path.join(root_dir, "label")
        self.num_frames_per_sample = num_frames_per_sample
        self.stride = stride
        self.class_folders = sorted(os.listdir(os.path.join(root_dir, "input")))
        self.data = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._prepare_data()

    def _prepare_data(self):
        for class_folder in tqdm(self.class_folders):
            class_path = os.path.join(self.input_dir, class_folder)
            input_path_left_eye = class_path + "/part_frames/left_eye_left_eyebrow/"
            input_path_right_eye = class_path + "/part_frames/right_eye_right_eyebrow/"
            input_path_mouth = class_path + "/part_frames/mouth/"
            input_path_facial = class_path + "/facial_landmark/"
            input_path_STMap = class_path + "/part_frames/STMap/"
            video_folders_left_eye = sorted(os.listdir(input_path_left_eye))
            label_class_path = os.path.join(self.label_dir, class_folder)

            for subject_idx, video_folder in enumerate(video_folders_left_eye):
                video_path_left_eye = os.path.join(input_path_left_eye, video_folder)
                video_path_right_eye = os.path.join(input_path_right_eye, video_folder)
                video_path_mouth = os.path.join(input_path_mouth, video_folder)
                video_folder = video_folder.split('.')[0]
                npy_facial_path = os.path.join(input_path_facial, f"{video_folder}_facial_landmarks.npy")
                STMap_path = os.path.join(input_path_STMap, f"{video_folder}.png")
                label_path = os.path.join(label_class_path, f"{video_folder}.csv")

                frames_left_eye = self._load_npy(video_path_left_eye)
                frames_right_eye = self._load_npy(video_path_right_eye)
                frames_mouth = self._load_npy(video_path_mouth)
                npys_facial = self._load_npy(npy_facial_path)
                STMaps = self._load_STMap(STMap_path)
                labels = self._load_labels(label_path)
                labels_subject = self._load_labels_subject(label_path)

                split_data = self._split_frames_and_labels(frames_left_eye, frames_right_eye, frames_mouth, npys_facial,
                                                           STMaps, labels, labels_subject, subject_idx=subject_idx)
                self.data.extend(split_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], idx)

    def _load_npy(self, npy_path):
        npy_file = npy_path
        frames = np.load(npy_file, mmap_mode='r')
        frames = np.copy(frames)
        return frames

    def _load_STMap(self, STMap_path):
        frame = Image.open(STMap_path)
        frame = np.array(frame)
        return frame

    def _load_labels(self, label_path):
        df = pd.read_csv(label_path)
        labels = df[['RESP', 'r_peaks']].values
        return labels

    def _load_labels_subject(self, label_path):
        df = pd.read_csv(label_path)
        labels = df[['drowsiness', 'cognitive workload', 'hr', 'rr']].values
        return labels

    def _split_frames_and_labels(self, frames_left_eye, frames_right_eye, frames_mouth, npys_facial, STMaps, labels,
                                 labels_subject, subject_idx):
        num_frames = len(labels)
        split_frames_and_labels = []
        for i in range(0, num_frames - self.num_frames_per_sample + 1, self.stride):
            frame_left_eye = frames_left_eye[i:i + self.num_frames_per_sample]
            frame_right_eye = frames_right_eye[i:i + self.num_frames_per_sample]
            frame_mouth = frames_mouth[i:i + self.num_frames_per_sample]
            npy_facial = npys_facial[i:i + self.num_frames_per_sample]
            STMap = STMaps[:, i:i + self.num_frames_per_sample, :]
            label = labels[i:i + self.num_frames_per_sample]
            label_subject = labels_subject[i:i + self.num_frames_per_sample][0]
            split_frames_and_labels.append(
                (frame_left_eye, frame_right_eye, frame_mouth, npy_facial, STMap, label, label_subject, subject_idx))
        return split_frames_and_labels
