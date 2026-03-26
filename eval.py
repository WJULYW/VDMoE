import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score, precision_score

def evaluate_phys2(HR_pr, HR_rel):
    HR_pr = np.array(HR_pr).reshape(-1)
    HR_rel = np.array(HR_rel).reshape(-1)
    temp = HR_pr - HR_rel
    me = np.mean(temp)
    std = np.std(temp)
    mae = np.sum(np.abs(temp)) / len(temp)
    rmse = np.sqrt(np.sum(np.power(temp, 2)) / len(temp))
    mer = np.mean(np.abs(temp) / HR_rel)
    p = np.sum((HR_pr - np.mean(HR_pr)) * (HR_rel - np.mean(HR_rel))) / (
            0.01 + np.linalg.norm(HR_pr - np.mean(HR_pr), ord=2) * np.linalg.norm(HR_rel - np.mean(HR_rel), ord=2))

    return me, std, mae, rmse, mer, p

def evaluate_signals(pr, rel):
    me = 0
    std = 0
    mae = 0
    rmse = 0
    mer = 0
    p = 0
    for i in range(pr.shape[0]):
        signal_pr = pr[i]
        signal_rel = rel[i]
        signal_pr = np.array(signal_pr).reshape(-1)
        signal_rel = np.array(signal_rel).reshape(-1)
        temp = signal_pr - signal_rel
        me += np.mean(temp)
        std += np.std(temp)
        mae += np.mean(np.abs(temp))
        rmse += np.sqrt(np.mean(np.power(temp, 2)))
        mer += np.mean(np.abs(temp) / signal_rel)

        numerator = np.sum((signal_pr - np.mean(signal_pr)) * (signal_rel - np.mean(signal_rel)))
        denominator = np.sqrt(
            np.sum((signal_pr - np.mean(signal_pr)) ** 2) * np.sum((signal_rel - np.mean(signal_rel)) ** 2))
        denominator = np.maximum(denominator, 1e-8)
        p += numerator / denominator

    return me / pr.shape[0], std / pr.shape[0], mae / pr.shape[0], rmse / pr.shape[0], mer / pr.shape[0], p / pr.shape[
        0]

def evaluate_phys(HR_pr, HR_rel):
    HR_pr = np.array(HR_pr).reshape(-1)
    HR_rel = np.array(HR_rel).reshape(-1)
    temp = HR_pr - HR_rel
    me = np.mean(temp)
    std = np.std(temp)
    mae = np.mean(np.abs(temp))
    rmse = np.sqrt(np.mean(np.power(temp, 2)))
    mer = np.mean(np.abs(temp) / HR_rel)

    numerator = np.sum((HR_pr - np.mean(HR_pr)) * (HR_rel - np.mean(HR_rel)))
    denominator = np.sqrt(np.sum((HR_pr - np.mean(HR_pr)) ** 2) * np.sum((HR_rel - np.mean(HR_rel)) ** 2))
    denominator = np.maximum(denominator, 1e-8)
    p = numerator / denominator

    return me, std, mae, rmse, mer, p

def calculate_heart_rate(r_peaks, sampling_rate=15):
    threshold = 0.5
    r_peaks = (r_peaks >= threshold)
    heart_rates = []
    for batch in r_peaks:
        peaks_indices = np.where(batch == 1)[0]
        if len(peaks_indices) > 1:
            rr_intervals = np.diff(peaks_indices) / sampling_rate
            heart_rate = 60 / rr_intervals.mean()
        else:
            heart_rate = 0
        heart_rates.append(heart_rate)
    return np.array(heart_rates)

def detrend_ecg_signal(ecg_data, degree=6):

    x = np.arange(len(ecg_data))
    p = np.polyfit(x, ecg_data, degree)
    trend = np.polyval(p, x)
    detrended_ecg = ecg_data - trend
    return detrended_ecg, trend

def normalize_signal(signal):

    min_val = np.min(signal)
    max_val = np.max(signal)
    normalized_signal = 2 * (signal - min_val) / (max_val - min_val) - 1
    return normalized_signal

def calculate_repiration_rate(breathing_signals, sampling_rate=30):
    rr = []
    for breathing_signal in breathing_signals:
        breathing_signal, trend = detrend_ecg_signal(breathing_signal, degree=6)
        breathing_signal = normalize_signal(breathing_signal)
        peaks, _ = find_peaks(breathing_signal, distance=30 // 4, height=0.5)
        num_of_breaths = len(peaks)
        duration_in_seconds = len(breathing_signal) / sampling_rate
        duration_in_minutes = duration_in_seconds / 60

        respiration_rate = num_of_breaths / duration_in_minutes
        rr.append(respiration_rate)
    return np.array(rr)

def SEN(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    return con_mat[1, 1] / (con_mat[1, 1] + con_mat[1, 0])

def SPE(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    return con_mat[0, 0] / (con_mat[0, 0] + con_mat[0, 1])

def test_model(model, test_dataloader, device, criterion1, criterion2):
    model.eval()
    all_drowsiness_labels = []
    all_cognitive_labels = []
    all_resp_labels = []
    all_rr_labels = []
    all_hr_labels = []

    all_drowsiness_preds = []
    all_cognitive_preds = []
    all_rr_preds = []
    all_hr_preds = []

    with torch.no_grad():
        for i, ((frames_left_eye, frames_right_eye, frames_mouth, facials, STMaps, labels,
                 labels_subject, _), idx) in enumerate(test_dataloader):
            frames_left_eye = frames_left_eye.permute(0, 4, 1, 2, 3).float().to(device)
            frames_right_eye = frames_right_eye.permute(0, 4, 1, 2, 3).float().to(device)
            frames_mouth = frames_mouth.permute(0, 4, 1, 2, 3).float().to(device)
            STMaps = STMaps.permute(0, 3, 1, 2).float().to(device)
            facials = facials.float().to(device)
            drowsiness = labels_subject[:, 0].unsqueeze(1).float()
            cognitive = labels_subject[:, 1].unsqueeze(1).float()
            resp = labels[:, :, 0].float()
            hr = labels_subject[:, 2].unsqueeze(1).float()
            rr = labels_subject[:, 3].unsqueeze(1).float()
            resp = resp.numpy()

            outputs, _, _ = model(
                frames_left_eye, frames_right_eye, frames_mouth, STMaps, facials)
            output_drowsiness = outputs[0]
            output_cognitive = outputs[1]
            output_hr = outputs[2]
            output_rr = outputs[3]
            _, output_drowsiness = torch.max(output_drowsiness.data, 1)
            output_drowsiness = output_drowsiness.unsqueeze(1)
            _, output_cognitive = torch.max(output_cognitive.data, 1)
            output_cognitive = output_cognitive.unsqueeze(1)

            all_drowsiness_labels.append(drowsiness.numpy())
            all_cognitive_labels.append(cognitive.numpy())
            all_resp_labels.append(resp)
            all_rr_labels.append(rr)
            all_hr_labels.append(hr)

            all_drowsiness_preds.append(output_drowsiness.cpu().numpy())
            all_cognitive_preds.append(output_cognitive.cpu().numpy())
            all_rr_preds.append(output_rr.cpu().numpy())
            all_hr_preds.append(output_hr.cpu().numpy())

    all_drowsiness_labels = np.vstack(all_drowsiness_labels)
    all_cognitive_labels = np.vstack(all_cognitive_labels)
    all_hr_labels = np.concatenate(all_hr_labels, axis=0)
    all_rr_labels = np.concatenate(all_rr_labels, axis=0)

    all_drowsiness_preds = np.vstack(all_drowsiness_preds)
    all_cognitive_preds = np.vstack(all_cognitive_preds)
    all_rr_preds = np.concatenate(all_rr_preds, axis=0)
    all_hr_preds = np.concatenate(all_hr_preds, axis=0)

    drowsiness_accuracy = accuracy_score(all_drowsiness_labels, all_drowsiness_preds)
    cognitive_accuracy = accuracy_score(all_cognitive_labels, all_cognitive_preds)
    drowsiness_f1 = f1_score(all_drowsiness_labels, all_drowsiness_preds, average='binary')
    cognitive_f1 = f1_score(all_cognitive_labels, all_cognitive_preds, average='binary')
    drowsiness_auc = roc_auc_score(all_drowsiness_labels, all_drowsiness_preds)
    cognitive_auc = roc_auc_score(all_cognitive_labels, all_cognitive_preds)
    drowsiness_sen = SEN(all_drowsiness_labels, all_drowsiness_preds)
    cognitive_sen = SEN(all_cognitive_labels, all_cognitive_preds)
    drowsiness_spe = SPE(all_drowsiness_labels, all_drowsiness_preds)
    cognitive_spe = SPE(all_cognitive_labels, all_cognitive_preds)
    drowsiness_recall = recall_score(all_drowsiness_labels, all_drowsiness_preds)
    cognitive_recall = recall_score(all_cognitive_labels, all_cognitive_preds)
    drowsiness_precision = precision_score(all_drowsiness_labels, all_drowsiness_preds)
    cognitive_precision = precision_score(all_cognitive_labels, all_cognitive_preds)

    print(
        f"Drowsiness - Accuracy: {drowsiness_accuracy:.4f}, F1 Score: {drowsiness_f1:.4f}, AUC: {drowsiness_auc:.4f}, Sensitivity: {drowsiness_sen:.4f}, Specificity: {drowsiness_spe:.4f}, Recall: {drowsiness_recall:.4f}, Precision: {drowsiness_precision:.4f}")
    print(
        f"Cognitive - Accuracy: {cognitive_accuracy:.4f}, F1 Score: {cognitive_f1:.4f}, AUC: {cognitive_auc:.4f}, Sensitivity: {cognitive_sen:.4f}, Specificity: {cognitive_spe:.4f}, Recall: {cognitive_recall:.4f}, Precision: {cognitive_precision:.4f}")

    hr_me, hr_std, hr_mae, hr_rmse, hr_mer, hr_p = evaluate_phys(all_hr_preds, all_hr_labels)
    print(
        'HR - me: {:.4f}, std: {:.4f}, mae: {:.4f}, rmse: {:.4f}, mer: {:.4f}, p: {:.4f}'.format(hr_me, hr_std, hr_mae,
                                                                                                 hr_rmse, hr_mer, hr_p))

    rr_me, rr_std, rr_mae, rr_rmse, rr_mer, rr_p = evaluate_phys(all_rr_preds, all_rr_labels)
    print(
        'RR - me: {:.4f}, std: {:.4f}, mae: {:.4f}, rmse: {:.4f}, mer: {:.4f}, p: {:.4f}'.format(rr_me, rr_std, rr_mae,
                                                                                                 rr_rmse, rr_mer, rr_p))
