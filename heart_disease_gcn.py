# -*- coding: utf-8 -*-
"""
Official Implementation for Master's Thesis:
"Prediction of Heart Failure Diseases using Deep Learning Models"
File: heart_disease_gcn.py
"""

import os
import shutil
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tqdm import tqdm

# ==========================================
# 1. MOUNT DRIVE & PATH CONFIGURATION
# ==========================================
from google.colab import drive
drive.mount('/content/drive')

root_image_dir = '/content/drive/MyDrive/archive (10)/ECG_DATA'
train_dir = os.path.join(root_image_dir, 'train')
test_dir = os.path.join(root_image_dir, 'test')
augmented_dir = '/content/drive/MyDrive/ECG_DATA/augmented_normals'
output_dir = '/content/drive/MyDrive/datacombine4'  # توحيد كافة المخرجات هنا

os.makedirs(output_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. DATASET STRUCTURE VERIFICATION
# ==========================================
def check_dataset_structure(root_dir):
    for split in ['train', 'test']:
        split_path = os.path.join(root_dir, split)
        if os.path.isdir(split_path):
            print(f"Contenu du répertoire {split}:")
            for class_dir in os.listdir(split_path):
                class_path = os.path.join(split_path, class_dir)
                if os.path.isdir(class_path):
                    print(f"  - {class_dir}: {len(os.listdir(class_path))} sous-dossiers")
        else:
            print(f"{split} n'existe pas dans {root_dir}")

check_dataset_structure(root_image_dir)

# ==========================================
# 3. DATAFRAME CREATION & MAPPING
# ==========================================
label_mapping = {
    'MI': 'Malade',
    'Abnormal': 'Malade',
    'History_MI': 'Malade',
    'Normal': 'Normal'
}

def create_dataframe(data_dir):
    data = []
    if not os.path.exists(data_dir):
        return pd.DataFrame(data)
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            label = label_mapping.get(class_dir)
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append({'image_path': file_path, 'label': label})
    return pd.DataFrame(data)

df_train = create_dataframe(train_dir)
df_test = create_dataframe(test_dir)

# ==========================================
# 4. DATA CLEANING & VALIDATION
# ==========================================
def nettoyer_dataframe(df, min_size=(100, 100)):
    chemins_valides, labels_valides = [], []
    for idx, row in df.iterrows():
        path = row['image_path']
        label = row['label']
        try:
            with Image.open(path) as img:
                if img.size[0] >= min_size[0] and img.size[1] >= min_size[1]:
                    chemins_valides.append(path)
                    labels_valides.append(label)
        except (UnidentifiedImageError, FileNotFoundError, OSError):
            continue
    return pd.DataFrame({'image_path': chemins_valides, 'label': labels_valides})

df_train_clean = nettoyer_dataframe(df_train)
df_test_clean = nettoyer_dataframe(df_test)

# ==========================================
# 5. CLASS BALANCING VIA DATA AUGMENTATION
# ==========================================
if os.path.exists(augmented_dir):
    shutil.rmtree(augmented_dir)
os.makedirs(augmented_dir, exist_ok=True)

normal_df = df_train_clean[df_train_clean['label'] == 'Normal']
malade_df = df_train_clean[df_train_clean['label'] == 'Malade']
n_to_generate = len(malade_df) - len(normal_df)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

generated = 0
if n_to_generate > 0 and len(normal_df) > 0:
    for idx, row in normal_df.iterrows():
        img = load_img(row['image_path'], color_mode='rgb')
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        for batch in datagen.flow(x, batch_size=1):
            unique_name = f"aug_{uuid.uuid4().hex}.jpg"
            save_path = os.path.join(augmented_dir, unique_name)
            img_aug = batch[0].astype('uint8')
            Image.fromarray(img_aug).save(save_path)
            generated += 1
            if generated >= n_to_generate:
                break
        if generated >= n_to_generate:
            break

augmented_images = [os.path.join(augmented_dir, f) for f in os.listdir(augmented_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
augmented_df = pd.DataFrame({'image_path': augmented_images, 'label': 'Normal'})
df_train_balanced = pd.concat([malade_df, normal_df, augmented_df], ignore_index=True)

# ==========================================
# 6. CLINICAL FEATURES GENERATION
# ==========================================
def add_clinical_features(df):
    ef, bnp, nyha, sbp, age = [], [], [], [], []
    for _, row in df.iterrows():
        is_diseased = row['label'] != 'Normal'
        
        # EF
        mu_ef = 0.60 if not is_diseased else 0.38
        sd_ef = 0.07 if not is_diseased else 0.09
        ef_val = np.clip(np.random.normal(mu_ef, sd_ef), 0.20, 0.75)
        
        # BNP
        mu_bnp = 4.2 if not is_diseased else 5.8
        bnp_val = np.clip(np.random.lognormal(mu_bnp, 0.6), 10, 2000)
        
        # NYHA
        probs = [0.50, 0.30, 0.15, 0.05] if not is_diseased else [0.10, 0.30, 0.40, 0.20]
        nyha_cls = np.random.choice([1, 2, 3, 4], p=probs)
        
        # SBP
        mu_sbp = 125 if not is_diseased else 110
        sbp_val = np.clip(np.random.normal(mu_sbp, 15), 80, 180)
        
        # AGE
        mu_age = 55 if not is_diseased else 68
        age_val = np.clip(np.random.normal(mu_age, 10), 20, 90)
        
        ef.append(round(ef_val, 3))
        bnp.append(round(bnp_val, 3))
        nyha.append(nyha_cls)
        sbp.append(round(sbp_val, 1))
        age.append(int(round(age_val)))
        
    df['ef'], df['bnp'], df['nyha'], df['sbp'], df['age'] = ef, bnp, nyha, sbp, age
    return df

df_train_final = add_clinical_features(df_train_balanced)
df_test_final = add_clinical_features(df_test_clean)

# ==========================================
# 7. DATA SPLITTING (TRAIN / VALIDATION)
# ==========================================
df_train_split, df_val_split = train_test_split(
    df_train_final,
    test_size=0.16,
    stratify=df_train_final['label'],
    random_state=42
)

# حفظ ملفات الـ CSV المحدثة والموحدة
df_train_split.to_csv(os.path.join(output_dir, 'train_top5.csv'), index=False)
df_val_split.to_csv(os.path.join(output_dir, 'val_split.csv'), index=False)
df_test_final.to_csv(os.path.join(output_dir, 'test_top5.csv'), index=False)

# ==========================================
# 8. LABEL ENCODING & FEATURE STANDARDIZATION
# ==========================================
label_encoder = LabelEncoder()
df_train_split['label'] = label_encoder.fit_transform(df_train_split['label'])
df_val_split['label'] = label_encoder.transform(df_val_split['label'])
df_test_final['label'] = label_encoder.transform(df_test_final['label'])

scaler = StandardScaler()
train_feats = scaler.fit_transform(df_train_split[['ef', 'bnp', 'nyha', 'sbp', 'age']])
val_feats = scaler.transform(df_val_split[['ef', 'bnp', 'nyha', 'sbp', 'age']])
test_feats = scaler.transform(df_test_final[['ef', 'bnp', 'nyha', 'sbp', 'age']])

joblib.dump(scaler, os.path.join(output_dir, 'scaler_top5.pkl'))
joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))

# ==========================================
# 9. MULTIMODAL FEATURE EXTRACTION & GRAPH BUILD
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
resnet.eval()

def extract_image_feature(path):
    try:
        full_path = path if os.path.isabs(path) else os.path.join(root_image_dir, path)
        image = Image.open(full_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = resnet(image).squeeze().cpu().numpy()
        return feat
    except Exception as e:
        return np.zeros(512)

def extract_and_save_image_features(df, split_name):
    features_path = os.path.join(output_dir, f'{split_name}_image_features.npy')
    if os.path.exists(features_path):
        return np.load(features_path)
    print(f" Extraction des features images pour {split_name}...")
    feats = np.array([extract_image_feature(p) for p in tqdm(df['image_path'])])
    np.save(features_path, feats)
    return feats

def build_graph(df, clinical_feats, split_name):
    image_feats = extract_and_save_image_features(df, split_name)
    all_feats = np.concatenate([image_feats, clinical_feats], axis=1)
    x = torch.tensor(all_feats, dtype=torch.float)
    y = torch.tensor(df['label'].values, dtype=torch.long)
    
    knn_graph = kneighbors_graph(all_feats, n_neighbors=5, mode='connectivity', include_self=False)
    edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
    edge_index_with_loops = torch_geometric.utils.add_self_loops(edge_index)[0]
    
    return Data(x=x, edge_index=edge_index_with_loops, y=y)

train_data = build_graph(df_train_split, train_feats, 'train').to(device)
val_data = build_graph(df_val_split, val_feats, 'val').to(device)
test_data = build_graph(df_test_final, test_feats, 'test').to(device)

# ==========================================
# 10. GCN MODEL DEFINITION & TRAINING
# ==========================================
class GlobalGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.classifier(x)

EPOCHS = 30
in_channels = 512 + 5
model = GlobalGCN(in_channels=in_channels, hidden_channels=128, out_channels=len(label_encoder.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    out = model(train_data)
    loss = criterion(out, train_data.y)
    loss.backward()
    optimizer.step()
    
    pred = out.argmax(dim=1)
    acc = (pred == train_data.y).sum().item() / train_data.num_nodes
    
    model.eval()
    with torch.no_grad():
        val_out = model(val_data)
        val_pred = val_out.argmax(dim=1)
        val_acc = (val_pred == val_data.y).sum().item() / val_data.num_nodes
        
    print(f"[Epoch {epoch}] Loss: {loss.item():.4f} | Train Acc: {acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

model_save_path = os.path.join(output_dir, 'model_gcn_top5.pth')
torch.save(model.state_dict(), model_save_path)
print(f" Modèle sauvegardé dans {model_save_path}")

# Evaluation final sur Test Set
model.eval()
with torch.no_grad():
    test_out = model(test_data)
    test_pred = test_out.argmax(dim=1)
    y_true = test_data.y.cpu().numpy()
    y_pred = test_pred.cpu().numpy()

print("\n📊 Rapport Test")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# ==========================================
# 11. REAL-TIME SINGLE-PATIENT INFERENCE
# ==========================================
print("\n Simulation d'inférence pour un seul patient...")

val_image_feats = np.load(os.path.join(output_dir, 'val_image_features.npy'))

patient_dict = {
    'image_path': 'test/MI/MI(100).jpg', # التأكد من صحة المسار النسبي داخل مجلد المشروع
    'ef': 0.334,
    'bnp': 463.331,
    'nyha': 3,
    'sbp': 132,
    'age': 70
}

patient_feats_raw = np.array([[patient_dict['ef'], patient_dict['bnp'], patient_dict['nyha'], patient_dict['sbp'], patient_dict['age']]])
patient_feats = scaler.transform(patient_feats_raw)
patient_img_feat = extract_image_feature(patient_dict['image_path'])

all_feats = np.concatenate([val_image_feats, val_feats], axis=1)
patient_all_feat = np.concatenate([patient_img_feat, patient_feats.squeeze()])
all_feats = np.vstack([all_feats, patient_all_feat])

labels = np.concatenate([df_val_split['label'].values, [-1]])
knn_graph = kneighbors_graph(all_feats, n_neighbors=5, mode='connectivity', include_self=False)
edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)

edge_index_with_loops = torch_geometric.utils.add_self_loops(edge_index)[0]

inference_data = Data(
    x=torch.tensor(all_feats, dtype=torch.float),
    edge_index=edge_index_with_loops,
    y=torch.tensor(labels, dtype=torch.long)
).to(device)

with torch.no_grad():
    inference_out = model(inference_data)
    inference_pred = inference_out.argmax(dim=1)
    patient_pred_class = label_encoder.inverse_transform([inference_pred[-1].cpu().item()])[0]

print(f" Résultat de prédiction : Le patient est diagnostiqué comme -> {patient_pred_class}")
