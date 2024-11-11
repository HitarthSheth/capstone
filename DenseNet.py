import copy
import traceback
from matplotlib import cm, pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_score, recall_score, f1_score, roc_curve
import os
import seaborn as sns


@dataclass
class MPMetrics:
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    accuracies: list = field(default_factory=list)
    mp_ratios: list = field(default_factory=list)
    predicted_labels: list = field(default_factory=list)
    true_labels: list = field(default_factory=list)
    best_val_loss: float = float('inf')
    best_accuracy: float = 0.0

    def update(self, train_loss, val_loss, predictions, labels, mp_ratios, accuracy):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.predicted_labels.extend(predictions)
        self.true_labels.extend(labels)
        self.mp_ratios.extend(mp_ratios)
        self.accuracies.append(accuracy)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy

def calculate_metrics(y_true, y_pred, y_pred_binary):
    """Calculate metrics with proper handling of edge cases."""
    try:
        metrics = {
            'sensitivity': recall_score(y_true, y_pred_binary, zero_division=0),
            'specificity': recall_score(y_true, y_pred_binary, pos_label=0, zero_division=0),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true, y_pred_binary, zero_division=0)
        }
        
        # Calculate ROC and AUC only if we have both classes
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            metrics['auc_roc'] = auc(fpr, tpr)
        else:
            metrics['auc_roc'] = 0.0
            
        return metrics, fpr, tpr
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            'sensitivity': 0.0,
            'specificity': 0.0,
            'precision': 0.0,
            'f1': 0.0,
            'auc_roc': 0.0
        }, None, None

 
def process_batch(model, images, device='cpu'):
    """Utility function to process a batch of images with proper error handling."""
    try:
        images = images.to(device)
        measurements, predictions, mp_ratios = model(images)
        return measurements, predictions, mp_ratios
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return None, None, None

def analyze_pd_data(metrics, test_samples):
    """Analyze PD vs Non-PD distributions and find optimal threshold."""
    mp_ratios = np.array(metrics.mp_ratios[-test_samples:])
    true_labels = np.array(metrics.true_labels[-test_samples:])
    
    # Separate PD and Non-PD ratios
    pd_ratios = mp_ratios[true_labels == 1]
    non_pd_ratios = mp_ratios[true_labels == 0]
    
    # Find optimal threshold using ROC curve
    thresholds = np.linspace(mp_ratios.min(), mp_ratios.max(), 1000)
    best_threshold = None
    best_accuracy = 0
    
    for threshold in thresholds:
        predictions = (mp_ratios >= threshold).astype(int)
        accuracy = (predictions == true_labels).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return pd_ratios, non_pd_ratios, best_threshold, best_accuracy

class MPDenseNet(nn.Module):
    def __init__(self):
        super(MPDenseNet, self).__init__()
        
        # Constants
        self.growth_rate = 32
        init_channels = 64
        
        # Initial feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, init_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks setup
        num_channels = init_channels
        
        self.dense1 = self._make_dense_block(num_channels, 6)
        num_channels = num_channels + 6 * self.growth_rate
        self.trans1 = self._make_transition(num_channels)
        num_channels = num_channels // 2
        
        self.dense2 = self._make_dense_block(num_channels, 8)
        num_channels = num_channels + 8 * self.growth_rate
        self.trans2 = self._make_transition(num_channels)
        num_channels = num_channels // 2
        
        self.dense3 = self._make_dense_block(num_channels, 8)
        num_channels = num_channels + 8 * self.growth_rate
        
        self.norm_final = nn.BatchNorm2d(num_channels)
        self.final_channels = num_channels
        
        # Simplified measurement head for only midbrain and pons areas
        self.measurement_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.final_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
            nn.Softplus()  
        )


    def _make_dense_layer(self, in_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, self.growth_rate, kernel_size=3, padding=1),
            nn.Dropout(0.2)
        )
    
    def _make_dense_block(self, in_channels, num_layers):
        layers = []
        current_channels = in_channels
        for _ in range(num_layers):
            layers.append(self._make_dense_layer(current_channels))
            current_channels += self.growth_rate
        return DenseBlock(nn.ModuleList(layers))
    
    def _make_transition(self, in_channels):
        out_channels = in_channels // 2
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Dropout(0.2),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
   
    def forward(self, x):
      x = self.features(x)
      x = self.dense1(x)
      x = self.trans1(x)
      x = self.dense2(x)
      x = self.trans2(x)
      x = self.dense3(x)
      x = self.norm_final(x)
      x = F.relu(x, inplace=True)
    
    # Get midbrain and pons measurements
      measurements = self.measurement_head(x)
    
    # Calculate M/P ratio
      midbrain, pons = measurements[:, 0], measurements[:, 1]
      mp_ratio = F.softplus(midbrain) / (F.softplus(pons) + 1e-6)
    
    # Use the optimal threshold determined from analysis
      OPTIMAL_THRESHOLD = 0.369533866643905  
    
    # Generate PD prediction using optimal threshold
      pd_prediction = torch.sigmoid((mp_ratio - OPTIMAL_THRESHOLD) * 20).unsqueeze(1)
    
      return measurements, pd_prediction, mp_ratio
        

class DenseBlock(nn.Module):
    def __init__(self, layers):
        super(DenseBlock, self).__init__()
        self.layers = layers
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)

class MPDataset(Dataset):
    def __init__(self, data, img_dir, transform=None):
        """
        Args:
            data (pd.DataFrame): DataFrame containing image data and measurements
            img_dir (str): Directory containing the images
            transform (callable, optional): Optional transform to be applied on images
        """
        self.data = data
        self.img_dir = img_dir
        self.transform = transform
        
        # Calculate class weights based on M/P ratio threshold
        self.mp_ratios = self.data['A/B'].values.astype(np.float32)
        labels = (self.mp_ratios >= 0.5).astype(np.float32)  # Changed threshold
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        self.class_weights = total_samples / (len(unique_labels) * counts)
        
        print("\nDataset class distribution (threshold=0.5):")
        for label, count in zip(unique_labels, counts):
            print(f"Class {label}: {count} samples")
            
        print(f"Total number of samples: {len(self.data)}")
            
        # Print total dataset size
        print(f"Total number of samples: {len(self.data)}")
    
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            img_name = f"{row['subject']}.jpg"
            img_path = os.path.join(self.img_dir, img_name)
            
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            midbrain = float(row['A Area cm^2'])
            pons = float(row['B Area cm^2'])
            
            mp_ratio = float(row['A/B'])
          
            label = torch.tensor(float(mp_ratio >= 0.5), dtype=torch.float32)
            
            measurements = torch.tensor([midbrain, pons], dtype=torch.float32)
            
            return image, measurements, label
            
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            return (torch.zeros((3, 224, 224), dtype=torch.float32), 
                    torch.zeros(2, dtype=torch.float32), 
                    torch.tensor(0.0, dtype=torch.float32))
            

def train_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize metrics and tracking variables
    metrics = MPMetrics()
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    patience = 20
    patience_counter = 0
    
    # Calculate class weights
    all_labels = []
    for _, _, labels in train_loader:
        all_labels.extend(labels.numpy())
    
    num_pos = sum(all_labels)
    num_neg = len(all_labels) - num_pos + 1e-6  # Add small epsilon to prevent division by zero
    pos_weight = torch.tensor([num_neg/num_pos]).to(device)
    
    print(f"\nClass balance - Positive weight: {pos_weight.item():.2f}")
    print(f"Positive samples: {num_pos}, Negative samples: {num_neg}")
    
    criterion_measurements = nn.SmoothL1Loss(beta=0.01)
    criterion_classification = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
        reduction='none'
    )
    
    # Initialize optimizer with different learning rates for different parts
    params = [
        {'params': model.features.parameters(), 'lr': 1e-5},
        {'params': model.dense1.parameters(), 'lr': 2e-5},
        {'params': model.dense2.parameters(), 'lr': 2e-5},
        {'params': model.dense3.parameters(), 'lr': 3e-5},
        {'params': model.measurement_head.parameters(), 'lr': 5e-5},
    ]
    
    optimizer = torch.optim.AdamW(params, weight_decay=0.01)
    
    # Use OneCycleLR scheduler instead
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-4, 2e-4, 2e-4, 3e-4, 5e-4],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    try:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Training loop
            for batch_idx, (images, targets, labels) in enumerate(train_loader):
                images = images.to(device).float()
                targets = targets.to(device).float()
                labels = labels.to(device).float().view(-1, 1)
                
                optimizer.zero_grad()
                
                # Forward pass
                measurements, predictions, mp_ratios = model(images)
                
                # Calculate losses
                measurement_loss = criterion_measurements(measurements, targets)
                
                # Classification loss with focal characteristics
                class_loss = criterion_classification(predictions, labels)
                pt = torch.where(labels == 1, torch.sigmoid(predictions), 1 - torch.sigmoid(predictions))
                focal_weight = (1 - pt) ** 2
                class_loss = (class_loss * focal_weight).mean()
                
                # Combined loss
                loss = 0.4 * measurement_loss + 0.6 * class_loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                scheduler.step()
                
                # Track metrics
                train_loss += loss.item()
                predictions_binary = (torch.sigmoid(predictions) > 0.5).float()
                train_correct += (predictions_binary == labels).sum().item()
                train_total += labels.size(0)
                
                if (batch_idx + 1) % 5 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Batch {batch_idx+1}/{len(train_loader)}")
                    print(f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
                    print(f"M/P ratios: min={mp_ratios.min().item():.3f}, max={mp_ratios.max().item():.3f}")
                    print(f"Predictions: {predictions_binary.mean().item():.3f}")
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            val_ratios = []
            
            with torch.no_grad():
                for images, targets, labels in val_loader:
                    images = images.to(device).float()
                    targets = targets.to(device).float()
                    labels = labels.to(device).float().view(-1, 1)
                    
                    measurements, predictions, mp_ratios = model(images)
                    
                    val_loss += criterion_measurements(measurements, targets).item()
                    val_preds.extend(torch.sigmoid(predictions).cpu().numpy().flatten())
                    val_labels.extend(labels.cpu().numpy().flatten())
                    val_ratios.extend(mp_ratios.cpu().numpy())
            
            # Update metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            accuracy = accuracy_score(
                np.array(val_labels),
                (np.array(val_preds) > 0.5).astype(int)
            )
            
            metrics.update(avg_train_loss, avg_val_loss, val_preds, 
                         val_labels, val_ratios, accuracy)
            
            # Print epoch results
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                print("New best model saved!")
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        
        model.load_state_dict(best_model_state)
        return metrics
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        traceback.print_exc()
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return metrics

def test_model(model, image_path, transform):
    """Modified test function to handle three return values."""
    model.eval()
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            # Get model predictions - now handling three return values
            measurements, pred, mp_ratio = model(image)
            prediction = 'PD' if torch.sigmoid(pred).item() > 0.5 else 'No-PD'
            
            # Extract values
            mp_ratio_value = mp_ratio.item()
            confidence = torch.sigmoid(pred).item()
            
            return {
                'Image': os.path.basename(image_path),
                'M/P Ratio': mp_ratio_value,
                'Prediction': prediction,
                'Confidence': confidence
            }
            
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def create_validation_visualizations(val_results_df, optimal_threshold, save_dir):
    """Create essential visualizations for validation data only."""
    try:
        if val_results_df is None or len(val_results_df) == 0:
            print("No validation data available for visualization")
            return
            
        print("\nGenerating validation visualizations...")
        
        # Prepare validation data
        val_true = np.array([1 if 'PD_UnMarked' in img else 0 
                           for img in val_results_df['Image']])
        val_ratios = val_results_df['M/P Ratio'].values
        val_pred = (val_ratios >= optimal_threshold).astype(int)
        
        # 1. Validation Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm_val = confusion_matrix(val_true, val_pred)
        
        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No-PD', 'PD'],
                   yticklabels=['No-PD', 'PD'])
        plt.title('Validation Data - Confusion Matrix', fontsize=12, pad=15)
        plt.xlabel('Predicted Label', fontsize=10)
        plt.ylabel('True Label', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'validation_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Validation ROC Curve
        plt.figure(figsize=(10, 8))
        fpr_val, tpr_val, _ = roc_curve(val_true, val_ratios)
        roc_auc_val = auc(fpr_val, tpr_val)
        plt.plot(fpr_val, tpr_val, 'r-', 
                label=f'Validation (AUC = {roc_auc_val:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=10)
        plt.ylabel('True Positive Rate', fontsize=10)
        plt.title('Validation Data - ROC Curve', fontsize=12, pad=15)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'validation_roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate and return validation metrics
        val_true = (val_results_df['True_Label'] == 'PD').astype(int)
        val_pred = (val_results_df['Prediction'] == 'PD').astype(int)
        metrics = {
            'accuracy': accuracy_score(val_true, val_pred),
            'sensitivity': recall_score(val_true, val_pred),
            'specificity': recall_score(val_true, val_pred, pos_label=0),
            'precision': precision_score(val_true, val_pred),
            'f1': f1_score(val_true, val_pred),
            'roc_auc': roc_auc_val
        }
        
        # Save metrics summary
        metrics_summary = f"""
Validation Performance Metrics:
-----------------------------
Accuracy: {metrics['accuracy']:.4f}
Sensitivity (Recall): {metrics['sensitivity']:.4f}
Specificity: {metrics['specificity']:.4f}
Precision: {metrics['precision']:.4f}
F1 Score: {metrics['f1']:.4f}
ROC AUC: {metrics['roc_auc']:.4f}
"""
        metrics_path = os.path.join(save_dir, 'validation_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(metrics_summary)
            
        return roc_auc_val, metrics
        
    except Exception as e:
        print(f"Error in create_validation_visualizations: {str(e)}")
        traceback.print_exc()
        return None, None


def main():
    try:
        # Define paths
        train_path = r"C:\Users\omsha\Desktop\densenet\PDM"
        val_path = r"C:\Users\omsha\Desktop\densenet\validation1"
        labeled_csv_path = r"C:\Users\omsha\Desktop\densenet\PD_data\PD_mark_data.csv"
        vis_path = r"C:\Users\omsha\Desktop\densenet\visualizations\capstone_1"
        
        # Create visualization directory
        os.makedirs(vis_path, exist_ok=True)
        
        # Load and print CSV data
        print("\nChecking CSV file...")
        df = pd.read_csv(labeled_csv_path)
        print("CSV columns:", df.columns.tolist())
        
        # Get training and validation images
        train_files = [f for f in os.listdir(train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        val_files = [f for f in os.listdir(val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Randomly select validation images
        if len(val_files) > 15:
            val_files = np.random.choice(val_files, size=100, replace=False)
        
        print("\nData Statistics:")
        print(f"Training directory: {len(train_files)} images")
        print(f"Selected validation images: {len(val_files)} images")
        
        # Process training data
        train_subjects = [f.split('.')[0] for f in train_files]
        train_df = df[df['subject'].isin(train_subjects)].copy()
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
        
        # Set up data loaders
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        train_dataset = MPDataset(data=train_df, img_dir=train_path, transform=transform)
        val_dataset = MPDataset(data=val_df, img_dir=train_path, transform=transform)
        
        batch_size = min(16, len(train_dataset), len(val_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        # Initialize and train model
        print("\nInitializing and training model...")
        model = MPDenseNet()
        metrics = train_model(model, train_loader, val_loader)
        
        if metrics is None:
            print("Error: Training failed to return valid metrics")
            return
        
        _, _, optimal_threshold, _ = analyze_pd_data(metrics, 43)  
            
        # Process validation images
        print("\nProcessing validation images...")
        model.eval()
        val_results = []
        
        for val_file in val_files:
            try:
                img_path = os.path.join(val_path, val_file)
                result = test_model(model, img_path, transform)
                if result is not None:
                    result['True_Label'] = 'PD' if 'PD_UnMarked' in val_file else 'No-PD'
                    val_results.append(result)
            except Exception as e:
                print(f"Error processing validation image {val_file}: {str(e)}")
                continue
        
        # Generate validation visualizations and metrics
        if val_results:
            val_results_df = pd.DataFrame(val_results)
            roc_auc_val, val_metrics = create_validation_visualizations(val_results_df, optimal_threshold, vis_path)
            
            if roc_auc_val is not None and val_metrics is not None:
                print("\nValidation Performance Summary:")
                print(f"ROC AUC: {roc_auc_val:.4f}")
                print(f"Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"Sensitivity: {val_metrics['sensitivity']:.4f}")
                print(f"Specificity: {val_metrics['specificity']:.4f}")
                print(f"F1 Score: {val_metrics['f1']:.4f}")
                
                
                print(f"\nResults saved to: {vis_path}")

    except Exception as e:
        print(f"\nAn error occurred in main: {str(e)}")
        traceback.print_exc()
        return None
    
    finally:
        print("\nProgram completed.")

if __name__ == "__main__":
    main()

