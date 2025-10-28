# -*- coding: utf-8 -*-
"""
KaraciÄŸer HastalÄ±ÄŸÄ± Tahmini - ML + SHAP Analizi
Bu script, karaciÄŸer hastalÄ±ÄŸÄ± veri setini analiz eder ve SHAP ile aÃ§Ä±klanabilir ML modeli oluÅŸturur.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, roc_curve, auc)
import shap
import warnings
warnings.filterwarnings('ignore')

# TÃ¼rkÃ§e karakter desteÄŸi iÃ§in
plt.rcParams['font.family'] = 'DejaVu Sans'

class LiverDiseasePredictor:
    """KaraciÄŸer hastalÄ±ÄŸÄ± tahmin modeli"""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, file_path='data/indian_liver_patient.csv'):
        """Veri setini yÃ¼kle"""
        print("Veri seti yÃ¼kleniyor...")
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            # EÄŸer dosya yoksa UCI repository'den indir
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv'
            df = pd.read_csv(url, header=None)
            df.columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
                         'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 
                         'Aspartate_Aminotransferase', 'Total_Protiens', 
                         'Albumin', 'Albumin_and_Globulin_Ratio', 'Dataset']
            df.to_csv(file_path, index=False, encoding='utf-8')
        
        print(f"Veri seti boyutu: {df.shape}")
        print(f"\nÄ°lk 5 satÄ±r:\n{df.head()}")
        return df
    
    def preprocess_data(self, df):
        """Veri Ã¶n iÅŸleme"""
        print("\n" + "="*50)
        print("VERÄ° Ã–N Ä°ÅLEME")
        print("="*50)
        
        # Eksik deÄŸerleri kontrol et
        print(f"\nEksik deÄŸerler:\n{df.isnull().sum()}")
        
        # Eksik deÄŸerleri medyan ile doldur
        df = df.fillna(df.median(numeric_only=True))
        
        # Gender kolonunu encode et
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])
        
        # Target deÄŸiÅŸkeni ayarla (1: hasta, 0: saÄŸlÄ±klÄ±)
        if 'Dataset' in df.columns:
            df['Target'] = df['Dataset'].apply(lambda x: 1 if x == 1 else 0)
            df = df.drop('Dataset', axis=1)
        
        # Feature ve target ayÄ±rma
        X = df.drop('Target', axis=1)
        y = df['Target']
        
        self.feature_names = X.columns.tolist()
        print(f"\nÃ–zellikler: {self.feature_names}")
        print(f"Hedef deÄŸiÅŸken daÄŸÄ±lÄ±mÄ±:\n{y.value_counts()}")
        
        return X, y
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """TÃ¼m modelleri eÄŸit ve deÄŸerlendir"""
        print("\n" + "="*50)
        print("MODEL EÄÄ°TÄ°MÄ° VE DEÄERLENDÄ°RME")
        print("="*50)
        
        results = {}
        best_accuracy = 0
        
        for name, model in self.models.items():
            print(f"\n{name} eÄŸitiliyor...")
            
            # Model eÄŸitimi
            model.fit(X_train, y_train)
            
            # Tahmin
            y_pred = model.predict(X_test)
            
            # Metrikler
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # En iyi modeli seÃ§
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name
        
        print(f"\n\nEn iyi model: {self.best_model_name} (Accuracy: {best_accuracy:.4f})")
        return results
    
    def plot_model_comparison(self, results):
        """Model karÅŸÄ±laÅŸtÄ±rma grafikleri"""
        print("\nModel karÅŸÄ±laÅŸtÄ±rma grafikleri oluÅŸturuluyor...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            values = [results[m][metric] for m in models]
            bars = ax.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.set_title(f'Model Karsilastirmasi - {metric.capitalize()}', fontsize=14)
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            # DeÄŸerleri barlarÄ±n Ã¼zerine yaz
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        print("Model karÅŸÄ±laÅŸtÄ±rma grafiÄŸi kaydedildi: results/model_comparison.png")
        plt.close()
    
    def plot_confusion_matrices(self, results, y_test):
        """Confusion matrix grafikleri"""
        print("\nConfusion matrix grafikleri oluÅŸturuluyor...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, (name, result) in enumerate(results.items()):
            ax = axes[idx // 2, idx % 2]
            cm = confusion_matrix(y_test, result['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Saglikli', 'Hasta'],
                       yticklabels=['Saglikli', 'Hasta'])
            ax.set_title(f'{name} - Confusion Matrix', fontsize=14)
            ax.set_ylabel('Gercek Deger', fontsize=12)
            ax.set_xlabel('Tahmin', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix grafiÄŸi kaydedildi: results/confusion_matrices.png")
        plt.close()
    
    def shap_analysis(self, X_train, X_test):
        """SHAP analizi ile model aÃ§Ä±klanabilirliÄŸi"""
        print("\n" + "="*50)
        print("SHAP ANALÄ°ZÄ°")
        print("="*50)
        
        print(f"\n{self.best_model_name} modeli iÃ§in SHAP analizi yapÄ±lÄ±yor...")
        
        # SHAP explainer oluÅŸtur
        if self.best_model_name in ['Random Forest', 'Gradient Boosting']:
            explainer = shap.TreeExplainer(self.best_model)
            shap_values = explainer.shap_values(X_test)
            
            # Binary classification iÃ§in shap_values listesi gelirse ikinci elemanÄ± al
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # KernelExplainer diÄŸer modeller iÃ§in
            explainer = shap.KernelExplainer(self.best_model.predict_proba, 
                                            shap.sample(X_train, 100))
            shap_values = explainer.shap_values(X_test[:100])
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            X_test = X_test[:100]
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=self.feature_names, 
                         show=False)
        plt.title('SHAP Summary Plot - Ozellik Onemi', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('results/shap_summary.png', dpi=300, bbox_inches='tight')
        print("SHAP summary plot kaydedildi: results/shap_summary.png")
        plt.close()
        
        # Bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=self.feature_names,
                         plot_type='bar', show=False)
        plt.title('SHAP Bar Plot - Ortalama Ozellik Onemi', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('results/shap_bar.png', dpi=300, bbox_inches='tight')
        print("SHAP bar plot kaydedildi: results/shap_bar.png")
        plt.close()
        
        # Feature importance tablosu
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('Importance', ascending=False)
        
        print("\n\nÃ–zellik Ã–nem SÄ±ralamasÄ±:")
        print(feature_importance.to_string(index=False))
        
        return shap_values, explainer
    
    def generate_report(self, results, X_test, y_test):
        """DetaylÄ± rapor oluÅŸtur"""
        print("\n" + "="*50)
        print("DETAYLI RAPOR OLUÅTURULUYOR")
        print("="*50)
        
        report_text = []
        report_text.append("=" * 70)
        report_text.append("KARACIÄER HASTALIÄI TAHMÄ°N MODELÄ° - PERFORMANS RAPORU")
        report_text.append("=" * 70)
        report_text.append("")
        
        for name, result in results.items():
            report_text.append(f"\n{name}:")
            report_text.append("-" * 50)
            report_text.append(f"Accuracy:  {result['accuracy']:.4f}")
            report_text.append(f"Precision: {result['precision']:.4f}")
            report_text.append(f"Recall:    {result['recall']:.4f}")
            report_text.append(f"F1-Score:  {result['f1']:.4f}")
            report_text.append(f"CV Score:  {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
            report_text.append("")
            report_text.append("Classification Report:")
            report_text.append(classification_report(y_test, result['y_pred']))
        
        report_text.append("\n" + "=" * 70)
        report_text.append(f"EN Ä°YÄ° MODEL: {self.best_model_name}")
        report_text.append("=" * 70)
        
        report = "\n".join(report_text)
        
        # Raporu kaydet
        with open('results/model_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\nRapor kaydedildi: results/model_report.txt")
        print("\n" + report)

def main():
    """Ana fonksiyon"""
    print("=" * 70)
    print("KARACIÄER HASTALIÄI TAHMÄ°N MODELÄ° - ML + SHAP ANALÄ°ZÄ°")
    print("=" * 70)
    
    # Model oluÅŸtur
    predictor = LiverDiseasePredictor()
    
    # Veri yÃ¼kle
    df = predictor.load_data()
    
    # Veri Ã¶n iÅŸleme
    X, y = predictor.preprocess_data(df)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Veriyi Ã¶lÃ§eklendir
    X_train_scaled = predictor.scaler.fit_transform(X_train)
    X_test_scaled = predictor.scaler.transform(X_test)
    
    # DataFrame'e Ã§evir (SHAP iÃ§in)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=predictor.feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=predictor.feature_names)
    
    # Modelleri eÄŸit
    results = predictor.train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # GÃ¶rselleÅŸtirmeler
    predictor.plot_model_comparison(results)
    predictor.plot_confusion_matrices(results, y_test)
    
    # SHAP analizi
    shap_values, explainer = predictor.shap_analysis(X_train_scaled, X_test_scaled)
    
    # Rapor oluÅŸtur
    predictor.generate_report(results, y_test)
    
    print("\n" + "=" * 70)
    print("ANALIZ TAMAMLANDI!")
    print("=" * 70)
    print("\nSonuÃ§lar 'results/' klasÃ¶rÃ¼nde kaydedildi.")
    print("- model_comparison.png: Model karÅŸÄ±laÅŸtÄ±rma grafikleri")
    print("- confusion_matrices.png: Confusion matrix grafikleri")
    print("- shap_summary.png: SHAP Ã¶zet grafiÄŸi")
    print("- shap_bar.png: SHAP Ã¶zellik Ã¶nemi grafiÄŸi")
    print("- model_report.txt: DetaylÄ± performans raporu")

if __name__ == "__main__":
    main()
