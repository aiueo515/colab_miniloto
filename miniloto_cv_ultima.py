# -*- coding: utf-8 -*-
# ミニロト時系列交差検証バックグラウンド実行システム - Part1: 基本システム
# 改修版: 段階的保存・継続実行・スコア順実行対応

import pandas as pd
import numpy as np
import os
import pickle
import time
import traceback
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import clone
import json


# ======================================================================
# 1. 段階的時系列CV保存システム
# ======================================================================

class IncrementalTimeSeriesCV:
    def __init__(self, save_interval=10):
        self.save_interval = save_interval  # 10件ごとに保存
        self.cv_dir = "miniloto_models/cv_results"
        self.models_dir = "miniloto_models/models"
        
        # ディレクトリ作成
        os.makedirs(self.cv_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Google Drive連携
        try:
            from google.colab import drive
            self.drive_available = True
            self.drive_cv_dir = "/content/drive/MyDrive/miniloto_models/cv_results"
            self.drive_models_dir = "/content/drive/MyDrive/miniloto_models/models"
            os.makedirs(self.drive_cv_dir, exist_ok=True)
            os.makedirs(self.drive_models_dir, exist_ok=True)
        except ImportError:
            self.drive_available = False

    def check_drive_mount_status(self):
        """Driveマウント状態の正確な確認（修正版）"""
        try:
            print("🔍 Drive認証・マウント状態の詳細確認...")
            
            # Step 1: /content/drive の存在確認
            drive_path = '/content/drive'
            mydrive_path = '/content/drive/MyDrive'
            
            print(f"📁 {drive_path} 存在: {'✅' if os.path.exists(drive_path) else '❌'}")
            print(f"📁 {mydrive_path} 存在: {'✅' if os.path.exists(mydrive_path) else '❌'}")
            
            if not os.path.exists(mydrive_path):
                print("❌ MyDriveディレクトリが存在しません")
                return False
            
            # Step 2: MyDriveディレクトリの内容確認
            try:
                items = os.listdir(mydrive_path)
                print(f"📂 MyDrive内のアイテム数: {len(items)}")
                
                # 内容を少し表示（プライバシーに配慮して最初の3個まで）
                if items:
                    print(f"📋 内容例: {items[:3]}")
                
                # Step 3: 実際のファイルアクセステスト
                print("🔍 実際のファイルアクセステスト実行...")
                
                # テストファイル作成・削除テスト
                test_file_path = os.path.join(mydrive_path, 'test_mount_check.tmp')
                try:
                    # ファイル作成テスト
                    with open(test_file_path, 'w') as f:
                        f.write('test')
                    
                    # ファイル読み込みテスト
                    with open(test_file_path, 'r') as f:
                        content = f.read()
                    
                    # ファイル削除
                    os.remove(test_file_path)
                    
                    if content == 'test':
                        print("✅ ファイルアクセステスト: 成功")
                        return True
                    else:
                        print("❌ ファイルアクセステスト: 読み書き不整合")
                        return False
                        
                except PermissionError as e:
                    print(f"❌ ファイルアクセステスト: 権限エラー ({e})")
                    return False
                except Exception as e:
                    print(f"❌ ファイルアクセステスト: エラー ({e})")
                    return False
                
            except PermissionError as e:
                print(f"❌ MyDriveアクセス: 権限エラー ({e})")
                return False
            except Exception as e:
                print(f"❌ MyDriveアクセス: エラー ({e})")
                return False
                
        except Exception as e:
            print(f"❌ マウント状態確認エラー: {e}")
            return False

    

    def debug_find_features_file(self):
        """特徴量ファイルを詳細検索してパスを特定"""
        import os
        
        print("🔍 特徴量ファイル詳細検索デバッグ")
        print("=" * 50)
        
        # 1. 基本パス確認
        print("📂 基本パス確認:")
        base_paths = [
            "/content/drive/MyDrive/",
            "/content/drive/MyDrive/miniloto_predictor_ultra/",
            "/content/drive/MyDrive/miniloto_models/",
        ]
        
        for base_path in base_paths:
            exists = os.path.exists(base_path)
            print(f"  {base_path}: {'✅' if exists else '❌'}")
            
            if exists:
                try:
                    contents = os.listdir(base_path)
                    miniloto_items = [item for item in contents if 'miniloto' in item.lower()]
                    print(f"    ミニロト関連: {miniloto_items}")
                except Exception as e:
                    print(f"    リスト取得エラー: {e}")
        
        # 2. 現在試しているパス確認
        print(f"\n📁 現在試しているパス:")
        current_paths = [
            "/content/drive/MyDrive/miniloto_predictor_ultra/miniloto_models/features/features_cache.pkl",
            "/content/drive/MyDrive/miniloto_models/features/features_cache.pkl",
        ]
        
        for path in current_paths:
            exists = os.path.exists(path)
            print(f"  {path}")
            print(f"    存在: {'✅' if exists else '❌'}")
            
            # 段階的にパスを確認
            path_parts = path.split('/')
            for i in range(3, len(path_parts)):
                partial_path = '/'.join(path_parts[:i+1])
                partial_exists = os.path.exists(partial_path)
                print(f"    {partial_path}: {'✅' if partial_exists else '❌'}")
                if not partial_exists:
                    break
        
        # 3. features_cache.pklファイルを再帰的に検索
        print(f"\n🔍 features_cache.pkl再帰検索:")
        search_roots = [
            "/content/drive/MyDrive/",
        ]
        
        found_files = []
        for root_path in search_roots:
            if os.path.exists(root_path):
                for root, dirs, files in os.walk(root_path):
                    for file in files:
                        if file == "features_cache.pkl":
                            full_path = os.path.join(root, file)
                            found_files.append(full_path)
                            print(f"  ✅ 発見: {full_path}")
        
        if found_files:
            print(f"\n🎉 {len(found_files)}個のfeatures_cache.pklファイルを発見!")
            return found_files
        else:
            print(f"\n😥 features_cache.pklファイルが見つかりませんでした")
            return []
    

    def force_mount_drive(self):
        """CV用Drive強制マウント（修正版）"""
        try:
            from google.colab import drive
            import os
            from datetime import datetime
            
            print("🔧 CV用Drive強制マウント開始...")
            
            # Step 1: 現在の状態確認
            mount_status = self.check_drive_mount_status()
            print(f"📊 現在のマウント状態: {'✅マウント済み' if mount_status else '❌未マウント'}")
            
            if mount_status:
                print("✅ Drive は正常にマウント済みです")
                return True
            
            # Step 2: 既存のdriveディレクトリの処理
            if os.path.exists('/content/drive'):
                print("🔧 既存のdriveディレクトリを処理中...")
                
                # 既存ディレクトリをバックアップに移動
                backup_name = f'/content/drive_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                try:
                    os.rename('/content/drive', backup_name)
                    print(f"📦 既存ディレクトリを {backup_name} にバックアップ")
                except Exception as e:
                    print(f"⚠️ ディレクトリバックアップ失敗: {e}")
                    # バックアップに失敗した場合は削除を試行
                    try:
                        import shutil
                        shutil.rmtree('/content/drive')
                        print("🗑️ 既存ディレクトリを削除")
                    except Exception as e2:
                        print(f"❌ ディレクトリ削除も失敗: {e2}")
                        print("🔄 手動でディレクトリを削除してから再実行してください")
                        return False
            
            # Step 3: 新規マウント実行
            print("🔐 Drive 認証とマウントを実行...")
            print("📱 ブラウザで認証画面が開きます。Google アカウントでログインしてください。")
            
            try:
                drive.mount('/content/drive', force_remount=True)
                print("✅ マウントコマンド実行完了")
            except Exception as e:
                print(f"❌ マウントコマンド実行エラー: {e}")
                return False
            
            # Step 4: マウント成功確認
            final_status = self.check_drive_mount_status()
            
            if final_status:
                print("🎉 Drive マウント成功!")
                
                # ミニロト関連フォルダ確認
                try:
                    items = os.listdir('/content/drive/MyDrive')
                    miniloto_items = [item for item in items if 'miniloto' in item.lower()]
                    
                    if miniloto_items:
                        print(f"📁 ミニロト関連フォルダ: {miniloto_items}")
                    else:
                        print("⚠️ ミニロト関連フォルダが見つかりません")
                        print("💡 メインシステムを先に実行してフォルダを作成してください")
                except Exception as e:
                    print(f"⚠️ フォルダ確認エラー: {e}")
                
                return True
            else:
                print("❌ Drive マウント後も正常に認識されません")
                print("🔄 手動でマウントを確認してください")
                return False
                
        except ImportError:
            print("❌ Google Colab環境ではありません")
            return False
        except Exception as e:
            print(f"❌ Drive マウントエラー: {e}")
            return False

    def load_features_for_cv(self):
        """CV用特徴量読み込み（y形状修正版）"""
        try:
            print(f"📂 現在の作業ディレクトリ: {os.getcwd()}")
            
            # Step 1: Drive マウント確認・実行
            if not self.force_mount_drive():
                print("❌ Drive マウントに失敗しました")
                return None, None
            
            # Step 2: main_v4の実際の保存先に合わせた検索パス（優先順修正）
            search_files = [
                # ★★★ main_v4の実際の保存パス（最優先）★★★
                "/content/drive/MyDrive/miniloto_models/features/features_cache.pkl",
                
                # ローカルパス（main_v4準拠）
                "miniloto_models/features/features_cache.pkl",
                
                # 従来の検索パス（互換性のため）
                "/content/drive/MyDrive/miniloto_predictor_ultra/miniloto_models/features/features_cache.pkl",
                "/content/drive/MyDrive/miniloto_predictor_ultra/features/features_cache.pkl",
                "/content/drive/MyDrive/features/features_cache.pkl",
            ]
            
            print("\n🔍 特徴量ファイル検索（main_v4準拠パス優先）:")
            
            # 再帰検索も追加
            found_files = []
            
            # まず固定パスを確認
            for i, file_path in enumerate(search_files, 1):
                print(f"  {i}. 試行: {file_path}")
                if os.path.exists(file_path):
                    abs_path = os.path.abspath(file_path)
                    if abs_path not in found_files:
                        found_files.append(abs_path)
                    print(f"     ✅ 存在します")
                else:
                    print(f"     ❌ 存在しません")
            
            # 再帰検索（念のため）
            print("\n🔍 再帰検索実行:")
            search_roots = [
                "/content/drive/MyDrive/miniloto_models",  # main_v4の実際の保存先
                "/content/drive/MyDrive/miniloto_predictor_ultra",
                "/content/drive/MyDrive"
            ]
            
            for search_root in search_roots:
                if os.path.exists(search_root):
                    print(f"  検索中: {search_root}")
                    try:
                        for root, dirs, files in os.walk(search_root):
                            if "features_cache.pkl" in files:
                                full_path = os.path.join(root, "features_cache.pkl")
                                if full_path not in found_files:
                                    found_files.append(full_path)
                                print(f"    ✅ 発見: {full_path}")
                    except Exception as e:
                        print(f"    ❌ 検索エラー: {e}")
            
            # Step 3: ファイル読み込み
            if not found_files:
                print("❌ 特徴量ファイルが見つかりません")
                print("🔧 先にメインシステム（run_ultra_maximum_precision_prediction）を実行してください")
                return None, None
            
            print(f"\n📖 {len(found_files)}個のファイルから読み込み試行:")
            
            for i, file_path in enumerate(found_files, 1):
                print(f"\n  {i}. 読み込み試行: {file_path}")
                
                try:
                    with open(file_path, 'rb') as f:
                        features_data = pickle.load(f)
                    
                    X = features_data["X"]
                    y = features_data["y"]
                    
                    # データ形状確認・修正
                    print(f"    📊 読み込み成功! X: {X.shape}, y: {y.shape}")
                    
                    # ===== yの形状確認・修正（ミニロト用）=====
                    if hasattr(y, 'ndim'):
                        if y.ndim == 1:
                            print(f"    ⚠️ yが1次元です: {y.shape}")
                            print(f"    🔧 ミニロト用31次元バイナリラベルに変換中...")
                            
                            # 1次元の場合、正しい形状に変換できないため警告
                            print(f"    ❌ 1次元のyは31次元マルチラベルに変換不可")
                            print(f"    💡 メインシステムで特徴量を再生成してください")
                            print(f"    🔧 run_ultra_maximum_precision_prediction() を実行")
                            return None, None
                            
                        elif y.ndim == 2:
                            if y.shape[1] == 31:
                                print(f"    ✅ 正しい形状: {y.shape} (31次元マルチラベル)")
                            elif y.shape[1] == 1:
                                print(f"    ⚠️ yが2次元だが列数が1: {y.shape}")
                                print(f"    ❌ 31次元マルチラベルではありません")
                                print(f"    💡 メインシステムで特徴量を再生成してください")
                                return None, None
                            else:
                                print(f"    ⚠️ 予期しない形状: {y.shape}")
                                print(f"    ❌ ミニロト用31次元と一致しません")
                                print(f"    💡 メインシステムで特徴量を再生成してください")
                                return None, None
                        else:
                            print(f"    ❌ 予期しない次元数: {y.ndim}次元")
                            print(f"    💡 メインシステムで特徴量を再生成してください")
                            return None, None
                    else:
                        print(f"    ❌ yにndim属性がありません")
                        print(f"    💡 メインシステムで特徴量を再生成してください")
                        return None, None
                    
                    # メタ情報
                    feature_version = features_data.get("feature_version", "unknown")
                    timestamp = features_data.get("timestamp", "unknown")
                    print(f"    📊 バージョン: {feature_version}")
                    print(f"    🕒 作成日時: {timestamp}")
                    print(f"    💾 使用ファイル: {file_path}")
                    print(f"    ✅ CV用データ準備完了: X{X.shape}, y{y.shape}")
                    
                    return X, y
                    
                except Exception as e:
                    print(f"    ❌ 読み込みエラー: {e}")
                    continue
            
            print("❌ すべてのファイルで読み込みに失敗しました")
            return None, None
            
        except Exception as e:
            print(f"❌ 特徴量読み込みエラー: {e}")
            import traceback
            print(f"詳細: {traceback.format_exc()}")
            return None, None

    def debug_path_investigation(self):
        """パス問題の詳細調査"""
        try:
            print("🔍 === パス問題詳細調査 ===")
            
            # main_v4の実際の保存先確認
            main_v4_paths = [
                "/content/drive/MyDrive/miniloto_models/features/features_cache.pkl",
                "/content/miniloto_models/features/features_cache.pkl"
            ]
            
            print("📁 main_v4準拠パス確認:")
            for path in main_v4_paths:
                exists = os.path.exists(path)
                print(f"  {path}")
                print(f"    存在: {'✅' if exists else '❌'}")
                
                if exists:
                    try:
                        size = os.path.getsize(path)
                        mtime = datetime.fromtimestamp(os.path.getmtime(path))
                        print(f"    サイズ: {size:,} bytes")
                        print(f"    更新: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                    except Exception as e:
                        print(f"    詳細取得エラー: {e}")
            
            # ディレクトリ構造確認
            print("\n📁 ディレクトリ構造確認:")
            check_dirs = [
                "/content/drive/MyDrive/miniloto_models",
                "/content/drive/MyDrive/miniloto_predictor_ultra",
                "/content/miniloto_models"
            ]
            
            for check_dir in check_dirs:
                print(f"\n  {check_dir}:")
                if os.path.exists(check_dir):
                    try:
                        items = os.listdir(check_dir)
                        print(f"    ✅ 存在 ({len(items)}個のアイテム)")
                        for item in items:
                            item_path = os.path.join(check_dir, item)
                            if os.path.isdir(item_path):
                                print(f"      📁 {item}/")
                            else:
                                print(f"      📄 {item}")
                    except Exception as e:
                        print(f"    ❌ リスト取得エラー: {e}")
                else:
                    print(f"    ❌ 存在しません")
            
            return True
            
        except Exception as e:
            print(f"❌ パス調査エラー: {e}")
            return False

    def continue_cv_from_checkpoint(self, model_name):
        """中断点からCV継続（バージョンチェック付き）"""
        try:
            # 特徴量バージョン互換性チェック
            is_compatible, start_split, current_results = self.check_feature_version_compatibility(model_name)
            
            if is_compatible and start_split > 0:
                print(f"🔄 {model_name} 継続実行: {start_split}件目から再開")
                return start_split, current_results
            else:
                print(f"🆕 {model_name} 新規CV開始")
                return 0, []
                
        except Exception as e:
            print(f"⚠️ {model_name} 進捗読み込みエラー: {e}")
            return 0, []

    

    def save_cv_progress(self, model_name, current_results, completed_splits, total_splits):
        """CV進捗保存（特徴量バージョン付き）"""
        try:
            current_version = self._calculate_feature_version()
            
            progress_data = {
                'model_name': model_name,
                'current_results': current_results,
                'completed_splits': completed_splits,
                'total_splits': total_splits,
                'timestamp': datetime.now(),
                'completion_rate': completed_splits / total_splits if total_splits > 0 else 0,
                'feature_version': current_version  # バージョン情報追加
            }
            
            # ローカル保存
            progress_file = os.path.join(self.cv_dir, f"{model_name}_cv_progress.pkl")
            with open(progress_file, 'wb') as f:
                pickle.dump(progress_data, f)
            
            # Drive保存
            if self.drive_available:
                drive_progress_file = os.path.join(self.drive_cv_dir, f"{model_name}_cv_progress.pkl")
                try:
                    with open(drive_progress_file, 'wb') as f:
                        pickle.dump(progress_data, f)
                except Exception as e:
                    print(f"⚠️ Drive保存失敗: {e}")
            
            print(f"💾 {model_name} 進捗保存: {completed_splits}/{total_splits} ({progress_data['completion_rate']*100:.1f}%) v{current_version}")
            return True
            
        except Exception as e:
            print(f"❌ {model_name} 進捗保存エラー: {e}")
            return False

    
    def update_model_with_cv_results(self, model_name, enhanced_model_data):
        """CV結果でモデル更新"""
        try:
            # 元のモデルファイル読み込み
            original_model_file = os.path.join(self.models_dir, f"{model_name}.pkl")
            
            if not os.path.exists(original_model_file):
                print(f"❌ {model_name} 元モデルが見つかりません")
                return False
            
            with open(original_model_file, 'rb') as f:
                original_data = pickle.load(f)
            
            # CV結果で強化
            enhanced_data = original_data.copy()
            enhanced_data.update({
                'cv_enhanced': True,
                'cv_score': enhanced_model_data.get('cv_score', 0),
                'cv_std': enhanced_model_data.get('cv_std', 0),
                'cv_results': enhanced_model_data.get('cv_results', []),
                'enhanced_model': enhanced_model_data.get('model', original_data['model']),
                'enhanced_scaler': enhanced_model_data.get('scaler', original_data['scaler']),
                'enhancement_timestamp': datetime.now()
            })
            
            # 強化モデル保存
            enhanced_file = os.path.join(self.models_dir, f"{model_name}_cv_enhanced.pkl")
            with open(enhanced_file, 'wb') as f:
                pickle.dump(enhanced_data, f)
            
            # Drive保存
            if self.drive_available:
                drive_enhanced_file = os.path.join(self.drive_models_dir, f"{model_name}_cv_enhanced.pkl")
                try:
                    with open(drive_enhanced_file, 'wb') as f:
                        pickle.dump(enhanced_data, f)
                except Exception as e:
                    print(f"⚠️ Drive保存失敗: {e}")
            
            cv_score = enhanced_model_data.get('cv_score', 0)
            print(f"✅ {model_name} CV強化完了 (CVスコア: {cv_score:.4f})")
            return True
            
        except Exception as e:
            print(f"❌ {model_name} モデル更新エラー: {e}")
            return False
    
    def load_available_models_for_cv(self):
        """CV実行用モデル読み込み"""
        try:
            models_data = {}

            models_dir_to_use = self.drive_models_dir if self.drive_available else self.models_dir
            
            # 利用可能モデルファイルを検索
            for filename in os.listdir(models_dir_to_use):
                if filename.endswith('.pkl') and not filename.endswith('_cv_enhanced.pkl'):
                    model_name = filename.replace('.pkl', '')
                    model_file = os.path.join(models_dir_to_use, filename)
                    
                    try:
                        with open(model_file, 'rb') as f:
                            model_data = pickle.load(f)
                        
                        models_data[model_name] = {
                            'model': model_data['model'],
                            'scaler': model_data['scaler'],
                            'score': model_data['score'],
                            'model_type': model_data.get('model_type', 'classification')
                        }
                        
                    except Exception as e:
                        print(f"⚠️ {model_name} 読み込みスキップ: {e}")
                        continue
            
            print(f"📥 CV用モデル読み込み: {len(models_data)}個")
            return models_data
            
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
            return {}
    
    def _calculate_feature_version(self):
        """特徴量のバージョンを計算（ハッシュベース）"""
        try:
            import hashlib
            # 特徴量ファイルの変更時刻ベースでバージョン計算
            features_file = "miniloto_models/features/features_cache.pkl"
            if os.path.exists(features_file):
                mtime = os.path.getmtime(features_file)
                version_string = str(mtime)
                return hashlib.md5(version_string.encode()).hexdigest()[:8]
            else:
                return "unknown"
        except:
            return "unknown"

    def check_feature_version_compatibility(self, model_name):
        """特徴量バージョン互換性チェック"""
        try:
            current_version = self._calculate_feature_version()
            
            # 進捗ファイルから保存済みバージョンを確認
            progress_file = os.path.join(self.cv_dir, f"{model_name}_cv_progress.pkl")
            
            if os.path.exists(progress_file):
                with open(progress_file, 'rb') as f:
                    progress_data = pickle.load(f)
                
                saved_version = progress_data.get('feature_version', 'unknown')
                
                if saved_version != current_version:
                    print(f"⚠️ {model_name} 特徴量バージョン不一致")
                    print(f"    保存済み: {saved_version}")
                    print(f"    現在: {current_version}")
                    
                    # 古い進捗をバックアップに移動
                    backup_file = progress_file.replace('.pkl', f'_v{saved_version}.pkl.bak')
                    os.rename(progress_file, backup_file)
                    print(f"    📦 古い進捗を {backup_file} にバックアップ")
                    
                    return False, 0, []  # 新規開始
                else:
                    print(f"✅ {model_name} 特徴量バージョン一致: {current_version}")
                    return True, progress_data.get('completed_splits', 0), progress_data.get('current_results', [])
            else:
                print(f"🆕 {model_name} 新規CV開始（バージョン: {current_version}）")
                return True, 0, []
                
        except Exception as e:
            print(f"⚠️ {model_name} バージョンチェックエラー: {e}")
            return True, 0, []  # エラー時は新規開始


# ======================================================================
# 2. スコア順CV実行管理
# ======================================================================

class ScoreBasedCVManager:
    def __init__(self):
        self.cv_system = IncrementalTimeSeriesCV()
        

    def get_models_sorted_by_score(self):
        """新指標ベース優先度順でソートされたモデルリストを取得（修正版）"""
        try:
            models_data = self.cv_system.load_available_models_for_cv()
            
            if not models_data:
                print("❌ CV対象モデルが見つかりません")
                return []
            
            # 新指標ベース優先度順にソート
            model_priorities = []
            
            for model_name, model_info in models_data.items():
                try:
                    # 新指標を取得（修正: max_match_score追加）
                    avg_match = model_info.get('avg_match_score', 0.0)
                    max_match = model_info.get('max_match_score', 0)  # ★修正: 追加
                    recall = model_info.get('recall_score', 0.0)
                    
                    # 優先度スコア計算: avg_match * 0.4 + max_match * 0.5 + recall * 0.1
                    priority_score = avg_match * 0.4 + max_match * 0.5 + recall * 0.1
                    
                    # 新指標がない場合は従来のaccuracyスコアを使用
                    if priority_score <= 0:
                        priority_score = model_info.get('score', 0.0) * 0.1  # 低めの重み
                    
                    model_priorities.append((model_name, model_info, priority_score))
                    
                except Exception as e:
                    print(f"⚠️ {model_name} 優先度計算エラー: {e}")
                    # エラー時は最低優先度
                    model_priorities.append((model_name, model_info, 0.0))
            
            # 優先度順にソート（高い順）
            sorted_models = sorted(model_priorities, key=lambda x: x[2], reverse=True)
            
            print("📊 CV実行順序（新指標ベース優先度順）:")
            for rank, (model_name, model_info, priority_score) in enumerate(sorted_models, 1):
                avg_match = model_info.get('avg_match_score', 0.0)
                max_match = model_info.get('max_match_score', 0)  # ★修正: 追加
                recall = model_info.get('recall_score', 0.0)
                print(f"  {rank:2d}位: {model_name:15s} 優先度: {priority_score:.4f} "
                      f"(平均一致: {avg_match:.2f}, 最大一致: {max_match}, Recall: {recall:.3f})")
            
            # 元の形式に戻す
            result = [(name, info) for name, info, _ in sorted_models]
            return result
            
        except Exception as e:
            print(f"❌ モデルソートエラー: {e}")
            return []

    
    def execute_cv_in_score_order(self, X, y, max_models=None, splits_per_batch=10):
        """スコア順でCV実行"""
        try:
            print("🔁 === スコア順時系列CV実行開始 ===")
            print(f"⚡ 高精度モデル優先・段階的保存モード")
            
            sorted_models = self.get_models_sorted_by_score()
            
            if not sorted_models:
                return {}
            
            if max_models:
                sorted_models = sorted_models[:max_models]
                print(f"🎯 実行対象: 上位{max_models}モデル")
            
            cv_results = {}
            total_models = len(sorted_models)
            
            for model_idx, (model_name, model_info) in enumerate(sorted_models, 1):
                print(f"\n🤖 [{model_idx}/{total_models}] {model_name} CV実行中...")
                
                try:
                    # 段階的CV実行
                    model_cv_result = self.execute_incremental_cv_for_model(
                        model_name, model_info, X, y, splits_per_batch
                    )
                    
                    if model_cv_result:
                        cv_results[model_name] = model_cv_result
                        print(f"✅ {model_name} CV完了")
                    else:
                        print(f"❌ {model_name} CV失敗")
                        
                except Exception as e:
                    print(f"❌ {model_name} CVエラー: {e}")
                    continue
            
            print(f"\n🎉 スコア順CV実行完了: {len(cv_results)}/{total_models}モデル")
            return cv_results
            
        except Exception as e:
            print(f"❌ スコア順CV実行エラー: {e}")
            return {}

    def _execute_single_split_cv(self, model_info, X, y, train_start, train_end, test_start, test_end, split_info):
        """単一分割のCV実行（単一クラス問題修正版）"""
        try:
            print(f"🔍 DEBUG: 分割前 y.shape = {y.shape}, y.ndim = {y.ndim}")
            print(f"🔍 DEBUG: train_range = {train_start}:{train_end}, test_range = {test_start}:{test_end}")
            
            # データ範囲チェック
            if train_end > len(X) or test_end > len(X) or train_start < 0 or test_start < 0:
                print(f"❌ 範囲エラー: データ長={len(X)}")
                return None
            
            if train_end <= train_start or test_end <= test_start:
                print(f"❌ 無効な範囲")
                return None
            
            # データ分割（強制2次元維持）
            X_train = X.iloc[train_start:train_end]
            X_test = X.iloc[test_start:test_end]
            
            # ===== 強制2次元維持（修正版）=====
            y_train = y[train_start:train_end]
            y_test = y[test_start:test_end]
            
            print(f"🔍 DEBUG: 分割後 y_train.shape = {y_train.shape}, y_test.shape = {y_test.shape}")
            
            # 強制的に2次元を保持
            if len(y_train.shape) == 1:
                print(f"⚠️ y_train 1次元検出: {y_train.shape} → 修正中")
                if len(y_train) == 0:
                    y_train = y_train.reshape(0, 31)
                else:
                    y_train = y_train.reshape(-1, 31) if y_train.size % 31 == 0 else y_train.reshape(len(y_train), 1)
                print(f"✅ y_train 修正完了: {y_train.shape}")
            
            if len(y_test.shape) == 1:
                print(f"⚠️ y_test 1次元検出: {y_test.shape} → 修正中")
                if len(y_test) == 0:
                    y_test = y_test.reshape(0, 31)
                else:
                    y_test = y_test.reshape(-1, 31) if y_test.size % 31 == 0 else y_test.reshape(len(y_test), 1)
                print(f"✅ y_test 修正完了: {y_test.shape}")
            
            # 31次元チェック
            if y_train.shape[1] != 31:
                print(f"❌ y_train 次元エラー: {y_train.shape[1]} != 31")
                return None
            
            if y_test.shape[1] != 31:
                print(f"❌ y_test 次元エラー: {y_test.shape[1]} != 31")
                return None
            
            if len(X_train) < 5 or len(X_test) == 0:
                print(f"❌ サンプル数不足: train={len(X_train)}, test={len(X_test)}")
                return None
            
            # ★修正: 単一クラス問題チェック強化版
            unique_classes_per_output = []
            valid_outputs = 0
            
            for i in range(y_train.shape[1]):
                unique_values = np.unique(y_train[:, i])
                unique_classes = len(unique_values)
                unique_classes_per_output.append(unique_classes)
                
                # 有効な出力（2クラス以上）をカウント
                if unique_classes >= 2:
                    valid_outputs += 1
            
            print(f"🔍 クラス多様性チェック: 有効出力={valid_outputs}/{len(unique_classes_per_output)}")
            
            # 条件1: 有効な出力が全体の30%未満の場合はスキップ
            valid_ratio = valid_outputs / len(unique_classes_per_output)
            if valid_ratio < 0.3:
                print(f"⚠️ 有効出力比率が低すぎます: {valid_ratio:.2%} < 30% → スキップ")
                return None
            
            # 条件2: 訓練データサイズが小さすぎる場合はより厳格に
            if len(X_train) < 30:
                min_required_outputs = max(10, len(unique_classes_per_output) * 0.5)
                if valid_outputs < min_required_outputs:
                    print(f"⚠️ 小データセット、有効出力数不足: {valid_outputs} < {min_required_outputs} → スキップ")
                    return None
            
            # 条件3: 全出力の分散をチェック
            output_variances = []
            for i in range(y_train.shape[1]):
                if len(np.unique(y_train[:, i])) >= 2:
                    variance = np.var(y_train[:, i])
                    output_variances.append(variance)
            
            if output_variances:
                mean_variance = np.mean(output_variances)
                if mean_variance < 0.01:  # 非常に低い分散
                    print(f"⚠️ 出力分散が低すぎます: {mean_variance:.4f} → スキップ")
                    return None
            
            print(f"✅ クラス多様性チェック通過: 有効出力={valid_outputs}, 比率={valid_ratio:.2%}")
            
            print(f"✅ 最終形状: X_train{X_train.shape}, y_train{y_train.shape}, X_test{X_test.shape}, y_test{y_test.shape}")
print(f"✅ クラス数チェック済み: 有効出力={valid_outputs}/{len(unique_classes_per_output)}")

            
            # 重み付きアンサンブル予測の実装
            try:
                # mainで保存された重みファイルを読み込み
                model_weights = self._load_main_model_weights()
                
                # 全モデルの予測を収集
                all_models_data = self.cv_system.load_available_models_for_cv()
                ensemble_predictions = []
                ensemble_weights = []
                
                for ensemble_model_name, ensemble_model_info in all_models_data.items():
                    try:
                        # 各モデルで予測
                        model = ensemble_model_info['model']
                        model_type = ensemble_model_info.get('model_type', 'classification')
                        
                        # スケーリング
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # モデル複製・学習
                        if model_type == "regression_multioutput":
                            from sklearn.base import clone
                            trained_models = []
                            predictions = np.zeros_like(y_test)
                            
                            for i in range(y_train.shape[1]):
                                try:
                                    clf = clone(model[i]) if hasattr(model, '__getitem__') else clone(model)
                                    clf.fit(X_train_scaled, y_train[:, i])
                                    pred = clf.predict(X_test_scaled)
                                    predictions[:, i] = pred
                                    trained_models.append(clf)
                                except Exception as e:
                                    print(f"        ⚠️ 出力{i}学習エラー: {e}")
                                    continue
                            
                        else:
                            # 分類モデル
                            try:
                                from sklearn.base import clone
                                from sklearn.multioutput import MultiOutputClassifier
                                
                                # ★修正: 学習前の最終データチェック
                                train_classes_check = []
                                for i in range(y_train.shape[1]):
                                    n_classes = len(np.unique(y_train[:, i]))
                                    train_classes_check.append(n_classes)
                                
                                insufficient_classes = sum(1 for n in train_classes_check if n < 2)
                                if insufficient_classes > len(train_classes_check) * 0.7:  # 70%以上が単一クラス
                                    print(f"        ⚠️ {ensemble_model_name} 学習スキップ: 単一クラス出力多数 ({insufficient_classes}/{len(train_classes_check)})")
                                    continue
                                
                                if hasattr(model, 'estimators_'):
                                    clf = clone(model)
                                    clf.fit(X_train_scaled, y_train)
                                else:
                                    multi_clf = MultiOutputClassifier(clone(model))
                                    clf = multi_clf
                                    clf.fit(X_train_scaled, y_train)
                                
                                predictions = clf.predict(X_test_scaled)
                                
                                if len(predictions.shape) == 1:
                                    predictions = predictions.reshape(-1, 31)
                            
                            except Exception as e:
                                error_msg = str(e).lower()
                                if any(keyword in error_msg for keyword in ['1 class', 'one class', 'single class', 'zero weights']):
                                    print(f"        ⚠️ {ensemble_model_name} 単一クラス問題でスキップ: {e}")
                                else:
                                    print(f"        ❌ {ensemble_model_name} 学習エラー: {e}")
                                continue
                        
                        # 重みを取得
                        weight = model_weights.get(ensemble_model_name, 1.0) if model_weights else 1.0
                        
                        ensemble_predictions.append(predictions)
                        ensemble_weights.append(weight)
                        
                    except Exception as model_error:
                        print(f"        ⚠️ {ensemble_model_name} アンサンブル予測エラー: {model_error}")
                        continue
                
                # 重み付きアンサンブル実行
                if ensemble_predictions and ensemble_weights:
                    total_weight = sum(ensemble_weights)
                    final_predictions = np.zeros_like(ensemble_predictions[0], dtype=np.float64)  # ★修正: dtype指定
                    
                    for pred, weight in zip(ensemble_predictions, ensemble_weights):
                        # ★修正: データ型を統一
                        pred_float = pred.astype(np.float64)
                        final_predictions += pred_float * weight
                    
                    final_predictions /= total_weight
                    
                    # 新指標で評価
                    y_true_sets = []
                    for i in range(len(y_test)):
                        true_numbers = [j+1 for j in range(31) if y_test[i][j] == 1]
                        if len(true_numbers) == 5:
                            y_true_sets.append(true_numbers)
                    
                    predicted_sets = []
                    for i in range(len(final_predictions)):
                        top5_indices = np.argsort(final_predictions[i])[-5:]
                        predicted_sets.append([idx+1 for idx in top5_indices])
                    
                    # 新指標評価関数を呼び出し
                    quality_scores = self._evaluate_model_set_quality(y_true_sets, predicted_sets)
                    
                    # 従来のaccuracy計算も保持
                    predictions_binary = (final_predictions > 0.5).astype(int)
                    accuracy = accuracy_score(y_test.flatten(), predictions_binary.flatten())
                    
                    # 結果記録（新指標込み）
                    result = {
                        'strategy': split_info['strategy'],
                        'window': split_info['window'],
                        'train_size': len(X_train),
                        'test_size': len(X_test),
                        'score': accuracy,
                        'avg_match_score': quality_scores['avg_match_score'],
                        'max_match_score': quality_scores['max_match_score'],
                        'recall_score': quality_scores['recall_score'],
                        'train_start': train_start,
                        'train_end': train_end,
                        'test_start': test_start,
                        'test_end': test_end
                    }
                    
                    print(f"        ✅ アンサンブル評価成功: accuracy={accuracy:.4f}, "
                          f"平均一致={quality_scores['avg_match_score']:.2f}, "
                          f"最大一致={quality_scores['max_match_score']}, "
                          f"recall={quality_scores['recall_score']:.3f}")
                    
                    return result
                else:
                    print(f"        ❌ アンサンブル予測失敗、単一モデルにフォールバック")
                    
            except Exception as ensemble_error:
                print(f"        ⚠️ アンサンブル処理エラー: {ensemble_error}")
                print(f"        🔄 単一モデル評価にフォールバック")
            
            # フォールバック: 単一モデル評価（従来処理）
            model = model_info['model']
            model_type = model_info.get('model_type', 'classification')
            
            # スケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # モデル複製（元モデル保護）
            if model_type == "regression_multioutput":
                from sklearn.base import clone
                trained_models = []
                predictions = np.zeros_like(y_test)
                
                for i in range(y_train.shape[1]):
                    try:
                        clf = clone(model[i]) if hasattr(model, '__getitem__') else clone(model)
                        clf.fit(X_train_scaled, y_train[:, i])
                        pred = clf.predict(X_test_scaled)
                        predictions[:, i] = pred
                        trained_models.append(clf)
                    except Exception as e:
                        print(f"        ⚠️ 出力{i}学習エラー: {e}")
                        continue
                
                predictions_binary = (predictions > 0.5).astype(int)
                score = accuracy_score(y_test.flatten(), predictions_binary.flatten())
            else:
                try:
                    from sklearn.base import clone
                    from sklearn.multioutput import MultiOutputClassifier
                    
                    if hasattr(model, 'estimators_'):
                        clf = clone(model)
                        clf.fit(X_train_scaled, y_train)
                    else:
                        multi_clf = MultiOutputClassifier(clone(model))
                        clf = multi_clf
                        clf.fit(X_train_scaled, y_train)
                    
                    predictions = clf.predict(X_test_scaled)
                    
                    if len(predictions.shape) == 1:
                        predictions = predictions.reshape(-1, 31)
                    
                    score = accuracy_score(y_test.flatten(), predictions.flatten())
                    
                except Exception as e:
                    print(f"        ❌ 単一モデル学習エラー: {e}")
                    return None
            
            # 結果記録（フォールバック時は新指標なし）
            result = {
                'strategy': split_info['strategy'],
                'window': split_info['window'],
                'train_size': len(X_train),
                'test_size': len(X_test),
                'score': score,
                'avg_match_score': 0.0,
                'max_match_score': 0,
                'recall_score': 0.0,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            }
            
            return result
            
        except Exception as e:
            print(f"        ❌ 分割CV実行エラー: {e}")
            print(f"        🔍 エラー詳細: train_range({train_start}:{train_end}), test_range({test_start}:{test_end})")
            return None

    def _load_main_model_weights(self):
        """mainで保存されたモデル重みを読み込み（修正版）"""
        try:
            # 重みファイルのパス候補（修正: 拡張）
            weight_file_paths = [
                "miniloto_models/model_weights.pkl",
                "/content/drive/MyDrive/miniloto_models/model_weights.pkl",
                "model_weights.pkl",
                "ultra_comprehensive_learning_data_v4.pkl",  # ★修正: 追加
                "/content/drive/MyDrive/miniloto_predictor_ultra/ultra_comprehensive_learning_data_v4.pkl"  # ★修正: 追加
            ]
            
            for weight_path in weight_file_paths:
                if os.path.exists(weight_path):
                    try:
                        with open(weight_path, 'rb') as f:
                            data = pickle.load(f)
                        
                        # ★修正: ファイル形式による重み抽出
                        if isinstance(data, dict):
                            if 'model_weights' in data:
                                model_weights = data['model_weights']
                                print(f"        📥 モデル重み読み込み成功(統合ファイル): {weight_path}")
                                return model_weights
                            elif all(isinstance(v, (int, float)) for v in data.values()):
                                # 直接重み辞書の場合
                                print(f"        📥 モデル重み読み込み成功(直接): {weight_path}")
                                return data
                        
                    except Exception as e:
                        print(f"        ⚠️ 重みファイル読み込みエラー({weight_path}): {e}")
                        continue
            
            print(f"        📝 モデル重みファイルが見つかりません、等重みを使用")
            return None
            
        except Exception as e:
            print(f"        ❌ モデル重み読み込みエラー: {e}")
            return None

    def _evaluate_model_set_quality(self, y_true_sets, predicted_sets):
        """
        モデル予測セットの質を3つの指標で評価（CV版）
        """
        try:
            if not y_true_sets or not predicted_sets:
                return {"avg_match_score": 0.0, "max_match_score": 0, "recall_score": 0.0}
            
            # データ数を合わせる
            min_len = min(len(y_true_sets), len(predicted_sets))
            y_true_sets = y_true_sets[:min_len]
            predicted_sets = predicted_sets[:min_len]
            
            total_matches = []
            individual_recalls = []
            
            # 1対1で比較
            for true_set, pred_set in zip(y_true_sets, predicted_sets):
                if isinstance(true_set, (list, tuple)) and isinstance(pred_set, (list, tuple)):
                    # 各セットの一致数を計算
                    match_count = len(set(pred_set) & set(true_set))
                    total_matches.append(match_count)
                    
                    # 個別recall計算（その回の当選5番号のうち予測で拾えた割合）
                    if len(true_set) > 0:
                        individual_recall = len(set(pred_set) & set(true_set)) / len(true_set)
                        individual_recalls.append(individual_recall)
            
            if not total_matches:
                return {"avg_match_score": 0.0, "max_match_score": 0, "recall_score": 0.0}
            
            # 3つの指標を計算
            avg_match = sum(total_matches) / len(total_matches)
            max_match = max(total_matches)
            
            # recall = 個別recallの平均
            recall = sum(individual_recalls) / len(individual_recalls) if individual_recalls else 0.0
            
            return {
                "avg_match_score": avg_match,
                "max_match_score": max_match,
                "recall_score": recall
            }
            
        except Exception as e:
            print(f"❌ _evaluate_model_set_quality エラー: {e}")
            return {"avg_match_score": 0.0, "max_match_score": 0, "recall_score": 0.0}


# パート1ここまで

# -*- coding: utf-8 -*-
# ミニロト時系列交差検証バックグラウンド実行システム - Part2A: CV戦略・分割処理（前半）

# ======================================================================
# 3. 時系列CV戦略実装（ScoreBasedCVManagerクラスの続き）
# ======================================================================

    def execute_incremental_cv_for_model(self, model_name, model_info, X, y, splits_per_batch=10):
        """単一モデルの段階的CV実行"""
        try:
            # 継続実行の確認
            start_split, current_results = self.cv_system.continue_cv_from_checkpoint(model_name)
            
            # CV戦略定義（6戦略）
            cv_strategies = [
                {"name": "固定窓", "windows": [20, 30, 40, 50, 60, 70], "cumulative": False, "step": 2},
                {"name": "拡張窓", "windows": [20, 30, 40, 50, 60, 70], "cumulative": True, "step": 2},
                {"name": "適応窓", "windows": [15, 25, 35, 45, 55, 65], "cumulative": False, "step": 3},
                {"name": "重複窓", "windows": [25, 35, 45, 55], "cumulative": False, "step": 1},
                {"name": "季節調整窓", "windows": [30, 50, 70], "cumulative": False, "step": 4},
                {"name": "ローリング窓", "windows": [20, 40, 60], "cumulative": False, "step": 5}
            ]
            
            # 全分割生成
            all_splits = []
            strategy_info = []
            
            for strategy in cv_strategies:
                for window in strategy["windows"]:
                    splits = self._generate_time_series_splits(
                        len(X), window, strategy["cumulative"], strategy["step"]
                    )
                    
                    for split in splits:
                        all_splits.append(split)
                        strategy_info.append({
                            'strategy': strategy["name"],
                            'window': window,
                            'split': split
                        })
            
            total_splits = len(all_splits)
            print(f"    📊 総分割数: {total_splits} (開始位置: {start_split})")
            
            if start_split >= total_splits:
                print(f"    ✅ {model_name} 既に完了済み")
                return {"completed": True, "cv_score": 0}
            
            # バッチ処理でCV実行
            for batch_start in range(start_split, total_splits, splits_per_batch):
                batch_end = min(batch_start + splits_per_batch, total_splits)
                
                print(f"    🔄 バッチ処理: {batch_start+1}-{batch_end}/{total_splits}")
                
                # バッチ内分割処理
                batch_results = []
                skipped_splits = 0
                for split_idx in range(batch_start, batch_end):
                    split_info = strategy_info[split_idx]
                    train_start, train_end, test_start, test_end = all_splits[split_idx]
                    

                    try:
                        # 単一分割のCV実行
                        split_result = self._execute_single_split_cv(
                            model_info, X, y, 
                            train_start, train_end, test_start, test_end,
                            split_info
                        )
                        
                        if split_result:
                            batch_results.append(split_result)
                        else:
                            skipped_splits += 1
                            print(f"      ⚠️ 分割{split_idx+1} スキップ (単一クラス問題)")
                            

                    except Exception as e:
                        skipped_splits += 1
                        error_msg = str(e).lower()
                        if 'not defined' in error_msg or 'name' in error_msg:
                            print(f"      ❌ 分割{split_idx+1}コードエラー: {e}")
                        elif any(keyword in error_msg for keyword in ['1 class', 'one class', 'single class']):
                            print(f"      ⚠️ 分割{split_idx+1}単一クラス問題: {e}")
                        else:
                            print(f"      ❌ 分割{split_idx+1}エラー: {e}")
                        continue

                
                # バッチ完了報告にスキップ数を追加
                print(f"    📊 バッチ{batch_start+1}-{batch_end}完了: 成功={len(batch_results)}, スキップ={skipped_splits}")
                
                # バッチ結果を累積
                current_results.extend(batch_results)
                
                # 進捗保存
                self.cv_system.save_cv_progress(
                    model_name, current_results, batch_end, total_splits
                )
                
                # 完了チェック
                if batch_end >= total_splits:
                    break
            
            # CV結果統計計算
            if current_results:
                cv_scores = [r['score'] for r in current_results if 'score' in r]
                cv_summary = {
                    'cv_score': np.mean(cv_scores) if cv_scores else 0,
                    'cv_std': np.std(cv_scores) if cv_scores else 0,
                    'cv_results': current_results,
                    'total_splits': len(current_results),
                    'completed': True
                }
                
                # 強化モデル保存
                enhanced_model_data = {
                    'model': model_info['model'],
                    'scaler': model_info['scaler'],
                    'cv_score': cv_summary['cv_score'],
                    'cv_std': cv_summary['cv_std'],
                    'cv_results': current_results
                }
                
                self.cv_system.update_model_with_cv_results(model_name, enhanced_model_data)
                
                return cv_summary
            else:
                print(f"    ❌ {model_name} CV結果なし")
                return None
                
        except Exception as e:
            print(f"❌ {model_name} 段階的CV実行エラー: {e}")
            return None
    
    def _generate_time_series_splits(self, data_length, window_size, cumulative, step):
        """時系列分割生成"""
        try:
            splits = []
            
            if data_length < window_size + 1:
                return splits
            
            if cumulative:
                # 拡張窓: 開始点固定、終了点を拡張
                for end_idx in range(window_size, data_length, step):
                    train_start = 0
                    train_end = end_idx
                    test_start = end_idx
                    test_end = min(end_idx + 1, data_length)
                    
                    if test_end <= data_length:
                        splits.append((train_start, train_end, test_start, test_end))
            else:
                # 固定窓: ウィンドウサイズ固定でスライド
                for start_idx in range(0, data_length - window_size, step):
                    train_start = start_idx
                    train_end = start_idx + window_size
                    test_start = train_end
                    test_end = min(train_end + 1, data_length)
                    
                    if test_end <= data_length:
                        splits.append((train_start, train_end, test_start, test_end))
            
            return splits
            
        except Exception as e:
            print(f"❌ 分割生成エラー: {e}")
            return []

# パート2Aここまで

# -*- coding: utf-8 -*-
# ミニロト時系列交差検証バックグラウンド実行システム - Part2B: CV戦略・分割処理（後半）

# ======================================================================
# 4. CV分割実行・戦略詳細実装
# ======================================================================

# ======================================================================
# 5. CV戦略詳細実装
# ======================================================================

class CVStrategyImplementation:
    def __init__(self):
        self.strategy_configs = self._get_strategy_configurations()
    
    def _get_strategy_configurations(self):
        """CV戦略設定"""
        return {
            "固定窓": {
                "description": "固定サイズウィンドウでスライド",
                "windows": [20, 30, 40, 50, 60, 70],
                "cumulative": False,
                "step": 2,
                "weight": 1.0
            },
            "拡張窓": {
                "description": "開始点固定、終了点拡張",
                "windows": [20, 30, 40, 50, 60, 70],
                "cumulative": True,
                "step": 2,
                "weight": 1.2  # より重要視
            },
            "適応窓": {
                "description": "適応的ウィンドウサイズ",
                "windows": [15, 25, 35, 45, 55, 65],
                "cumulative": False,
                "step": 3,
                "weight": 0.9
            },
            "重複窓": {
                "description": "高重複ウィンドウ",
                "windows": [25, 35, 45, 55],
                "cumulative": False,
                "step": 1,
                "weight": 0.8
            },
            "季節調整窓": {
                "description": "季節性考慮ウィンドウ",
                "windows": [30, 50, 70],
                "cumulative": False,
                "step": 4,
                "weight": 1.1
            },
            "ローリング窓": {
                "description": "ローリングウィンドウ",
                "windows": [20, 40, 60],
                "cumulative": False,
                "step": 5,
                "weight": 0.9
            }
        }
    
    def get_weighted_cv_score(self, cv_results):
        """戦略重み付きCV スコア計算"""
        try:
            if not cv_results:
                return 0
            
            weighted_scores = []
            
            for result in cv_results:
                strategy = result.get('strategy', '固定窓')
                score = result.get('score', 0)
                
                weight = self.strategy_configs.get(strategy, {}).get('weight', 1.0)
                weighted_score = score * weight
                
                weighted_scores.append(weighted_score)
            
            return np.mean(weighted_scores) if weighted_scores else 0
            
        except Exception as e:
            print(f"❌ 重み付きスコア計算エラー: {e}")
            return 0
    
    def analyze_strategy_performance(self, cv_results):
        """戦略別性能分析"""
        try:
            strategy_performance = {}
            
            for result in cv_results:
                strategy = result.get('strategy', 'unknown')
                score = result.get('score', 0)
                
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = []
                
                strategy_performance[strategy].append(score)
            
            # 統計計算
            strategy_stats = {}
            for strategy, scores in strategy_performance.items():
                if scores:
                    strategy_stats[strategy] = {
                        'count': len(scores),
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'min': min(scores),
                        'max': max(scores)
                    }
            
            return strategy_stats
            
        except Exception as e:
            print(f"❌ 戦略性能分析エラー: {e}")
            return {}
    
    def get_strategy_summary(self):
        """戦略サマリー取得"""
        try:
            summary = {}
            
            for strategy_name, config in self.strategy_configs.items():
                summary[strategy_name] = {
                    'description': config['description'],
                    'windows_count': len(config['windows']),
                    'weight': config['weight'],
                    'step_size': config['step'],
                    'cumulative': config['cumulative']
                }
            
            return summary
            
        except Exception as e:
            print(f"❌ 戦略サマリー取得エラー: {e}")
            return {}
    
    def calculate_total_splits_estimate(self, data_length):
        """総分割数推定"""
        try:
            total_splits = 0
            
            for strategy_name, config in self.strategy_configs.items():
                for window in config['windows']:
                    if data_length < window + 1:
                        continue
                    
                    if config['cumulative']:
                        # 拡張窓
                        splits_count = max(0, (data_length - window) // config['step'])
                    else:
                        # 固定窓
                        splits_count = max(0, (data_length - window) // config['step'])
                    
                    total_splits += splits_count
            
            return total_splits
            
        except Exception as e:
            print(f"❌ 分割数推定エラー: {e}")
            return 0

# ======================================================================
# 6. CV品質管理システム
# ======================================================================

class CVQualityManager:
    def __init__(self):
        self.strategy_impl = CVStrategyImplementation()
        
    def validate_cv_configuration(self, data_length, model_count):
        """CV設定検証"""
        try:
            print("🔍 === CV設定検証 ===")
            
            validation_results = {
                'data_sufficient': True,
                'estimated_splits': 0,
                'estimated_time_hours': 0,
                'warnings': [],
                'recommendations': []
            }
            
            # データ量チェック
            min_required_length = 100
            if data_length < min_required_length:
                validation_results['data_sufficient'] = False
                validation_results['warnings'].append(f"データ不足: {data_length}件 (最低{min_required_length}件必要)")
            
            # 分割数推定
            total_splits = self.strategy_impl.calculate_total_splits_estimate(data_length)
            validation_results['estimated_splits'] = total_splits
            
            # 処理時間推定（1分割あたり約0.1秒と仮定）
            estimated_seconds = total_splits * model_count * 0.1
            validation_results['estimated_time_hours'] = estimated_seconds / 3600
            
            print(f"📊 検証結果:")
            print(f"  データ長: {data_length}")
            print(f"  モデル数: {model_count}")
            print(f"  推定分割数: {total_splits:,}")
            print(f"  推定処理時間: {validation_results['estimated_time_hours']:.1f}時間")
            
            # 推奨事項
            if validation_results['estimated_time_hours'] > 24:
                validation_results['recommendations'].append("処理時間が長いため、段階的実行を推奨")
            
            if validation_results['estimated_time_hours'] > 72:
                validation_results['warnings'].append("処理時間が非常に長い（3日以上）")
                validation_results['recommendations'].append("クイック実行または対象モデル数を削減を推奨")
            
            # 結果表示
            if validation_results['warnings']:
                print(f"⚠️ 警告:")
                for warning in validation_results['warnings']:
                    print(f"  • {warning}")
            
            if validation_results['recommendations']:
                print(f"💡 推奨:")
                for rec in validation_results['recommendations']:
                    print(f"  • {rec}")
            
            return validation_results
            
        except Exception as e:
            print(f"❌ CV設定検証エラー: {e}")
            return None
    
    def monitor_cv_health(self, cv_results):
        """CV実行品質監視"""
        try:
            if not cv_results:
                return {}
            
            health_metrics = {
                'success_rate': 0,
                'average_score': 0,
                'score_stability': 0,
                'strategy_balance': {},
                'anomaly_count': 0
            }
            
            scores = [r['score'] for r in cv_results if 'score' in r and r['score'] is not None]
            
            if scores:
                health_metrics['success_rate'] = len(scores) / len(cv_results)
                health_metrics['average_score'] = np.mean(scores)
                health_metrics['score_stability'] = 1 / (1 + np.std(scores))
                
                # 異常値検出（3σ外れ値）
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                threshold = 3 * std_score
                
                anomalies = [s for s in scores if abs(s - mean_score) > threshold]
                health_metrics['anomaly_count'] = len(anomalies)
            
            # 戦略バランス
            strategy_counts = {}
            for result in cv_results:
                strategy = result.get('strategy', 'unknown')
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            total_results = len(cv_results)
            if total_results > 0:
                health_metrics['strategy_balance'] = {
                    strategy: count / total_results 
                    for strategy, count in strategy_counts.items()
                }
            
            return health_metrics
            
        except Exception as e:
            print(f"❌ CV健全性監視エラー: {e}")
            return {}

# パート2Bここまで

# -*- coding: utf-8 -*-
# ミニロト時系列交差検証バックグラウンド実行システム - Part3: CV評価・重み決定

# ======================================================================
# 5. CV評価・重み決定システム
# ======================================================================

class CVEvaluationSystem:
    def __init__(self):
        self.cv_strategy = CVStrategyImplementation()
        
    def determine_ultra_model_weights(self, all_cv_results):
        """超精密モデル重み決定"""
        try:
            print("⚖️ === 超精密モデル重み決定開始 ===")
            
            if not all_cv_results:
                print("❌ CV結果がありません")
                return {}
            
            model_weights = {}
            
            for model_name, cv_result in all_cv_results.items():
                try:
                    if not cv_result or 'cv_results' not in cv_result:
                        model_weights[model_name] = 0.1  # 最小重み
                        continue
                    
                    cv_results = cv_result['cv_results']
                    
                    # 基本統計
                    scores = [r['score'] for r in cv_results if 'score' in r]
                    if not scores:
                        model_weights[model_name] = 0.1
                        continue
                    
                    base_score = np.mean(scores)
                    score_std = np.std(scores)
                    
                    # 戦略重み付きスコア
                    weighted_score = self.cv_strategy.get_weighted_cv_score(cv_results)
                    
                    # 安定性指標
                    stability = 1 / (1 + score_std) if score_std > 0 else 1.0
                    
                    # 信頼性指標（サンプル数ベース）
                    reliability = min(len(scores) / 100, 1.0)  # 100分割で最大信頼性
                    
                    # 戦略別性能
                    strategy_stats = self.cv_strategy.analyze_strategy_performance(cv_results)
                    strategy_bonus = self._calculate_strategy_bonus(strategy_stats)
                    
                    # 最終重み計算
                    final_weight = (
                        weighted_score * 0.4 +
                        base_score * 0.3 +
                        stability * 0.15 +
                        reliability * 0.1 +
                        strategy_bonus * 0.05
                    )
                    
                    model_weights[model_name] = max(final_weight, 0.01)  # 最小重み保証
                    
                    print(f"  {model_name:15s}: 重み {final_weight:.4f} "
                          f"(基本:{base_score:.3f}, 重み付:{weighted_score:.3f}, "
                          f"安定:{stability:.3f}, 信頼:{reliability:.3f})")
                    
                except Exception as e:
                    print(f"  ⚠️ {model_name} 重み計算エラー: {e}")
                    model_weights[model_name] = 0.1
                    continue
            
            # 重み正規化
            total_weight = sum(model_weights.values())
            if total_weight > 0:
                normalized_weights = {
                    name: weight / total_weight 
                    for name, weight in model_weights.items()
                }
            else:
                # 全て等重み
                num_models = len(model_weights)
                normalized_weights = {
                    name: 1.0 / num_models 
                    for name in model_weights.keys()
                }
            
            print(f"\n⚖️ 正規化後重み:")
            sorted_weights = sorted(normalized_weights.items(), key=lambda x: x[1], reverse=True)
            for rank, (model_name, weight) in enumerate(sorted_weights, 1):
                print(f"  {rank:2d}位: {model_name:15s} 重み: {weight:.4f}")
            
            return normalized_weights
            
        except Exception as e:
            print(f"❌ モデル重み決定エラー: {e}")
            return {}
    
    def _calculate_strategy_bonus(self, strategy_stats):
        """戦略別ボーナス計算"""
        try:
            if not strategy_stats:
                return 0
            
            # 各戦略の相対性能を評価
            strategy_scores = []
            for strategy, stats in strategy_stats.items():
                strategy_scores.append(stats['mean'])
            
            if not strategy_scores:
                return 0
            
            max_score = max(strategy_scores)
            min_score = min(strategy_scores)
            
            if max_score == min_score:
                return 0.5  # 全戦略同等
            
            # 高性能戦略の割合
            high_performance_ratio = sum(
                1 for score in strategy_scores 
                if score > (max_score + min_score) / 2
            ) / len(strategy_scores)
            
            return high_performance_ratio
            
        except Exception as e:
            print(f"❌ 戦略ボーナス計算エラー: {e}")
            return 0
    
    def evaluate_cv_quality(self, all_cv_results):
        """CV品質評価"""
        try:
            print("📊 === CV品質評価 ===")
            
            quality_metrics = {
                'total_models': len(all_cv_results),
                'completed_models': 0,
                'avg_splits_per_model': 0,
                'avg_cv_score': 0,
                'score_variance': 0,
                'reliability_score': 0
            }
            
            completed_results = []
            total_splits = []
            all_scores = []
            
            for model_name, cv_result in all_cv_results.items():
                if cv_result and cv_result.get('completed', False):
                    quality_metrics['completed_models'] += 1
                    
                    cv_results = cv_result.get('cv_results', [])
                    total_splits.append(len(cv_results))
                    
                    scores = [r['score'] for r in cv_results if 'score' in r]
                    all_scores.extend(scores)
                    completed_results.append(cv_result)
            
            if completed_results:
                quality_metrics['avg_splits_per_model'] = np.mean(total_splits)
                quality_metrics['avg_cv_score'] = np.mean(all_scores) if all_scores else 0
                quality_metrics['score_variance'] = np.var(all_scores) if all_scores else 0
                
                # 信頼性スコア計算
                min_splits = min(total_splits) if total_splits else 0
                max_splits = max(total_splits) if total_splits else 0
                
                if max_splits > 0:
                    completion_rate = quality_metrics['completed_models'] / quality_metrics['total_models']
                    split_consistency = 1 - (max_splits - min_splits) / max_splits
                    score_stability = 1 / (1 + quality_metrics['score_variance'])
                    
                    quality_metrics['reliability_score'] = (
                        completion_rate * 0.4 +
                        split_consistency * 0.3 +
                        score_stability * 0.3
                    )
            
            print(f"  📊 CV品質メトリクス:")
            print(f"    総モデル数: {quality_metrics['total_models']}")
            print(f"    完了モデル数: {quality_metrics['completed_models']}")
            print(f"    完了率: {quality_metrics['completed_models']/quality_metrics['total_models']*100:.1f}%")
            print(f"    平均分割数: {quality_metrics['avg_splits_per_model']:.1f}")
            print(f"    平均CVスコア: {quality_metrics['avg_cv_score']:.4f}")
            print(f"    信頼性スコア: {quality_metrics['reliability_score']:.4f}")
            
            return quality_metrics
            
        except Exception as e:
            print(f"❌ CV品質評価エラー: {e}")
            return {}

# ======================================================================
# 6. CV結果統合・出力システム
# ======================================================================

# CVResultIntegrationクラスの_generate_cv_reportメソッドの修正

class CVResultIntegration:
    def __init__(self):
        self.evaluation_system = CVEvaluationSystem()
        
    def integrate_and_save_cv_results(self, all_cv_results):
        """CV結果統合・保存"""
        try:
            print("💾 === CV結果統合・保存開始 ===")
            
            # 統合結果作成
            integrated_results = {
                'timestamp': datetime.now(),
                'cv_results': all_cv_results,
                'model_weights': {},
                'quality_metrics': {},
                'summary_statistics': {}
            }
            
            # モデル重み決定
            model_weights = self.evaluation_system.determine_ultra_model_weights(all_cv_results)
            integrated_results['model_weights'] = model_weights
            
            # 品質評価
            quality_metrics = self.evaluation_system.evaluate_cv_quality(all_cv_results)
            integrated_results['quality_metrics'] = quality_metrics
            
            # 統計サマリー
            summary_stats = self._calculate_summary_statistics(all_cv_results)
            integrated_results['summary_statistics'] = summary_stats
            
            # 結果保存
            self._save_integrated_results(integrated_results)
            
            # レポート生成
            self._generate_cv_report(integrated_results)
            
            print("✅ CV結果統合・保存完了")
            return integrated_results
            
        except Exception as e:
            print(f"❌ CV結果統合エラー: {e}")
            return None
    
    def _calculate_summary_statistics(self, all_cv_results):
        """統計サマリー計算"""
        try:
            summary = {
                'total_models': len(all_cv_results),
                'successful_models': 0,
                'total_cv_splits': 0,
                'avg_model_score': 0,
                'best_model': None,
                'worst_model': None,
                'score_distribution': {}
            }
            
            model_scores = {}
            
            for model_name, cv_result in all_cv_results.items():
                if cv_result and cv_result.get('completed', False):
                    summary['successful_models'] += 1
                    
                    cv_score = cv_result.get('cv_score', 0)
                    model_scores[model_name] = cv_score
                    
                    cv_results = cv_result.get('cv_results', [])
                    summary['total_cv_splits'] += len(cv_results)
            
            if model_scores:
                summary['avg_model_score'] = np.mean(list(model_scores.values()))
                
                # 最高・最低モデル
                best_model = max(model_scores.items(), key=lambda x: x[1])
                worst_model = min(model_scores.items(), key=lambda x: x[1])
                
                summary['best_model'] = {'name': best_model[0], 'score': best_model[1]}
                summary['worst_model'] = {'name': worst_model[0], 'score': worst_model[1]}
                
                # スコア分布
                scores = list(model_scores.values())
                summary['score_distribution'] = {
                    'min': min(scores),
                    'max': max(scores),
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'median': np.median(scores)
                }
            
            return summary
            
        except Exception as e:
            print(f"❌ 統計サマリー計算エラー: {e}")
            return {}
    
    def _save_integrated_results(self, integrated_results):
        """統合結果保存"""
        try:
            # ローカル保存
            results_file = "miniloto_models/cv_results/integrated_cv_results.pkl"
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            
            with open(results_file, 'wb') as f:
                pickle.dump(integrated_results, f)
            
            print(f"💾 統合結果保存: {results_file}")
            
            # CSV形式でもエクスポート
            self._export_cv_results_csv(integrated_results)
            
            # Google Drive保存
            try:
                from google.colab import drive
                drive_file = "/content/drive/MyDrive/miniloto_models/cv_results/integrated_cv_results.pkl"
                os.makedirs(os.path.dirname(drive_file), exist_ok=True)
                
                with open(drive_file, 'wb') as f:
                    pickle.dump(integrated_results, f)
                
                print(f"☁️ Drive保存完了: {drive_file}")
            except:
                print("⚠️ Drive保存スキップ")
            
        except Exception as e:
            print(f"❌ 統合結果保存エラー: {e}")
    
    def _export_cv_results_csv(self, integrated_results):
        """CSV形式でエクスポート"""
        try:
            # モデル重みCSV
            weights_data = []
            model_weights = integrated_results.get('model_weights', {})
            
            for model_name, weight in model_weights.items():
                cv_result = integrated_results['cv_results'].get(model_name, {})
                cv_score = cv_result.get('cv_score', 0)
                cv_std = cv_result.get('cv_std', 0)
                
                weights_data.append({
                    'モデル名': model_name,
                    '重み': weight,
                    'CVスコア': cv_score,
                    'CV標準偏差': cv_std,
                    '完了': cv_result.get('completed', False)
                })
            
            weights_df = pd.DataFrame(weights_data)
            weights_csv = "miniloto_models/cv_results/model_weights.csv"
            weights_df.to_csv(weights_csv, index=False, encoding='utf-8-sig')
            
            print(f"📄 重みCSV保存: {weights_csv}")
            
        except Exception as e:
            print(f"❌ CSV エクスポートエラー: {e}")

# ======================================================================
# 7. CVレポート生成システム
# ======================================================================


    def _generate_cv_report(self, integrated_results):
        """CV実行レポート生成"""
        try:
            print("📋 === CV実行レポート生成 ===")
            
            report_lines = []
            report_lines.append("="*60)
            report_lines.append("🔁 ミニロト時系列交差検証実行レポート")
            report_lines.append("="*60)
            report_lines.append(f"実行日時: {integrated_results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # 実行概要
            quality_metrics = integrated_results.get('quality_metrics', {})
            summary_stats = integrated_results.get('summary_statistics', {})
            
            report_lines.append("📊 実行概要:")
            report_lines.append(f"  総モデル数: {quality_metrics.get('total_models', 0)}")
            report_lines.append(f"  完了モデル数: {quality_metrics.get('completed_models', 0)}")
            report_lines.append(f"  完了率: {quality_metrics.get('completed_models', 0)/max(quality_metrics.get('total_models', 1), 1)*100:.1f}%")
            report_lines.append(f"  総CV分割数: {summary_stats.get('total_cv_splits', 0)}")
            report_lines.append(f"  平均CVスコア: {quality_metrics.get('avg_cv_score', 0):.4f}")
            report_lines.append(f"  信頼性スコア: {quality_metrics.get('reliability_score', 0):.4f}")
            report_lines.append("")
            
            # 最高・最低モデル
            if summary_stats.get('best_model'):
                best = summary_stats['best_model']
                worst = summary_stats['worst_model']
                
                report_lines.append("🏆 性能ランキング:")
                report_lines.append(f"  最高性能: {best['name']} (スコア: {best['score']:.4f})")
                report_lines.append(f"  最低性能: {worst['name']} (スコア: {worst['score']:.4f})")
                report_lines.append("")
            
            # モデル重み
            model_weights = integrated_results.get('model_weights', {})
            if model_weights:
                sorted_weights = sorted(model_weights.items(), key=lambda x: x[1], reverse=True)
                
                report_lines.append("⚖️ モデル重み（上位10位）:")
                for rank, (model_name, weight) in enumerate(sorted_weights[:10], 1):
                    report_lines.append(f"  {rank:2d}位: {model_name:15s} 重み: {weight:.4f}")
                report_lines.append("")
            
            # 品質指標詳細
            score_dist = summary_stats.get('score_distribution', {})
            if score_dist:
                report_lines.append("📈 スコア分布:")
                report_lines.append(f"  最高スコア: {score_dist.get('max', 0):.4f}")
                report_lines.append(f"  最低スコア: {score_dist.get('min', 0):.4f}")
                report_lines.append(f"  平均スコア: {score_dist.get('mean', 0):.4f}")
                report_lines.append(f"  標準偏差: {score_dist.get('std', 0):.4f}")
                report_lines.append(f"  中央値: {score_dist.get('median', 0):.4f}")
                report_lines.append("")
            
            # 推奨事項
            report_lines.append("💡 推奨事項:")
            
            completion_rate = quality_metrics.get('completed_models', 0) / max(quality_metrics.get('total_models', 1), 1)
            if completion_rate < 0.8:
                report_lines.append("  • CV完了率が低いため、継続実行を推奨")
            
            avg_score = quality_metrics.get('avg_cv_score', 0)
            if avg_score < 0.7:
                report_lines.append("  • 平均スコアが低いため、モデル調整を推奨")
            elif avg_score > 0.9:
                report_lines.append("  • 高スコア達成、過学習チェックを推奨")
            
            reliability = quality_metrics.get('reliability_score', 0)
            if reliability > 0.8:
                report_lines.append("  • 高信頼性達成、メイン予測に適用可能")
            
            report_lines.append("")
            report_lines.append("="*60)
            
            # レポート保存
            report_text = "\n".join(report_lines)
            report_file = f"miniloto_models/cv_results/cv_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print(f"📄 CVレポート保存: {report_file}")
            
            # コンソール出力
            print("\n" + report_text)
            
        except Exception as e:
            print(f"❌ CVレポート生成エラー: {e}")


# パート3ここまで

# -*- coding: utf-8 -*-
# ミニロト時系列交差検証バックグラウンド実行システム - Part4: レポート・継続機能

# ======================================================================
# 8. CV継続・監視システム
# ======================================================================

class CVMonitoringSystem:
    def __init__(self):
        self.cv_manager = ScoreBasedCVManager()
        self.result_integration = CVResultIntegration()
        

    def monitor_cv_progress(self):
        """CV進捗監視"""
        try:
            print("👁️ === CV進捗監視開始 ===")
            
            cv_dir = "miniloto_models/cv_results"
            if not os.path.exists(cv_dir):
                print("❌ CV結果ディレクトリが見つかりません")
                return {}
            
            progress_files = [f for f in os.listdir(cv_dir) if f.endswith('_cv_progress.pkl')]
            
            if not progress_files:
                print("📝 CV進捗ファイルが見つかりません")
                return {}
            
            progress_summary = {}
            
            for progress_file in progress_files:
                try:
                    model_name = progress_file.replace('_cv_progress.pkl', '')
                    file_path = os.path.join(cv_dir, progress_file)
                    
                    with open(file_path, 'rb') as f:
                        progress_data = pickle.load(f)
                    
                    progress_summary[model_name] = {
                        'completed_splits': progress_data.get('completed_splits', 0),
                        'total_splits': progress_data.get('total_splits', 0),
                        'completion_rate': progress_data.get('completion_rate', 0),
                        'timestamp': progress_data.get('timestamp', datetime.now())
                    }
                    
                except Exception as e:
                    print(f"⚠️ {progress_file} 読み込みエラー: {e}")
                    continue
            
            # 進捗表示
            print(f"📊 CV進捗サマリー ({len(progress_summary)}モデル):")
            
            sorted_progress = sorted(
                progress_summary.items(),
                key=lambda x: x[1]['completion_rate'],
                reverse=True
            )
            
            for model_name, progress in sorted_progress:
                completed = progress['completed_splits']
                total = progress['total_splits']
                rate = progress['completion_rate']
                
                print(f"  {model_name:15s}: {completed:4d}/{total:4d} ({rate*100:5.1f}%)")
            
            return progress_summary
            
        except Exception as e:
            print(f"❌ CV進捗監視エラー: {e}")
            return {}

    
    def resume_incomplete_cv(self, max_models=None):
        """未完了CV再開"""
        try:
            print("🔄 === 未完了CV再開実行 ===")
            
            # 進捗確認
            progress_summary = self.monitor_cv_progress()
            
            if not progress_summary:
                print("❌ 再開対象が見つかりません")
                return False
            
            # 未完了モデル抽出
            incomplete_models = [
                model_name for model_name, progress in progress_summary.items()
                if progress['completion_rate'] < 1.0
            ]
            
            if not incomplete_models:
                print("✅ 全モデル完了済み")
                return True
            
            if max_models:
                incomplete_models = incomplete_models[:max_models]
            
            print(f"🎯 再開対象: {len(incomplete_models)}モデル")
            
            # 特徴量読み込み
            X, y = self.cv_manager.cv_system.load_features_for_cv()
            if X is None or y is None:
                print("❌ 特徴量読み込み失敗")
                return False
            
            # 再開実行
            resumed_results = {}
            
            for model_name in incomplete_models:
                print(f"\n🔄 {model_name} CV再開...")
                
                # モデル情報取得
                models_data = self.cv_manager.cv_system.load_available_models_for_cv()
                if model_name not in models_data:
                    print(f"❌ {model_name} モデルデータが見つかりません")
                    continue
                
                model_info = models_data[model_name]
                
                # CV実行
                try:
                    cv_result = self.cv_manager.execute_incremental_cv_for_model(
                        model_name, model_info, X, y, splits_per_batch=10
                    )
                    
                    if cv_result:
                        resumed_results[model_name] = cv_result
                        print(f"✅ {model_name} CV再開完了")
                    else:
                        print(f"❌ {model_name} CV再開失敗")
                        
                except Exception as e:
                    print(f"❌ {model_name} CV再開エラー: {e}")
                    continue
            
            if resumed_results:
                print(f"\n🎉 CV再開完了: {len(resumed_results)}モデル")
                
                # 結果統合
                self.result_integration.integrate_and_save_cv_results(resumed_results)
                return True
            else:
                print("❌ CV再開に失敗しました")
                return False
                
        except Exception as e:
            print(f"❌ CV再開エラー: {e}")
            return False
    
    def cleanup_cv_files(self, keep_recent=3):
        """CV一時ファイルクリーンアップ"""
        try:
            print("🧹 === CVファイルクリーンアップ ===")
            
            cv_dir = "miniloto_models/cv_results"
            if not os.path.exists(cv_dir):
                return
            
            # 進捗ファイルの整理
            progress_files = []
            for filename in os.listdir(cv_dir):
                if filename.endswith('_cv_progress.pkl'):
                    file_path = os.path.join(cv_dir, filename)
                    modification_time = os.path.getmtime(file_path)
                    progress_files.append((filename, modification_time))
            
            # 古いファイルを削除
            if len(progress_files) > keep_recent:
                progress_files.sort(key=lambda x: x[1], reverse=True)
                files_to_delete = progress_files[keep_recent:]
                
                deleted_count = 0
                for filename, _ in files_to_delete:
                    try:
                        file_path = os.path.join(cv_dir, filename)
                        os.remove(file_path)
                        deleted_count += 1
                        print(f"  🗑️ 削除: {filename}")
                    except Exception as e:
                        print(f"  ⚠️ 削除失敗: {filename} - {e}")
                
                print(f"✅ クリーンアップ完了: {deleted_count}ファイル削除")
            else:
                print("📝 削除対象ファイルなし")
                
        except Exception as e:
            print(f"❌ クリーンアップエラー: {e}")

# パート4ここまで

# -*- coding: utf-8 -*-
# ミニロト時系列交差検証バックグラウンド実行システム - Part5: メイン実行機能

# ======================================================================
# 9. メインCV実行システム
# ======================================================================

class MainCVExecutor:
    def __init__(self):
        self.cv_manager = ScoreBasedCVManager()
        self.monitoring = CVMonitoringSystem()
        self.result_integration = CVResultIntegration()
        
    def execute_full_background_cv(self, resume_incomplete=True, max_models=None):
        """フルバックグラウンドCV実行"""
        try:
            start_time = time.time()
            
            print("🚀 === フルバックグラウンドCV実行開始 ===")
            print("⏱️ 予想処理時間: 数時間～数日（モデル数・分割数による）")
            print("📊 スコア順実行・段階的保存・継続可能")
            
            # 未完了CV確認・再開
            if resume_incomplete:
                print("\n🔍 未完了CV確認中...")
                progress_summary = self.monitoring.monitor_cv_progress()
                
                if progress_summary:
                    incomplete_count = sum(
                        1 for progress in progress_summary.values()
                        if progress['completion_rate'] < 1.0
                    )
                    
                    if incomplete_count > 0:
                        print(f"🔄 未完了モデル発見: {incomplete_count}個")
                        print("未完了CV再開を実行します...")
                        
                        if self.monitoring.resume_incomplete_cv(max_models):
                            print("✅ 未完了CV再開完了")
                        else:
                            print("⚠️ 未完了CV再開に一部問題がありました")
            
            # 特徴量読み込み
            print("\n📥 CV用データ読み込み...")
            X, y = self.cv_manager.cv_system.load_features_for_cv()
            
            if X is None or y is None:
                print("❌ 特徴量データが見つかりません")
                print("🔧 先にメインシステム（run_ultra_maximum_precision_prediction）を実行してください")
                return None
            
            print(f"✅ 特徴量読み込み完了: {X.shape}")
            
            # スコア順CV実行
            print("\n🔁 スコア順CV実行開始...")
            cv_results = self.cv_manager.execute_cv_in_score_order(
                X, y, max_models=max_models, splits_per_batch=10
            )
            
            if not cv_results:
                print("❌ CV実行結果なし")
                return None
            
            # 結果統合・保存
            print("\n💾 CV結果統合・保存...")
            integrated_results = self.result_integration.integrate_and_save_cv_results(cv_results)
            
            # 実行時間計算
            total_elapsed = time.time() - start_time
            
            print(f"\n🎉 === フルバックグラウンドCV実行完了 ===")
            print(f"⏱️ 総実行時間: {total_elapsed/3600:.1f}時間")
            print(f"🤖 完了モデル数: {len(cv_results)}")
            
            if integrated_results:
                quality_metrics = integrated_results.get('quality_metrics', {})
                print(f"📊 平均CVスコア: {quality_metrics.get('avg_cv_score', 0):.4f}")
                print(f"🎯 信頼性スコア: {quality_metrics.get('reliability_score', 0):.4f}")
                
                # 次回実行推奨
                reliability = quality_metrics.get('reliability_score', 0)
                if reliability > 0.8:
                    print("✅ 高信頼性達成。メイン予測での使用を推奨します")
                elif reliability > 0.6:
                    print("📈 良好な結果。継続実行でさらなる改善可能")
                else:
                    print("🔄 信頼性向上のため継続実行を推奨します")
            
            return integrated_results
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ フルバックグラウンドCV実行エラー: {e}")
            print(f"⏱️ エラー時点での実行時間: {elapsed/3600:.1f}時間")
            print(f"詳細: {traceback.format_exc()}")
            return None
    
    def execute_quick_cv(self, target_models=5, splits_per_batch=5):
        """クイックCV実行（高速版）"""
        try:
            print("⚡ === クイックCV実行開始 ===")
            print(f"🎯 対象: 上位{target_models}モデル")
            
            start_time = time.time()
            
            # 特徴量読み込み
            X, y = self.cv_manager.cv_system.load_features_for_cv()
            if X is None or y is None:
                print("❌ 特徴量データなし")
                return None
            
            # 上位モデルのみCV実行
            cv_results = self.cv_manager.execute_cv_in_score_order(
                X, y, max_models=target_models, splits_per_batch=splits_per_batch
            )
            
            elapsed_time = time.time() - start_time
            
            if cv_results:
                print(f"✅ クイックCV完了: {len(cv_results)}モデル ({elapsed_time/60:.1f}分)")
                
                # 簡易結果保存
                quick_results = {
                    'timestamp': datetime.now(),
                    'cv_results': cv_results,
                    'execution_time': elapsed_time,
                    'type': 'quick_cv'
                }
                
                quick_file = f"miniloto_models/cv_results/quick_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                with open(quick_file, 'wb') as f:
                    pickle.dump(quick_results, f)
                
                print(f"💾 クイック結果保存: {quick_file}")
                return quick_results
            else:
                print("❌ クイックCV失敗")
                return None
                
        except Exception as e:
            print(f"❌ クイックCV実行エラー: {e}")
            return None
    
    def execute_single_model_cv(self, model_name, splits_per_batch=10):
        """単一モデルCV実行"""
        try:
            print(f"🎯 === 単一モデルCV実行: {model_name} ===")
            
            start_time = time.time()
            
            # 特徴量読み込み
            X, y = self.cv_manager.cv_system.load_features_for_cv()
            if X is None or y is None:
                print("❌ 特徴量データなし")
                return None
            
            # モデル情報取得
            models_data = self.cv_manager.cv_system.load_available_models_for_cv()
            if model_name not in models_data:
                print(f"❌ {model_name} が見つかりません")
                print(f"利用可能モデル: {list(models_data.keys())}")
                return None
            
            model_info = models_data[model_name]
            
            # CV実行
            cv_result = self.cv_manager.execute_incremental_cv_for_model(
                model_name, model_info, X, y, splits_per_batch
            )
            
            elapsed_time = time.time() - start_time
            
            if cv_result:
                print(f"✅ {model_name} CV完了 ({elapsed_time/60:.1f}分)")
                print(f"📊 CVスコア: {cv_result.get('cv_score', 0):.4f}")
                
                return {model_name: cv_result}
            else:
                print(f"❌ {model_name} CV失敗")
                return None
                
        except Exception as e:
            print(f"❌ 単一モデルCV実行エラー: {e}")
            return None

    def validate_feature_compatibility(self):
        """特徴量互換性検証"""
        try:
            print("🔍 特徴量互換性検証中...")
            
            cv_dir = "miniloto_models/cv_results"
            if not os.path.exists(cv_dir):
                print("📝 CV結果ディレクトリなし、問題なし")
                return True
            
            # 現在の特徴量バージョン
            features_file = "miniloto_models/features/features_cache.pkl"
            if os.path.exists(features_file):
                import hashlib
                mtime = os.path.getmtime(features_file)
                current_version = hashlib.md5(str(mtime).encode()).hexdigest()[:8]
            else:
                print("❌ 特徴量ファイルが見つかりません")
                return False
            
            # 進捗ファイルのバージョンチェック
            progress_files = [f for f in os.listdir(cv_dir) if f.endswith('_cv_progress.pkl')]
            
            compatible_count = 0
            incompatible_count = 0
            
            for progress_file in progress_files:
                try:
                    file_path = os.path.join(cv_dir, progress_file)
                    with open(file_path, 'rb') as f:
                        progress_data = pickle.load(f)
                    
                    saved_version = progress_data.get('feature_version', 'unknown')
                    
                    if saved_version == current_version:
                        compatible_count += 1
                    else:
                        incompatible_count += 1
                        model_name = progress_file.replace('_cv_progress.pkl', '')
                        print(f"  ⚠️ {model_name}: v{saved_version} → v{current_version} (バージョン不一致)")
                        
                except Exception as e:
                    print(f"  ❌ {progress_file} チェックエラー: {e}")
                    incompatible_count += 1
            
            print(f"✅ 互換性チェック完了:")
            print(f"  互換: {compatible_count}モデル")
            print(f"  非互換: {incompatible_count}モデル")
            
            if incompatible_count > 0:
                print(f"💡 非互換モデルは新しい特徴量で新規学習を開始します")
            
            return True
            
        except Exception as e:
            print(f"❌ 特徴量互換性検証エラー: {e}")
            return False

# ======================================================================
# 10. CV状態管理・ユーティリティ
# ======================================================================

class CVUtilities:
    @staticmethod
    def show_cv_status():
        """CV状態表示"""
        try:
            print("📊 === CV状態表示 ===")
            
            monitoring = CVMonitoringSystem()
            progress_summary = monitoring.monitor_cv_progress()
            
            if not progress_summary:
                print("📝 CV実行履歴なし")
                return
            
            # 完了状況
            completed_models = [
                name for name, progress in progress_summary.items()
                if progress['completion_rate'] >= 1.0
            ]
            
            incomplete_models = [
                name for name, progress in progress_summary.items()
                if progress['completion_rate'] < 1.0
            ]
            
            print(f"✅ 完了モデル: {len(completed_models)}個")
            print(f"🔄 未完了モデル: {len(incomplete_models)}個")
            
            if incomplete_models:
                print("\n🔄 未完了モデル詳細:")
                for model_name in incomplete_models[:5]:  # 上位5個表示
                    progress = progress_summary[model_name]
                    rate = progress['completion_rate']
                    print(f"  {model_name}: {rate*100:.1f}%完了")
            
            # 最新結果確認
            cv_results_dir = "miniloto_models/cv_results"
            if os.path.exists(cv_results_dir):
                result_files = [
                    f for f in os.listdir(cv_results_dir) 
                    if f.startswith('integrated_cv_results') and f.endswith('.pkl')
                ]
                
                if result_files:
                    latest_file = max(result_files, key=lambda f: os.path.getmtime(os.path.join(cv_results_dir, f)))
                    file_time = datetime.fromtimestamp(os.path.getmtime(os.path.join(cv_results_dir, latest_file)))
                    print(f"\n📄 最新統合結果: {latest_file}")
                    print(f"🕒 更新日時: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    
        except Exception as e:
            print(f"❌ CV状態表示エラー: {e}")
    
    @staticmethod
    def clean_cv_data(confirm=False):
        """CVデータクリーンアップ"""
        try:
            if not confirm:
                print("⚠️ CVデータクリーンアップ")
                print("全ての進捗データが削除されます。")
                print("実行する場合は clean_cv_data(confirm=True) を実行してください")
                return
            
            print("🧹 === CVデータクリーンアップ実行 ===")
            
            cv_dir = "miniloto_models/cv_results"
            if not os.path.exists(cv_dir):
                print("📝 CVディレクトリが存在しません")
                return
            
            # 進捗ファイル削除
            progress_files = [f for f in os.listdir(cv_dir) if f.endswith('_cv_progress.pkl')]
            deleted_count = 0
            
            for filename in progress_files:
                try:
                    file_path = os.path.join(cv_dir, filename)
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"  🗑️ 削除: {filename}")
                except Exception as e:
                    print(f"  ⚠️ 削除失敗: {filename} - {e}")
            
            print(f"✅ クリーンアップ完了: {deleted_count}ファイル削除")
            
        except Exception as e:
            print(f"❌ クリーンアップエラー: {e}")
    
    @staticmethod
    def export_cv_summary():
        """CV結果サマリーエクスポート"""
        try:
            print("📊 === CV結果サマリーエクスポート ===")
            
            # 最新統合結果読み込み
            cv_results_dir = "miniloto_models/cv_results"
            integrated_file = os.path.join(cv_results_dir, "integrated_cv_results.pkl")
            
            if not os.path.exists(integrated_file):
                print("❌ 統合CV結果が見つかりません")
                return None
            
            with open(integrated_file, 'rb') as f:
                integrated_results = pickle.load(f)
            
            # サマリー作成
            summary = {
                "実行日時": integrated_results['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                "品質メトリクス": integrated_results.get('quality_metrics', {}),
                "統計サマリー": integrated_results.get('summary_statistics', {}),
                "モデル重み上位5位": {}
            }
            
            # 上位5モデル重み
            model_weights = integrated_results.get('model_weights', {})
            if model_weights:
                sorted_weights = sorted(model_weights.items(), key=lambda x: x[1], reverse=True)
                for rank, (model_name, weight) in enumerate(sorted_weights[:5], 1):
                    summary["モデル重み上位5位"][f"{rank}位"] = f"{model_name} ({weight:.4f})"
            
            # JSON形式で保存
            import json
            summary_file = f"miniloto_models/cv_results/cv_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"📄 サマリーエクスポート完了: {summary_file}")
            return summary_file
            
        except Exception as e:
            print(f"❌ サマリーエクスポートエラー: {e}")
            return None

# パート5ここまで

# -*- coding: utf-8 -*-
# ミニロト時系列交差検証バックグラウンド実行システム - Part6: エントリーポイント

# ======================================================================
# 11. メイン実行関数群
# ======================================================================

def run_background_cv_full():
    """フルバックグラウンドCV実行"""
    try:
        print("🌟 === ミニロト時系列交差検証バックグラウンド実行 ===")
        print("🚀 フルモード: 全モデル・スコア順・段階的保存")
        print("="*60)
        
        # メイン実行
        executor = MainCVExecutor()
        result = executor.execute_full_background_cv(
            resume_incomplete=True,
            max_models=None  # 全モデル実行
        )
        
        if result:
            print("\n🎉 フルバックグラウンドCV実行成功！")
            
            # 結果サマリー
            quality_metrics = result.get('quality_metrics', {})
            model_weights = result.get('model_weights', {})
            
            print(f"📊 実行結果:")
            print(f"  完了モデル数: {quality_metrics.get('completed_models', 0)}")
            print(f"  平均CVスコア: {quality_metrics.get('avg_cv_score', 0):.4f}")
            print(f"  信頼性スコア: {quality_metrics.get('reliability_score', 0):.4f}")
            
            if model_weights:
                sorted_weights = sorted(model_weights.items(), key=lambda x: x[1], reverse=True)
                print(f"\n⚖️ 上位3モデル重み:")
                for rank, (model_name, weight) in enumerate(sorted_weights[:3], 1):
                    print(f"    {rank}位: {model_name} ({weight:.4f})")
            
            print(f"\n💡 次のステップ:")
            print(f"  🔄 メインシステム再実行で最高精度予測")
            print(f"  📊 run_ultra_maximum_precision_prediction()")
            
        else:
            print("\n😞 フルバックグラウンドCV実行に失敗しました")
            print("🔍 システム診断を実行することを推奨します")
        
    except Exception as e:
        print(f"\n💥 フルバックグラウンドCV実行エラー: {e}")
        print(f"詳細: {traceback.format_exc()}")

def run_background_cv_quick():
    """クイックバックグラウンドCV実行"""
    try:
        print("⚡ === ミニロト時系列交差検証 クイック実行 ===")
        print("🎯 高速モード: 上位5モデル・最小分割")
        print("="*50)
        
        executor = MainCVExecutor()
        result = executor.execute_quick_cv(
            target_models=5,
            splits_per_batch=5
        )
        
        if result:
            print("\n✅ クイックバックグラウンドCV完了！")
            print("🔄 フル実行での精度向上も可能です")
        else:
            print("\n❌ クイックバックグラウンドCV失敗")
        
    except Exception as e:
        print(f"\n💥 クイックCV実行エラー: {e}")

def run_background_cv_resume():
    """未完了CV再開実行"""
    try:
        print("🔄 === 未完了CV再開実行 ===")
        
        monitoring = CVMonitoringSystem()
        success = monitoring.resume_incomplete_cv(max_models=10)
        
        if success:
            print("✅ 未完了CV再開完了")
        else:
            print("❌ 未完了CV再開失敗")
        
    except Exception as e:
        print(f"💥 CV再開エラー: {e}")

def run_background_cv_single(model_name):
    """単一モデルCV実行"""
    try:
        print(f"🎯 === 単一モデルCV実行: {model_name} ===")
        
        executor = MainCVExecutor()
        result = executor.execute_single_model_cv(model_name)
        
        if result:
            print(f"✅ {model_name} CV完了")
        else:
            print(f"❌ {model_name} CV失敗")
        
    except Exception as e:
        print(f"💥 単一モデルCV実行エラー: {e}")

# ======================================================================
# 12. ユーティリティ実行関数
# ======================================================================

def show_cv_progress():
    """CV進捗表示"""
    CVUtilities.show_cv_status()

def clean_cv_progress(confirm=False):
    """CV進捗クリーンアップ"""
    CVUtilities.clean_cv_data(confirm=confirm)

def export_cv_results():
    """CV結果エクスポート"""
    return CVUtilities.export_cv_summary()

def cleanup_cv_files():
    """CV一時ファイルクリーンアップ"""
    monitoring = CVMonitoringSystem()
    monitoring.cleanup_cv_files()

# ======================================================================
# 13. メイン実行メニュー
# ======================================================================

def show_cv_menu():
    """CVメニュー表示"""
    print("\n" + "="*60)
    print("🔁 ミニロト時系列交差検証バックグラウンド実行システム")
    print("🚀 改修版: スコア順・段階的保存・継続実行対応")
    print("="*60)
    print("\n📋 利用可能な機能:")
    print("  1. フル実行: run_background_cv_full()")
    print("  2. クイック実行: run_background_cv_quick()")
    print("  3. 未完了再開: run_background_cv_resume()")
    print("  4. 単一モデル: run_background_cv_single('model_name')")
    print("  5. 進捗確認: show_cv_progress()")
    print("  6. 結果エクスポート: export_cv_results()")
    print("  7. ファイル整理: cleanup_cv_files()")
    print("\n🔧 改修の特徴:")
    print("  • スコア順CV実行（高精度モデル優先）")
    print("  • 10件ごと段階的保存（中断安全）")
    print("  • 継続実行可能（いつでも再開）")
    print("  • Google Drive自動同期")
    print("\n⚡ 推奨実行順序:")
    print("  1. メインシステム実行（モデル学習）")
    print("  2. バックグラウンドCV実行（精度向上）")
    print("  3. メインシステム再実行（最高精度予測）")

def run_cv_interactive():
    """インタラクティブCV実行（改善版）"""
    try:
        show_cv_menu()
        
        # 進捗状況を自動判定
        monitoring = CVMonitoringSystem()
        progress_summary = monitoring.monitor_cv_progress()
        
        if not progress_summary:
            # 進捗ファイルなし = 初回実行
            print(f"\n🆕 初回実行: フルバックグラウンドCVを開始します")
            run_background_cv_full()
            
        else:
            # 特徴量互換性をチェック
            executor = MainCVExecutor()
            compatibility_ok = executor.validate_feature_compatibility()
            
            if not compatibility_ok:
                print(f"\n🔄 特徴量変更検出: フルCV再実行を開始します")
                run_background_cv_full()
                
            else:
                # 互換性OK = 現在のバージョンでの進捗状況を判定
                incomplete_count = sum(
                    1 for progress in progress_summary.values()
                    if progress['completion_rate'] < 1.0
                )
                
                if incomplete_count > 0:
                    print(f"\n🔄 未完了CV発見: 継続実行を開始します")
                    run_background_cv_resume()
                else:
                    print(f"\n✅ 全CV完了済み: メインシステムでの予測実行を推奨")
                    print(f"📊 追加実行が必要な場合は手動で関数を実行してください")
        
    except Exception as e:
        print(f"💥 インタラクティブCV実行エラー: {e}")

# ======================================================================
# 14. エントリーポイント
# ======================================================================

if __name__ == "__main__":
    print("🔁 ミニロト時系列交差検証バックグラウンド実行システム起動")
    print("🎯 改修版: 段階的保存・継続実行・スコア順対応")
    print("="*60)
    
    try:
        # システム確認
        print("🔍 システム確認中...")
        
        # 必要ディレクトリ作成
        os.makedirs("miniloto_models/cv_results", exist_ok=True)
        os.makedirs("miniloto_models/models", exist_ok=True)
        
        # CV状態確認
        print("📊 現在のCV状態:")
        CVUtilities.show_cv_status()
        
        print(f"\n🕐 システム開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # メイン実行
        run_cv_interactive()
        
    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによって中断されました")
        print("🔄 継続実行機能により、いつでも再開可能です")
    except Exception as e:
        print(f"\n💥 システム起動エラー: {e}")
        print("🔧 メインシステムが先に実行されているか確認してください")
    finally:
        print(f"\n🕒 システム終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🙏 時系列交差検証バックグラウンド実行システム終了")

# ======================================================================
# パート6ここまで
# 
# 【バックグラウンドCV改修完了】
# - スコア順CV実行（高精度モデル優先）
# - 段階的保存システム（10件ごと保存）
# - 継続実行機能（中断・再開対応）
# - CV結果統合・重み決定
# - Google Drive自動同期
# - 包括的監視・レポート機能
# 
# 【使用方法】
# 1. メインシステム実行後にバックグラウンドで実行
# 2. run_background_cv_full() でフル実行
# 3. run_background_cv_quick() で高速実行
# 4. show_cv_progress() で進捗確認
# 5. メインシステム再実行で最高精度予測
# ======================================================================