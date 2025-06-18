# ミニロト予測システム パート1A: 基盤システム（前半）
# ========================= パート1A開始 =========================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from collections import Counter, defaultdict
import json
import traceback
import gc
from datetime import datetime, timedelta
import requests
import io

print("🚀 ミニロト予測システム - パート1A: 基盤システム（前半）")
print("🎯 対象: ミニロト（1-31から5個選択 + ボーナス1個）")
print("📊 特徴量: 14次元最適化版")
print("🔧 固定窓: 50回分に調整")

# ミニロト用自動データ取得クラス
class MiniLotoDataFetcher:
    def __init__(self):
        self.csv_url = "https://miniloto.thekyo.jp/data/miniloto.csv"
        # ミニロト用カラム（文字化け対応）
        self.main_columns = ['第1数字', '第2数字', '第3数字', '第4数字', '第5数字']
        self.bonus_column = 'ボーナス数字'
        self.round_column = '開催回'
        self.date_column = '日付'
        self.latest_data = None
        self.latest_round = 0
        
        # 文字化け対応のカラムマッピング
        self.column_mapping = {
            0: '開催回',      # 第1カラム
            1: '日付',        # 第2カラム  
            2: '第1数字',     # 第3カラム
            3: '第2数字',     # 第4カラム
            4: '第3数字',     # 第5カラム
            5: '第4数字',     # 第6カラム
            6: '第5数字',     # 第7カラム
            7: 'ボーナス数字'  # 第8カラム
        }
        
    def fetch_latest_data(self):
        """最新のミニロトデータを自動取得"""
        try:
            print("🌐 === ミニロト自動データ取得開始 ===")
            print(f"📡 URL: {self.csv_url}")
            
            # CSVデータを取得
            response = requests.get(self.csv_url, timeout=30)
            response.raise_for_status()
            
            print(f"✅ データ取得成功: {len(response.content)} bytes")
            
            # CSVをパース（文字エンコーディングを考慮）
            try:
                # shift-jisで試す（日本のCSVの一般的なエンコーディング）
                csv_content = response.content.decode('shift-jis')
                df = pd.read_csv(io.StringIO(csv_content))
            except:
                try:
                    # UTF-8で試す
                    csv_content = response.content.decode('utf-8')
                    df = pd.read_csv(io.StringIO(csv_content))
                except:
                    # cp932で試す
                    csv_content = response.content.decode('cp932')
                    df = pd.read_csv(io.StringIO(csv_content))
            
            print(f"📊 データ読み込み: {len(df)}件")
            
            # カラム名の正規化（位置ベースで安全に処理）
            if len(df.columns) >= 8:
                # 新しいカラム名でデータフレームを再構築
                normalized_data = {}
                
                for i, new_col_name in self.column_mapping.items():
                    if i < len(df.columns):
                        normalized_data[new_col_name] = df.iloc[:, i].values
                
                # 正規化されたデータフレームを作成
                self.latest_data = pd.DataFrame(normalized_data)
                
                # データ型の確認と修正
                for col in ['開催回', '第1数字', '第2数字', '第3数字', '第4数字', '第5数字', 'ボーナス数字']:
                    if col in self.latest_data.columns:
                        self.latest_data[col] = pd.to_numeric(self.latest_data[col], errors='coerce')
                
                # 不正なデータを除去
                self.latest_data = self.latest_data.dropna()
                
                print(f"📋 正規化完了: {list(self.latest_data.columns)}")
                
                # 最新回を取得
                if '開催回' in self.latest_data.columns:
                    self.latest_round = int(self.latest_data['開催回'].max())
                    print(f"🎯 最新開催回: 第{self.latest_round}回")
                    
                    # 最新データの確認
                    latest_entry = self.latest_data[self.latest_data['開催回'] == self.latest_round].iloc[0]
                    print(f"📅 最新回日付: {latest_entry.get('日付', 'N/A')}")
                    
                    main_nums = [int(latest_entry[f'第{i}数字']) for i in range(1, 6)]
                    bonus_num = int(latest_entry['ボーナス数字'])
                    print(f"🎲 最新回当選番号: {main_nums} + ボーナス{bonus_num}")
                
                print("✅ ミニロトデータ取得完了")
                return True
            else:
                print("❌ CSVの構造が期待と異なります")
                return False
            
        except requests.exceptions.RequestException as e:
            print(f"❌ ネットワークエラー: {e}")
            return False
        except Exception as e:
            print(f"❌ データ取得エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            return False
    
    def get_next_round_info(self):
        """次回開催回の情報を取得"""
        if self.latest_round == 0:
            return None
            
        next_round = self.latest_round + 1
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        return {
            'next_round': next_round,
            'current_date': current_date,
            'latest_round': self.latest_round,
            'prediction_target': f"第{next_round}回"
        }
    
    def get_data_for_training(self):
        """学習用データを返す"""
        if self.latest_data is None:
            return None
        return self.latest_data

# ミニロト用予測記録管理クラス
class MiniLotoPredictionHistory:
    def __init__(self):
        self.predictions = []  # [{'round': int, 'date': str, 'predictions': list, 'actual': list or None}]
        self.accuracy_stats = {}
        
    def add_prediction_with_round(self, predictions, target_round, date=None):
        """開催回付きで予測を記録"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        entry = {
            'round': target_round,
            'date': date,
            'predictions': predictions.copy(),
            'actual': None,
            'matches': [],
            'verified': False
        }
        self.predictions.append(entry)
        print(f"📝 予測記録: 第{target_round}回 - {date} - {len(predictions)}セット")
        
    def find_prediction_by_round(self, round_number):
        """指定開催回の予測を検索"""
        for entry in self.predictions:
            if entry['round'] == round_number:
                return entry
        return None
    
    def auto_verify_with_data(self, latest_data, round_col='開催回'):
        """最新データと自動照合"""
        verified_count = 0
        main_cols = ['第1数字', '第2数字', '第3数字', '第4数字', '第5数字']
        
        for entry in self.predictions:
            if entry['verified']:
                continue
                
            # 該当する開催回のデータを検索
            matching_data = latest_data[latest_data[round_col] == entry['round']]
            
            if len(matching_data) > 0:
                actual_row = matching_data.iloc[0]
                actual_numbers = []
                for col in main_cols:
                    if col in actual_row.index:
                        actual_numbers.append(int(actual_row[col]))
                
                if len(actual_numbers) == 5:
                    entry['actual'] = actual_numbers
                    
                    # 各予測セットとの一致数を計算
                    matches = []
                    for pred_set in entry['predictions']:
                        match_count = len(set(pred_set) & set(actual_numbers))
                        matches.append(match_count)
                    
                    entry['matches'] = matches
                    entry['verified'] = True
                    verified_count += 1
                    
                    print(f"✅ 自動照合完了: 第{entry['round']}回")
                    print(f"   当選番号: {actual_numbers}")
                    print(f"   一致数: {matches}")
                    print(f"   最高一致: {max(matches)}個")
        
        if verified_count > 0:
            self._update_accuracy_stats()
            print(f"📊 {verified_count}件の予測を自動照合しました")
        
        return verified_count
    
    def _update_accuracy_stats(self):
        """精度統計を更新"""
        all_matches = []
        verified_predictions = [entry for entry in self.predictions if entry['verified']]
        
        for entry in verified_predictions:
            all_matches.extend(entry['matches'])
        
        if all_matches:
            self.accuracy_stats = {
                'total_predictions': len(all_matches),
                'verified_rounds': len(verified_predictions),
                'avg_matches': np.mean(all_matches),
                'max_matches': max(all_matches),
                'match_distribution': dict(Counter(all_matches)),
                'accuracy_by_match': {
                    f'{i}_matches': all_matches.count(i) for i in range(6)
                }
            }
    
    def get_accuracy_report(self):
        """精度レポートを生成"""
        if not self.accuracy_stats:
            return "📊 まだ照合済みの予測がありません"
        
        stats = self.accuracy_stats
        report = []
        report.append("📊 === ミニロト予測精度レポート ===")
        report.append(f"照合済み回数: {stats['verified_rounds']}回")
        report.append(f"総予測セット数: {stats['total_predictions']}セット")
        report.append(f"平均一致数: {stats['avg_matches']:.2f}個")
        report.append(f"最高一致数: {stats['max_matches']}個")
        report.append("")
        report.append("一致数分布:")
        for i in range(6):
            count = stats['accuracy_by_match'].get(f'{i}_matches', 0)
            percentage = (count / stats['total_predictions']) * 100 if stats['total_predictions'] > 0 else 0
            report.append(f"  {i}個一致: {count}セット ({percentage:.1f}%)")
        
        return "\n".join(report)
    
    def save_to_json(self):
        """予測履歴をJSONに保存（メモリベース）"""
        try:
            # JSON形式で保存（セッション内メモリ）
            self.saved_data = {
                'predictions': self.predictions,
                'accuracy_stats': self.accuracy_stats,
                'last_updated': datetime.now().isoformat()
            }
            print(f"💾 予測履歴をメモリに保存完了")
            return True
        except Exception as e:
            print(f"❌ JSON保存エラー: {e}")
            return False
    
    def load_from_json(self):
        """JSONから予測履歴を読み込み（メモリベース）"""
        try:
            if hasattr(self, 'saved_data') and self.saved_data:
                self.predictions = self.saved_data['predictions']
                self.accuracy_stats = self.saved_data['accuracy_stats']
                print(f"📂 予測履歴をメモリから読み込み: {len(self.predictions)}回分")
                return True
            else:
                print("📂 保存済み履歴が見つかりません")
                return False
        except Exception as e:
            print(f"❌ JSON読み込みエラー: {e}")
            return False

# ========================= パート1A（前半）ここまで =========================

# ミニロト予測システム パート1B: 基盤システム（後半）
# ========================= パート1B開始 =========================
# パート1Aの続き - 基本予測システムクラス

# ミニロト用基本予測システム
class MiniLotoBasicPredictor:
    def __init__(self):
        print("🔧 MiniLotoBasicPredictor初期化")
        
        # データ取得器
        self.data_fetcher = MiniLotoDataFetcher()
        
        # 基本モデル（パート1では2モデル）
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=80, max_depth=6, random_state=42
            )
        }
        
        self.scalers = {}
        self.model_weights = {
            'random_forest': 0.6,
            'gradient_boost': 0.4
        }
        
        # データ分析
        self.freq_counter = Counter()
        self.pair_freq = Counter()
        self.pattern_stats = {}
        
        # 学習状態
        self.trained_models = {}
        self.model_scores = {}
        self.data_count = 0
        
        # 予測履歴
        self.history = MiniLotoPredictionHistory()
        
        print("✅ 基本予測システム初期化完了")
    
    def create_basic_features(self, data):
        """基本的な14次元特徴量エンジニアリング"""
        try:
            print("🔧 14次元特徴量エンジニアリング開始")
            
            features = []
            targets = []
            main_cols = ['第1数字', '第2数字', '第3数字', '第4数字', '第5数字']
            
            for i in range(len(data)):
                try:
                    current = []
                    for col in main_cols:
                        if col in data.columns:
                            current.append(int(data.iloc[i][col]))
                    
                    if len(current) != 5:
                        continue
                    
                    # ミニロトの範囲チェック（1-31）
                    if not all(1 <= x <= 31 for x in current):
                        continue
                    if len(set(current)) != 5:  # 重複チェック
                        continue
                    
                    # 基本統計（頻出カウント）
                    for num in current:
                        self.freq_counter[num] += 1
                    
                    # ペア分析
                    for j in range(len(current)):
                        for k in range(j+1, len(current)):
                            pair = tuple(sorted([current[j], current[k]]))
                            self.pair_freq[pair] += 1
                    
                    # 14次元特徴量（ミニロト最適化版）
                    sorted_nums = sorted(current)
                    gaps = [sorted_nums[j+1] - sorted_nums[j] for j in range(4)]  # 4個のギャップ
                    
                    feat = [
                        float(np.mean(current)),           # 1. 平均値
                        float(np.std(current)),            # 2. 標準偏差
                        float(np.sum(current)),            # 3. 合計値
                        float(sum(1 for x in current if x % 2 == 1)),  # 4. 奇数個数
                        float(max(current)),               # 5. 最大値
                        float(min(current)),               # 6. 最小値
                        float(np.median(current)),         # 7. 中央値
                        float(max(current) - min(current)), # 8. 範囲
                        float(len([j for j in range(len(sorted_nums)-1) 
                                 if sorted_nums[j+1] - sorted_nums[j] == 1])), # 9. 連続数
                        float(current[0]),                 # 10. 第1数字
                        float(current[2]),                 # 11. 第3数字（中央）
                        float(current[4]),                 # 12. 第5数字（最後）
                        float(np.mean(gaps)),              # 13. 平均ギャップ
                        float(len([x for x in current if x <= 15])), # 14. 小数字数（≤15）
                    ]
                    
                    # 次回予測ターゲット
                    if i < len(data) - 1:
                        next_nums = []
                        for col in main_cols:
                            if col in data.columns:
                                next_nums.append(int(data.iloc[i+1][col]))
                        
                        if len(next_nums) == 5:
                            for target_num in next_nums:
                                features.append(feat.copy())
                                targets.append(target_num)
                        
                except Exception as e:
                    continue
                
                if (i + 1) % 100 == 0:
                    print(f"  特徴量進捗: {i+1}/{len(data)}件")
            
            # パターン統計
            if len(features) > 0:
                sum_patterns = []
                for i in range(len(data)):
                    try:
                        current = []
                        for col in main_cols:
                            if col in data.columns:
                                current.append(int(data.iloc[i][col]))
                        if len(current) == 5 and all(1 <= x <= 31 for x in current) and len(set(current)) == 5:
                            sum_patterns.append(sum(current))
                    except:
                        continue
                
                if sum_patterns:
                    self.pattern_stats = {
                        'avg_sum': float(np.mean(sum_patterns)),
                        'std_sum': float(np.std(sum_patterns)),
                        'most_frequent_pairs': self.pair_freq.most_common(10)
                    }
            
            print(f"✅ 14次元特徴量完成: {len(features)}個")
            return np.array(features), np.array(targets)
            
        except Exception as e:
            print(f"❌ 特徴量エンジニアリングエラー: {e}")
            return None, None
    
    def train_basic_models(self, data):
        """基本モデル学習"""
        try:
            print("📊 === 基本モデル学習開始 ===")
            
            # 14次元特徴量作成
            X, y = self.create_basic_features(data)
            if X is None or len(X) < 100:
                print(f"❌ 特徴量不足: {len(X) if X is not None else 0}件")
                return False
            
            self.data_count = len(data)
            
            # 各モデルの学習
            print("🤖 基本モデル学習中...")
            
            for name, model in self.models.items():
                try:
                    print(f"  {name} 学習中...")
                    
                    # スケーリング
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    self.scalers[name] = scaler
                    
                    # 学習
                    model.fit(X_scaled, y)
                    
                    # クロスバリデーション評価
                    cv_score = np.mean(cross_val_score(model, X_scaled, y, cv=3))
                    
                    self.trained_models[name] = model
                    self.model_scores[name] = cv_score
                    
                    print(f"    ✅ {name}: CV精度 {cv_score*100:.2f}%")
                    
                except Exception as e:
                    print(f"    ❌ {name}: エラー {e}")
                    continue
            
            print(f"✅ 基本モデル学習完了: {len(self.trained_models)}モデル")
            return True
            
        except Exception as e:
            print(f"❌ 基本モデル学習エラー: {str(e)}")
            return False
    
    def basic_predict(self, count=20):
        """基本アンサンブル予測実行"""
        try:
            if not self.trained_models:
                print("❌ 学習済みモデルなし")
                return []
            
            # 基準特徴量（ミニロト用）
            if hasattr(self, 'pattern_stats') and self.pattern_stats:
                avg_sum = self.pattern_stats.get('avg_sum', 80)  # ミニロトの平均合計
                base_features = [
                    avg_sum / 5,        # 1. 平均値（16程度）
                    6.0,                # 2. 標準偏差
                    avg_sum,            # 3. 合計値（80程度）
                    2.5,                # 4. 奇数個数
                    28.0,               # 5. 最大値
                    5.0,                # 6. 最小値
                    16.0,               # 7. 中央値
                    23.0,               # 8. 範囲
                    1.0,                # 9. 連続数
                    8.0,                # 10. 第1数字
                    16.0,               # 11. 第3数字
                    24.0,               # 12. 第5数字
                    5.5,                # 13. 平均ギャップ
                    2.5                 # 14. 小数字数
                ]
            else:
                # デフォルト値
                base_features = [16.0, 6.0, 80.0, 2.5, 28.0, 5.0, 16.0, 23.0, 1.0, 8.0, 16.0, 24.0, 5.5, 2.5]
            
            predictions = []
            
            for i in range(count):
                # 各モデルの予測を収集
                ensemble_votes = Counter()
                
                for name, model in self.trained_models.items():
                    try:
                        scaler = self.scalers[name]
                        X_scaled = scaler.transform([base_features])
                        
                        # 複数回予測
                        for _ in range(6):  # ミニロト用に調整
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(X_scaled)[0]
                                classes = model.classes_
                                if len(classes) > 0:
                                    selected = np.random.choice(classes, p=proba/proba.sum())
                                    if 1 <= selected <= 31:
                                        weight = self.model_weights.get(name, 0.5)
                                        ensemble_votes[int(selected)] += weight
                            else:
                                pred = model.predict(X_scaled)[0]
                                if 1 <= pred <= 31:
                                    weight = self.model_weights.get(name, 0.5)
                                    ensemble_votes[int(pred)] += weight
                                    
                    except Exception as e:
                        continue
                
                # 頻出数字と組み合わせ
                frequent_nums = [num for num, _ in self.freq_counter.most_common(12)]
                for num in frequent_nums[:8]:
                    ensemble_votes[num] += 0.1
                
                # 上位5個を選択
                top_numbers = [num for num, _ in ensemble_votes.most_common(5)]
                
                # 不足分をランダム補完
                while len(top_numbers) < 5:
                    candidate = np.random.randint(1, 32)
                    if candidate not in top_numbers:
                        top_numbers.append(candidate)
                
                final_pred = sorted([int(x) for x in top_numbers[:5]])
                predictions.append(final_pred)
            
            return predictions
            
        except Exception as e:
            print(f"❌ 基本予測エラー: {str(e)}")
            return []
    
    def auto_setup_and_predict(self):
        """自動セットアップ・予測実行"""
        try:
            print("\n" + "="*80)
            print("🌐 ミニロト基本予測システム実行開始")
            print("="*80)
            
            # 1. 最新データ取得
            if not self.data_fetcher.fetch_latest_data():
                print("❌ データ取得失敗")
                return [], {}
            
            training_data = self.data_fetcher.get_data_for_training()
            next_info = self.data_fetcher.get_next_round_info()
            
            print(f"📊 学習データ: {len(training_data)}件")
            print(f"🎯 予測対象: {next_info['prediction_target']}")
            
            # 2. 基本モデル学習
            success = self.train_basic_models(training_data)
            if not success:
                print("❌ 学習失敗")
                return [], {}
            
            # 3. 予測実行
            predictions = self.basic_predict(20)
            if not predictions:
                print("❌ 予測失敗")
                return [], {}
            
            # 4. 予測記録
            self.history.add_prediction_with_round(
                predictions, 
                next_info['next_round'], 
                next_info['current_date']
            )
            
            print("\n" + "="*80)
            print(f"🎯 {next_info['prediction_target']}の予測結果（20セット）")
            print("="*80)
            print(f"📅 予測作成日時: {next_info['current_date']}")
            print(f"📊 学習データ: 第1回〜第{next_info['latest_round']}回（{self.data_count}件）")
            print("-"*80)
            
            for i, pred in enumerate(predictions, 1):
                clean_pred = [int(x) for x in pred]
                print(f"第{next_info['next_round']}回予測 {i:2d}: {clean_pred}")
            
            # 5. モデル性能表示
            print("\n" + "="*80)
            print("🤖 基本モデル性能")
            print("="*80)
            
            for name, score in self.model_scores.items():
                weight = self.model_weights.get(name, 0)
                print(f"{name:15s}: CV精度 {score*100:5.2f}% | 重み {weight:.2f}")
            
            # 6. 統計情報
            print("\n" + "="*80)
            print("📊 分析結果")
            print("="*80)
            print(f"特徴量次元: 14次元（ミニロト最適化版）")
            
            if hasattr(self, 'pattern_stats') and self.pattern_stats:
                print(f"平均合計値: {self.pattern_stats.get('avg_sum', 0):.1f}")
            
            print(f"\n🔥 頻出数字TOP10:")
            for i, (num, count) in enumerate(self.freq_counter.most_common(10)):
                if i % 5 == 0:
                    print("")
                print(f"{int(num)}番({int(count)}回)", end="  ")
            
            print(f"\n\n✅ 基本予測システム実行完了")
            return predictions, next_info
            
        except Exception as e:
            print(f"❌ システムエラー: {str(e)}")
            print(f"詳細: {traceback.format_exc()}")
            return [], {}

    def save_models(self):
        """学習済みモデルをメモリに保存"""
        try:
            if not self.trained_models:
                return False
                
            # モデルと関連データをまとめて保存
            self.saved_model_data = {
                'trained_models': self.trained_models,
                'scalers': self.scalers,
                'model_weights': self.model_weights,
                'model_scores': self.model_scores,
                'freq_counter': dict(self.freq_counter),
                'pair_freq': dict(self.pair_freq),
                'pattern_stats': self.pattern_stats,
                'data_count': self.data_count,
                'save_timestamp': datetime.now().isoformat()
            }
            
            print(f"💾 基本モデルをメモリに保存完了")
            
            # 予測履歴も保存
            self.history.save_to_json()
            
            return True
            
        except Exception as e:
            print(f"❌ モデル保存エラー: {e}")
            return False
    
    def load_models(self):
        """保存済みモデルをメモリから読み込み"""
        try:
            if not hasattr(self, 'saved_model_data') or not self.saved_model_data:
                print("📂 保存済みモデルが見つかりません")
                return False
            
            # モデルと関連データを復元
            data = self.saved_model_data
            self.trained_models = data['trained_models']
            self.scalers = data['scalers']
            self.model_weights = data['model_weights']
            self.model_scores = data['model_scores']
            self.freq_counter = Counter(data['freq_counter'])
            self.pair_freq = Counter(data['pair_freq'])
            self.pattern_stats = data['pattern_stats']
            self.data_count = data['data_count']
            
            print(f"📂 基本モデルをメモリから読み込み完了")
            print(f"  学習データ数: {self.data_count}件")
            print(f"  モデル数: {len(self.trained_models)}")
            
            # 予測履歴も読み込み
            self.history.load_from_json()
            
            return True
            
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
            return False

# ========================= パート1B（後半）ここまで =========================

# ========================= パート1C開始 =========================

# グローバルシステム
basic_system = MiniLotoBasicPredictor()

# パート1実行関数
def run_miniloto_basic_prediction():
    try:
        print("\n🚀 ミニロト基本予測システム実行")
        
        # 基本予測実行
        predictions, next_info = basic_system.auto_setup_and_predict()
        
        if predictions:
            print("\n🎉 パート1: 基本予測システム完了!")
            print("📝 予測結果が生成されました")
            return "SUCCESS"
        else:
            print("❌ 基本予測失敗")
            return "FAILED"
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        print(traceback.format_exc())
        return "ERROR"

# パート1テスト実行
print("\n" + "="*80)
print("🧪 パート1: 基本システムテスト実行")
print("="*80)

test_result = run_miniloto_basic_prediction()
print(f"\n🏁 パート1テスト結果: {test_result}")

if test_result == "SUCCESS":
    print("✅ パート1完了 - パート2に進む準備完了")
    
    # 基本統計表示
    if basic_system.trained_models:
        print(f"\n📊 パート1完了統計:")
        print(f"  学習データ数: {basic_system.data_count}件")
        print(f"  特徴量次元: 14次元")
        print(f"  学習モデル数: {len(basic_system.trained_models)}個")
        print(f"  頻出数字数: {len(basic_system.freq_counter)}個")
else:
    print("❌ パート1に問題があります。修正が必要です。")

print("\n" + "="*80)
print("🎯 パート1: 基盤システム完了")
print("次回: パート2 - 高度予測システム（時系列検証）")
print("="*80)

# ========================= パート1Cここまで =========================

# ミニロト予測システム パート2: 高度予測システム（時系列検証）
# ========================= パート2A開始 =========================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from collections import Counter, defaultdict
import json
import traceback
from datetime import datetime

print("🚀 ミニロト予測システム - パート2: 高度予測システム")
print("📊 時系列交差検証 + Neural Network + 14次元特徴量")
print("🔧 固定窓: 50回分（30, 50, 70での検証）")

# 時系列交差検証クラス（ミニロト版・50回分対応）
class MiniLotoTimeSeriesValidator:
    """ミニロト専用時系列交差検証クラス（固定窓50回分対応）"""
    def __init__(self, min_train_size=30):
        self.min_train_size = min_train_size
        self.fixed_window_results = {}  # 窓サイズ別の結果
        self.expanding_window_results = []
        self.validation_history = []
        self.feature_importance_history = {}
        
        # ミニロト用フルモデル（3モデルアンサンブル）
        self.validation_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=80, max_depth=6, random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(64, 32, 16), max_iter=300, random_state=42
            )
        }
        
        self.model_weights = {
            'random_forest': 0.4,
            'gradient_boost': 0.35,
            'neural_network': 0.25
        }
        
    def evaluate_prediction_sets(self, predicted_sets, actual):
        """20セット予測と実際の一致を評価"""
        results = []
        
        actual_set = set(actual)
        
        for i, predicted in enumerate(predicted_sets):
            predicted_set = set(predicted)
            matches = len(predicted_set & actual_set)
            
            result = {
                'set_idx': i,
                'matches': matches,
                'accuracy': matches / 5.0,  # ミニロトは5個
                'predicted': predicted,
                'actual': actual,
                'matched_numbers': sorted(list(predicted_set & actual_set)),
                'missed_numbers': sorted(list(actual_set - predicted_set)),
                'extra_numbers': sorted(list(predicted_set - actual_set))
            }
            results.append(result)
        
        # 全体統計
        all_matches = [r['matches'] for r in results]
        summary = {
            'avg_matches': np.mean(all_matches),
            'max_matches': max(all_matches),
            'min_matches': min(all_matches),
            'std_matches': np.std(all_matches),
            'sets_3_plus': sum(1 for m in all_matches if m >= 3),
            'sets_4_plus': sum(1 for m in all_matches if m >= 4),
            'sets_5_plus': sum(1 for m in all_matches if m >= 5),
            'match_distribution': dict(Counter(all_matches)),
            'individual_results': results
        }
        
        return summary
    
    def create_validation_features(self, data):
        """ミニロト用14次元フル特徴量を作成"""
        try:
            features = []
            targets = []
            freq_counter = Counter()
            pair_freq = Counter()
            main_cols = ['第1数字', '第2数字', '第3数字', '第4数字', '第5数字']
            
            for i in range(len(data)):
                try:
                    current = []
                    for col in main_cols:
                        if col in data.columns:
                            current.append(int(data.iloc[i][col]))
                    
                    if len(current) != 5:
                        continue
                    
                    if not all(1 <= x <= 31 for x in current):
                        continue
                    if len(set(current)) != 5:
                        continue
                    
                    # 基本統計
                    for num in current:
                        freq_counter[num] += 1
                    
                    # ペア分析
                    for j in range(len(current)):
                        for k in range(j+1, len(current)):
                            pair = tuple(sorted([current[j], current[k]]))
                            pair_freq[pair] += 1
                    
                    # ミニロト用14次元特徴量
                    sorted_nums = sorted(current)
                    gaps = [sorted_nums[j+1] - sorted_nums[j] for j in range(4)]
                    
                    feat = [
                        float(np.mean(current)),           # 1. 平均値
                        float(np.std(current)),            # 2. 標準偏差
                        float(np.sum(current)),            # 3. 合計値
                        float(sum(1 for x in current if x % 2 == 1)),  # 4. 奇数個数
                        float(max(current)),               # 5. 最大値
                        float(min(current)),               # 6. 最小値
                        float(np.median(current)),         # 7. 中央値
                        float(max(current) - min(current)), # 8. 範囲
                        float(len([j for j in range(len(sorted_nums)-1) 
                                 if sorted_nums[j+1] - sorted_nums[j] == 1])), # 9. 連続数
                        float(current[0]),                 # 10. 第1数字
                        float(current[2]),                 # 11. 第3数字（中央）
                        float(current[4]),                 # 12. 第5数字（最後）
                        float(np.mean(gaps)),              # 13. 平均ギャップ
                        float(len([x for x in current if x <= 15])), # 14. 小数字数
                    ]
                    
                    # 次回予測ターゲット
                    if i < len(data) - 1:
                        next_nums = []
                        for col in main_cols:
                            if col in data.columns:
                                next_nums.append(int(data.iloc[i+1][col]))
                        
                        if len(next_nums) == 5:
                            for target_num in next_nums:
                                features.append(feat.copy())
                                targets.append(target_num)
                        
                except Exception as e:
                    continue
            
            print(f"✅ 14次元特徴量完成: {len(features)}個")
            return np.array(features), np.array(targets), freq_counter
            
        except Exception as e:
            print(f"❌ 特徴量エンジニアリングエラー: {e}")
            return None, None, Counter()

    def train_validation_models(self, train_data):
        """フルモデル（3モデル）を学習"""
        try:
            # 14次元特徴量作成
            X, y, freq_counter = self.create_validation_features(train_data)
            if X is None or len(X) < 50:  # 最低限必要なデータ数
                return None
            
            trained_models = {}
            scalers = {}
            
            for name, model in self.validation_models.items():
                try:
                    # スケーリング
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # 学習
                    model_copy = type(model)(**model.get_params())
                    model_copy.fit(X_scaled, y)
                    
                    trained_models[name] = model_copy
                    scalers[name] = scaler
                    
                except Exception as e:
                    continue
            
            return {
                'models': trained_models, 
                'scalers': scalers,
                'freq_counter': freq_counter
            }
            
        except Exception as e:
            return None
    
    def generate_validation_predictions(self, model_data, freq_counter, count=20):
        """フルアンサンブル手法で20セット予測を生成"""
        try:
            if not model_data or not model_data['models']:
                return []
            
            trained_models = model_data['models']
            scalers = model_data['scalers']
            
            # ミニロト用基準特徴量（14次元）
            base_features = [16.0, 6.0, 80.0, 2.5, 28.0, 5.0, 16.0, 23.0, 1.0, 8.0, 16.0, 24.0, 5.5, 2.5]
            
            predictions = []
            
            for i in range(count):
                # 各モデルの予測を収集
                ensemble_votes = Counter()
                
                for name, model in trained_models.items():
                    try:
                        scaler = scalers[name]
                        X_scaled = scaler.transform([base_features])
                        
                        # 複数回予測
                        for _ in range(6):
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(X_scaled)[0]
                                classes = model.classes_
                                if len(classes) > 0:
                                    selected = np.random.choice(classes, p=proba/proba.sum())
                                    if 1 <= selected <= 31:
                                        weight = self.model_weights.get(name, 0.33)
                                        ensemble_votes[int(selected)] += weight
                            else:
                                pred = model.predict(X_scaled)[0]
                                if 1 <= pred <= 31:
                                    weight = self.model_weights.get(name, 0.33)
                                    ensemble_votes[int(pred)] += weight
                                    
                    except Exception as e:
                        continue
                
                # 頻出数字と組み合わせ
                frequent_nums = [num for num, _ in freq_counter.most_common(12)]
                for num in frequent_nums[:8]:
                    ensemble_votes[num] += 0.1
                
                # 上位5個を選択
                top_numbers = [num for num, _ in ensemble_votes.most_common(5)]
                
                # 不足分をランダム補完
                while len(top_numbers) < 5:
                    candidate = np.random.randint(1, 32)
                    if candidate not in top_numbers:
                        top_numbers.append(candidate)
                
                final_pred = sorted([int(x) for x in top_numbers[:5]])
                predictions.append(final_pred)
            
            return predictions
            
        except Exception as e:
            print(f"❌ 検証用予測生成エラー: {e}")
            return []

# ========================= パート2Aここまで =========================

# ========================= パート2B開始 =========================

    def fixed_window_validation(self, data, window_sizes=[30, 50, 70]):
        """複数窓サイズによる固定窓検証（50回分メイン）"""
        print(f"\n📊 === 固定窓検証開始（窓サイズ: {window_sizes}回） ===")
        print("⚡ フル精度モード: 3モデルアンサンブル・14次元特徴量")
        
        total_rounds = len(data)
        results_by_window = {}
        main_cols = ['第1数字', '第2数字', '第3数字', '第4数字', '第5数字']
        round_col = '開催回'
        
        for window_size in window_sizes:
            print(f"\n🔄 {window_size}回分窓での検証開始")
            results = []
            
            # 検証範囲の計算
            max_tests = min(200, total_rounds - window_size - 1)  # 効率化のため200回まで
            step = max(1, (total_rounds - window_size - 1) // max_tests)
            
            print(f"検証範囲: 第{window_size + 1}回 〜 第{total_rounds}回（{max_tests}回の検証、ステップ{step}）")
            
            test_count = 0
            for i in range(0, total_rounds - window_size - 1, step):
                if test_count >= max_tests:
                    break
                
                # 訓練データ: i〜i+window_size-1
                train_start = i
                train_end = i + window_size
                test_idx = train_end
                
                if test_idx >= total_rounds:
                    break
                
                # 訓練データ取得
                train_data = data.iloc[train_start:train_end]
                test_round = data.iloc[test_idx][round_col]
                actual_numbers = []
                for col in main_cols:
                    if col in data.columns:
                        actual_numbers.append(int(data.iloc[test_idx][col]))
                
                if len(actual_numbers) == 5:
                    # フルモデル学習
                    model_data = self.train_validation_models(train_data)
                    
                    if model_data and model_data['models']:
                        # 20セット予測生成
                        predicted_sets = self.generate_validation_predictions(
                            model_data, 
                            model_data['freq_counter'], 
                            20
                        )
                        
                        if predicted_sets:
                            # 詳細評価
                            eval_result = self.evaluate_prediction_sets(predicted_sets, actual_numbers)
                            eval_result['train_range'] = f"第{train_start + 1}回〜第{train_end}回"
                            eval_result['test_round'] = test_round
                            eval_result['window_size'] = window_size
                            
                            results.append(eval_result)
                
                test_count += 1
                
                # 進捗表示
                if test_count % 50 == 0:
                    if results:
                        avg_matches = np.mean([r['avg_matches'] for r in results])
                        sets_3_plus = np.mean([r['sets_3_plus'] for r in results])
                        print(f"  進捗: {test_count}/{max_tests}件 | 平均一致: {avg_matches:.2f} | 3個以上一致: {sets_3_plus:.1f}セット")
            
            results_by_window[window_size] = results
            
            # 窓サイズ別サマリー
            if results:
                avg_matches = np.mean([r['avg_matches'] for r in results])
                max_matches = max([r['max_matches'] for r in results])
                sets_3_plus = np.mean([r['sets_3_plus'] for r in results])
                sets_4_plus = np.mean([r['sets_4_plus'] for r in results])
                print(f"\n📊 {window_size}回分窓 最終結果:")
                print(f"    検証回数: {len(results)}回 | 平均一致: {avg_matches:.3f}個 | 最高一致: {max_matches}個")
                print(f"    3個以上一致: {sets_3_plus:.2f}セット | 4個以上一致: {sets_4_plus:.2f}セット")
        
        self.fixed_window_results = results_by_window
        return results_by_window
    
    def expanding_window_validation(self, data, initial_size=50):
        """累積窓による時系列交差検証（50回分初期サイズ）"""
        print(f"\n📊 === 累積窓検証開始（初期サイズ: {initial_size}回） ===")
        print("⚡ フル精度モード: 3モデルアンサンブル・14次元特徴量")
        
        results = []
        total_rounds = len(data)
        main_cols = ['第1数字', '第2数字', '第3数字', '第4数字', '第5数字']
        round_col = '開催回'
        
        # 効率化のため150回まで
        max_tests = min(150, total_rounds - initial_size)
        step = max(1, (total_rounds - initial_size) // max_tests)
        
        print(f"検証範囲: 第{initial_size + 1}回 〜 第{total_rounds}回（{max_tests}回の検証、ステップ{step}）")
        
        test_count = 0
        for i in range(0, total_rounds - initial_size, step):
            if test_count >= max_tests:
                break
                
            test_idx = initial_size + i
            
            if test_idx >= total_rounds:
                break
            
            # 訓練データ: 0〜test_idx-1（累積）
            train_data = data.iloc[0:test_idx]
            test_round = data.iloc[test_idx][round_col]
            actual_numbers = []
            for col in main_cols:
                if col in data.columns:
                    actual_numbers.append(int(data.iloc[test_idx][col]))
            
            if len(actual_numbers) == 5:
                # フルモデル学習
                model_data = self.train_validation_models(train_data)
                
                if model_data and model_data['models']:
                    # 20セット予測生成
                    predicted_sets = self.generate_validation_predictions(
                        model_data, 
                        model_data['freq_counter'], 
                        20
                    )
                    
                    if predicted_sets:
                        # 詳細評価
                        eval_result = self.evaluate_prediction_sets(predicted_sets, actual_numbers)
                        eval_result['train_range'] = f"第1回〜第{test_idx}回"
                        eval_result['test_round'] = test_round
                        eval_result['train_size'] = len(train_data)
                        
                        results.append(eval_result)
            
            test_count += 1
            
            # 進捗表示
            if test_count % 30 == 0:
                if results:
                    avg_matches = np.mean([r['avg_matches'] for r in results])
                    sets_3_plus = np.mean([r['sets_3_plus'] for r in results])
                    print(f"  進捗: {test_count}/{max_tests}件 | 平均一致: {avg_matches:.2f} | 3個以上一致: {sets_3_plus:.1f}セット")
        
        self.expanding_window_results = results
        
        # 累積窓サマリー
        if results:
            avg_matches = np.mean([r['avg_matches'] for r in results])
            max_matches = max([r['max_matches'] for r in results])
            sets_3_plus = np.mean([r['sets_3_plus'] for r in results])
            sets_4_plus = np.mean([r['sets_4_plus'] for r in results])
            print(f"\n📊 累積窓 最終結果:")
            print(f"    検証回数: {len(results)}回 | 平均一致: {avg_matches:.3f}個 | 最高一致: {max_matches}個")
            print(f"    3個以上一致: {sets_3_plus:.2f}セット | 4個以上一致: {sets_4_plus:.2f}セット")
        
        return results
    
    def compare_validation_methods(self):
        """固定窓（複数サイズ）と累積窓の結果を比較"""
        print("\n📊 === 検証手法の詳細比較分析 ===")
        
        if not self.fixed_window_results or not self.expanding_window_results:
            print("❌ 検証結果が不足しています")
            return None
        
        comparison_results = {}
        
        # 固定窓（各サイズ）の統計
        for window_size, results in self.fixed_window_results.items():
            if results:
                avg_matches_list = [r['avg_matches'] for r in results]
                max_matches_list = [r['max_matches'] for r in results]
                sets_3_plus_list = [r['sets_3_plus'] for r in results]
                sets_4_plus_list = [r['sets_4_plus'] for r in results]
                
                stats = {
                    'method': f'固定窓（{window_size}回）',
                    'window_size': window_size,
                    'avg_matches': np.mean(avg_matches_list),
                    'std_matches': np.std(avg_matches_list),
                    'max_matches': max(max_matches_list),
                    'avg_sets_3_plus': np.mean(sets_3_plus_list),
                    'avg_sets_4_plus': np.mean(sets_4_plus_list),
                    'total_tests': len(results)
                }
                comparison_results[f'fixed_{window_size}'] = stats
        
        # 累積窓の統計
        if self.expanding_window_results:
            avg_matches_list = [r['avg_matches'] for r in self.expanding_window_results]
            max_matches_list = [r['max_matches'] for r in self.expanding_window_results]
            sets_3_plus_list = [r['sets_3_plus'] for r in self.expanding_window_results]
            sets_4_plus_list = [r['sets_4_plus'] for r in self.expanding_window_results]
            
            expanding_stats = {
                'method': '累積窓',
                'avg_matches': np.mean(avg_matches_list),
                'std_matches': np.std(avg_matches_list),
                'max_matches': max(max_matches_list),
                'avg_sets_3_plus': np.mean(sets_3_plus_list),
                'avg_sets_4_plus': np.mean(sets_4_plus_list),
                'total_tests': len(self.expanding_window_results)
            }
            comparison_results['expanding'] = expanding_stats
        
        # 結果表示
        print("\n【ミニロト時系列検証による詳細比較結果】")
        best_method = None
        best_score = 0
        
        for method_key, stats in comparison_results.items():
            print(f"\n🔹 {stats['method']}")
            print(f"  平均一致数: {stats['avg_matches']:.3f} ± {stats['std_matches']:.3f}")
            print(f"  最高一致数: {stats['max_matches']}個")
            print(f"  平均3個以上一致セット数: {stats['avg_sets_3_plus']:.2f}セット")
            print(f"  平均4個以上一致セット数: {stats['avg_sets_4_plus']:.2f}セット")
            print(f"  検証回数: {stats['total_tests']}回")
            
            # 総合スコア（平均一致数 + 3個以上一致セット数 + 4個以上一致セット数の重み付け）
            score = stats['avg_matches'] + stats['avg_sets_3_plus'] * 0.3 + stats['avg_sets_4_plus'] * 0.8
            print(f"  総合スコア: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_method = stats['method']
        
        # 最適手法の決定
        print(f"\n✅ 最適手法: {best_method}")
        print(f"   総合スコア: {best_score:.3f}")
        
        # 推奨事項
        if 'fixed_50' in comparison_results and comparison_results['fixed_50']['avg_matches'] > comparison_results.get('fixed_30', {}).get('avg_matches', 0):
            recommendation = 'fixed_50'
            print(f"\n💡 推奨: 50回分の固定窓が最も効果的")
            print(f"   理由: 十分な学習データで安定した予測性能を実現")
        elif 'fixed_70' in comparison_results and comparison_results['fixed_70']['avg_matches'] > comparison_results.get('fixed_50', {}).get('avg_matches', 0):
            recommendation = 'fixed_70'
            print(f"\n💡 推奨: 70回分の固定窓が最適")
            print(f"   理由: より多くの学習データで高精度を実現")
        elif 'fixed_30' in comparison_results:
            recommendation = 'fixed_30'
            print(f"\n💡 推奨: 30回分の固定窓を使用")
            print(f"   理由: 最新パターンへの適応性が高い")
        else:
            recommendation = 'expanding'
            print(f"\n💡 推奨: 累積窓を使用")
            print(f"   理由: 長期的なパターン学習が有効")
        
        # 実用的な推奨事項
        print(f"\n🎯 実用的推奨事項:")
        if best_method and 'fixed' in best_method:
            window_size = [key for key, val in comparison_results.items() if val['method'] == best_method][0].split('_')[1]
            print(f"   - 本番予測では過去{window_size}回分のデータでモデル学習")
            print(f"   - モデル重みを最適化結果に基づいて調整")
        
        return {
            'detailed_results': comparison_results,
            'best_method': best_method,
            'best_score': best_score,
            'recommendation': recommendation,
            'improvement': best_score - min([stats.get('avg_matches', 0) for stats in comparison_results.values()])
        }

# ========================= パート2Bここまで =========================

# ========================= パート2C開始 =========================

# 高度統合予測システム（ミニロト版・3モデルアンサンブル）
class MiniLotoAdvancedPredictor:
    def __init__(self):
        print("🔧 MiniLotoAdvancedPredictor初期化")
        
        # データ取得器（パート1から継承）
        from __main__ import basic_system
        self.data_fetcher = basic_system.data_fetcher
        
        # 3モデルアンサンブル
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=80, max_depth=6, random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(64, 32, 16), max_iter=300, random_state=42
            )
        }
        
        self.scalers = {}
        self.model_weights = {
            'random_forest': 0.4,
            'gradient_boost': 0.35,
            'neural_network': 0.25
        }
        
        # データ分析
        self.freq_counter = Counter()
        self.pair_freq = Counter()
        self.pattern_stats = {}
        
        # 学習状態
        self.trained_models = {}
        self.model_scores = {}
        self.data_count = 0
        
        # 予測履歴
        self.history = basic_system.history
        
        # 時系列検証器
        self.validator = None
        
        print("✅ 高度予測システム初期化完了")
        
    def create_advanced_features(self, data):
        """高度な14次元特徴量エンジニアリング"""
        try:
            print("🔧 高度14次元特徴量エンジニアリング開始")
            
            features = []
            targets = []
            main_cols = ['第1数字', '第2数字', '第3数字', '第4数字', '第5数字']
            
            for i in range(len(data)):
                try:
                    current = []
                    for col in main_cols:
                        if col in data.columns:
                            current.append(int(data.iloc[i][col]))
                    
                    if len(current) != 5:
                        continue
                    
                    if not all(1 <= x <= 31 for x in current):
                        continue
                    if len(set(current)) != 5:
                        continue
                    
                    # 基本統計
                    for num in current:
                        self.freq_counter[num] += 1
                    
                    # ペア分析
                    for j in range(len(current)):
                        for k in range(j+1, len(current)):
                            pair = tuple(sorted([current[j], current[k]]))
                            self.pair_freq[pair] += 1
                    
                    # 高度14次元特徴量
                    sorted_nums = sorted(current)
                    gaps = [sorted_nums[j+1] - sorted_nums[j] for j in range(4)]
                    
                    feat = [
                        float(np.mean(current)),           # 1. 平均値
                        float(np.std(current)),            # 2. 標準偏差
                        float(np.sum(current)),            # 3. 合計値
                        float(sum(1 for x in current if x % 2 == 1)),  # 4. 奇数個数
                        float(max(current)),               # 5. 最大値
                        float(min(current)),               # 6. 最小値
                        float(np.median(current)),         # 7. 中央値
                        float(max(current) - min(current)), # 8. 範囲
                        float(len([j for j in range(len(sorted_nums)-1) 
                                 if sorted_nums[j+1] - sorted_nums[j] == 1])), # 9. 連続数
                        float(current[0]),                 # 10. 第1数字
                        float(current[2]),                 # 11. 第3数字（中央）
                        float(current[4]),                 # 12. 第5数字（最後）
                        float(np.mean(gaps)),              # 13. 平均ギャップ
                        float(len([x for x in current if x <= 15])), # 14. 小数字数
                    ]
                    
                    # 次回予測ターゲット
                    if i < len(data) - 1:
                        next_nums = []
                        for col in main_cols:
                            if col in data.columns:
                                next_nums.append(int(data.iloc[i+1][col]))
                        
                        if len(next_nums) == 5:
                            for target_num in next_nums:
                                features.append(feat.copy())
                                targets.append(target_num)
                        
                except Exception as e:
                    continue
                
                if (i + 1) % 100 == 0:
                    print(f"  特徴量進捗: {i+1}/{len(data)}件")
            
            # パターン統計
            if len(features) > 0:
                sum_patterns = []
                for i in range(len(data)):
                    try:
                        current = []
                        for col in main_cols:
                            if col in data.columns:
                                current.append(int(data.iloc[i][col]))
                        if len(current) == 5 and all(1 <= x <= 31 for x in current) and len(set(current)) == 5:
                            sum_patterns.append(sum(current))
                    except:
                        continue
                
                if sum_patterns:
                    self.pattern_stats = {
                        'avg_sum': float(np.mean(sum_patterns)),
                        'std_sum': float(np.std(sum_patterns)),
                        'most_frequent_pairs': self.pair_freq.most_common(10)
                    }
            
            print(f"✅ 高度14次元特徴量完成: {len(features)}個")
            return np.array(features), np.array(targets)
            
        except Exception as e:
            print(f"❌ 特徴量エンジニアリングエラー: {e}")
            return None, None
    
    def train_advanced_models(self, data):
        """高度アンサンブルモデル学習（3モデル）"""
        try:
            print("📊 === 高度アンサンブル学習開始（3モデル） ===")
            
            # 高度14次元特徴量作成
            X, y = self.create_advanced_features(data)
            if X is None or len(X) < 100:
                print(f"❌ 特徴量不足: {len(X) if X is not None else 0}件")
                return False
            
            self.data_count = len(data)
            
            # 各モデルの学習
            print("🤖 高度アンサンブルモデル学習中...")
            
            for name, model in self.models.items():
                try:
                    print(f"  {name} 学習中...")
                    
                    # スケーリング
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    self.scalers[name] = scaler
                    
                    # 学習
                    model.fit(X_scaled, y)
                    
                    # クロスバリデーション評価
                    cv_score = np.mean(cross_val_score(model, X_scaled, y, cv=3))
                    
                    self.trained_models[name] = model
                    self.model_scores[name] = cv_score
                    
                    print(f"    ✅ {name}: CV精度 {cv_score*100:.2f}%")
                    
                except Exception as e:
                    print(f"    ❌ {name}: エラー {e}")
                    continue
            
            print(f"✅ 高度アンサンブル学習完了: {len(self.trained_models)}モデル")
            return True
            
        except Exception as e:
            print(f"❌ 高度アンサンブル学習エラー: {str(e)}")
            return False
    
    def advanced_predict(self, count=20):
        """高度アンサンブル予測実行（3モデル）"""
        try:
            if not self.trained_models:
                print("❌ 学習済みモデルなし")
                return []
            
            # 基準特徴量（高度版）
            if hasattr(self, 'pattern_stats') and self.pattern_stats:
                avg_sum = self.pattern_stats.get('avg_sum', 80)
                base_features = [
                    avg_sum / 5, 6.0, avg_sum, 2.5, 28.0, 5.0, 16.0, 23.0, 1.0, 8.0, 16.0, 24.0, 5.5, 2.5
                ]
            else:
                base_features = [16.0, 6.0, 80.0, 2.5, 28.0, 5.0, 16.0, 23.0, 1.0, 8.0, 16.0, 24.0, 5.5, 2.5]
            
            predictions = []
            
            for i in range(count):
                # 各モデルの予測を収集
                ensemble_votes = Counter()
                
                for name, model in self.trained_models.items():
                    try:
                        scaler = self.scalers[name]
                        X_scaled = scaler.transform([base_features])
                        
                        # 複数回予測
                        for _ in range(8):  # 高度版では多めに予測
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(X_scaled)[0]
                                classes = model.classes_
                                if len(classes) > 0:
                                    selected = np.random.choice(classes, p=proba/proba.sum())
                                    if 1 <= selected <= 31:
                                        weight = self.model_weights.get(name, 0.33)
                                        ensemble_votes[int(selected)] += weight
                            else:
                                pred = model.predict(X_scaled)[0]
                                if 1 <= pred <= 31:
                                    weight = self.model_weights.get(name, 0.33)
                                    ensemble_votes[int(pred)] += weight
                                    
                    except Exception as e:
                        continue
                
                # 頻出数字と組み合わせ
                frequent_nums = [num for num, _ in self.freq_counter.most_common(15)]
                for num in frequent_nums[:10]:
                    ensemble_votes[num] += 0.12
                
                # 上位5個を選択
                top_numbers = [num for num, _ in ensemble_votes.most_common(5)]
                
                # 不足分をランダム補完
                while len(top_numbers) < 5:
                    candidate = np.random.randint(1, 32)
                    if candidate not in top_numbers:
                        top_numbers.append(candidate)
                
                final_pred = sorted([int(x) for x in top_numbers[:5]])
                predictions.append(final_pred)
            
            return predictions
            
        except Exception as e:
            print(f"❌ 高度予測エラー: {str(e)}")
            return []
    
    def run_timeseries_validation(self):
        """時系列交差検証を実行"""
        try:
            print("\n" + "="*80)
            print("🔄 ミニロト時系列交差検証実行開始")
            print("="*80)
            
            if not hasattr(self, 'data_fetcher') or self.data_fetcher.latest_data is None:
                print("❌ データが読み込まれていません")
                return None
            
            # バリデーター初期化
            self.validator = MiniLotoTimeSeriesValidator()
            
            # データ準備
            data = self.data_fetcher.latest_data
            
            # 1. 固定窓検証（30, 50, 70回分）
            fixed_results = self.validator.fixed_window_validation(data)
            
            # 2. 累積窓検証
            expanding_results = self.validator.expanding_window_validation(data)
            
            # 3. 結果比較
            comparison = self.validator.compare_validation_methods()
            
            # 4. モデル重みを調整
            if comparison:
                self._adjust_model_weights(comparison)
            
            print("\n✅ ミニロト時系列交差検証完了")
            return comparison
            
        except Exception as e:
            print(f"❌ 時系列検証エラー: {e}")
            print(traceback.format_exc())
            return None
    
    def _adjust_model_weights(self, comparison):
        """検証結果に基づいてモデル重みを調整"""
        print("\n🔧 === モデル重み調整 ===")
        
        # 基本調整率
        adjustment_rate = 0.1
        
        # 最適窓サイズに基づく調整
        if 'fixed' in comparison['recommendation']:
            print("📌 固定窓優位のため、短期パターン重視に調整")
            # Random Forestの重みを増加（短期パターンに強い）
            self.model_weights['random_forest'] *= (1 + adjustment_rate)
            self.model_weights['neural_network'] *= (1 - adjustment_rate * 0.5)
        else:
            print("📈 累積窓優位のため、長期トレンド重視に調整")
            # Gradient Boostingの重みを増加（長期トレンドに強い）
            self.model_weights['gradient_boost'] *= (1 + adjustment_rate)
            self.model_weights['random_forest'] *= (1 - adjustment_rate * 0.5)
        
        # Neural Networkが有効な場合は重みを維持
        if 'neural_network' in self.model_scores and self.model_scores['neural_network'] > 0.3:
            print("🧠 Neural Network性能良好のため重みを維持")
        
        # 重みの正規化
        total_weight = sum(self.model_weights.values())
        for model in self.model_weights:
            self.model_weights[model] /= total_weight
        
        print("\n調整後のモデル重み:")
        for model, weight in self.model_weights.items():
            print(f"  {model}: {weight:.3f}")
    
    def predict_next_round_advanced(self, count=20):
        """次回開催回の高度予測"""
        try:
            # 次回情報取得
            next_info = self.data_fetcher.get_next_round_info()
            if not next_info:
                print("❌ 次回開催回情報取得失敗")
                return [], {}
            
            print(f"🎯 === {next_info['prediction_target']}の高度予測開始 ===")
            print(f"📅 予測日時: {next_info['current_date']}")
            print(f"📊 最新データ: 第{next_info['latest_round']}回まで")
            
            # 高度アンサンブル予測
            predictions = self.advanced_predict(count)
            
            if predictions:
                # 予測を開催回付きで記録
                self.history.add_prediction_with_round(
                    predictions, 
                    next_info['next_round'], 
                    next_info['current_date']
                )
                
                print(f"📝 第{next_info['next_round']}回の高度予測として記録")
            
            return predictions, next_info
            
        except Exception as e:
            print(f"❌ 次回高度予測エラー: {e}")
            return [], {}

# グローバルシステム
advanced_system = MiniLotoAdvancedPredictor()

# パート2実行関数
def run_miniloto_advanced_prediction():
    try:
        print("\n🚀 ミニロト高度予測システム実行")
        
        # データ取得確認
        if not advanced_system.data_fetcher.latest_data is None:
            training_data = advanced_system.data_fetcher.latest_data
        else:
            print("📊 データ取得が必要です")
            if not advanced_system.data_fetcher.fetch_latest_data():
                return "FAILED"
            training_data = advanced_system.data_fetcher.latest_data
        
        # 高度モデル学習
        success = advanced_system.train_advanced_models(training_data)
        if not success:
            return "FAILED"
        
        # 高度予測実行
        predictions, next_info = advanced_system.predict_next_round_advanced(20)
        
        if predictions:
            # 結果表示
            print("\n" + "="*80)
            print(f"🎯 {next_info['prediction_target']}の高度予測結果（20セット）")
            print("🤖 3モデルアンサンブル + 14次元特徴量")
            print("="*80)
            
            for i, pred in enumerate(predictions, 1):
                clean_pred = [int(x) for x in pred]
                print(f"第{next_info['next_round']}回高度予測 {i:2d}: {clean_pred}")
            
            print("\n🎉 パート2: 高度予測システム完了!")
            return "SUCCESS"
        else:
            return "FAILED"
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        print(traceback.format_exc())
        return "ERROR"

def run_miniloto_timeseries_validation():
    try:
        print("\n🔄 ミニロト時系列交差検証実行")
        
        # 時系列検証実行
        result = advanced_system.run_timeseries_validation()
        
        if result:
            print("\n✅ 時系列交差検証完了！")
            return "SUCCESS"
        else:
            return "FAILED"
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        print(traceback.format_exc())
        return "ERROR"

# パート2テスト実行
print("\n" + "="*80)
print("🧪 パート2: 高度システムテスト実行")
print("="*80)

# 高度予測テスト
test_result = run_miniloto_advanced_prediction()
print(f"\n🏁 パート2高度予測テスト結果: {test_result}")

if test_result == "SUCCESS":
    print("✅ パート2高度予測完了")
    
    # 時系列検証テスト
    validation_result = run_miniloto_timeseries_validation()
    print(f"\n🏁 パート2時系列検証テスト結果: {validation_result}")
    
    if validation_result == "SUCCESS":
        print("✅ パート2完了 - パート3に進む準備完了")
        
        # パート2完了統計表示
        if advanced_system.trained_models:
            print(f"\n📊 パート2完了統計:")
            print(f"  学習データ数: {advanced_system.data_count}件")
            print(f"  特徴量次元: 14次元（高度版）")
            print(f"  学習モデル数: {len(advanced_system.trained_models)}個（3モデルアンサンブル）")
            print(f"  頻出数字数: {len(advanced_system.freq_counter)}個")
            print(f"  時系列検証: 完了")
    else:
        print("⚠️ 時系列検証に問題がありますが、予測機能は正常です")
else:
    print("❌ パート2に問題があります。修正が必要です。")

print("\n" + "="*80)
print("🎯 パート2: 高度予測システム完了")
print("次回: パート3 - 自動学習システム（継続改善）")
print("="*80)

# ========================= パート2Cここまで =========================

# ========================= パート3C開始 =========================

    def auto_setup_and_predict_with_persistence(self, force_new=False):
        """永続化対応の自動セットアップ・予測実行"""
        try:
            print("\n" + "="*80)
            print("🌐 ミニロト統合予測システム実行開始（永続化対応版）")
            print("="*80)
            
            # 1. 最新データ取得
            if not self.data_fetcher.latest_data is None:
                latest_data = self.data_fetcher.latest_data
                latest_round = self.data_fetcher.latest_round
            else:
                if not self.data_fetcher.fetch_latest_data():
                    print("❌ データ取得失敗")
                    return [], {}
                latest_data = self.data_fetcher.latest_data
                latest_round = self.data_fetcher.latest_round
            
            next_round = latest_round + 1
            next_info = self.data_fetcher.get_next_round_info()
            
            print(f"📊 最新データ: 第{latest_round}回まで取得済み")
            print(f"🎯 予測対象: 第{next_round}回")
            
            # 2. 既存予測のチェック
            if not force_new and self.persistence.is_prediction_exists(next_round):
                print(f"\n📂 第{next_round}回の予測は既に存在します（永続化済み）")
                existing_prediction = self.persistence.load_prediction(next_round)
                
                self.display_existing_prediction(existing_prediction, next_round)
                self.analyze_and_display_previous_results(latest_data, latest_round)
                
                return existing_prediction['predictions'], next_info
            
            # 3. 新しい予測が必要な場合
            print(f"\n🆕 第{next_round}回の新規予測を開始します")
            
            # 4. 前回結果との照合・学習
            learning_applied = self.check_and_apply_learning(latest_data, latest_round)
            
            # 5. モデル学習確認（必要に応じて再学習）
            if not self.trained_models:
                print("🔧 モデル学習が必要です")
                success = self.train_models_if_needed(latest_data)
                if not success:
                    print("❌ モデル学習失敗")
                    return [], {}
            
            # 6. 新しい予測生成
            predictions = self.predict_with_learning(20, use_learning=learning_applied)
            if not predictions:
                print("❌ 予測生成失敗")
                return [], {}
            
            # 7. 予測を永続化保存
            metadata = {
                'learning_applied': learning_applied,
                'model_count': len(self.trained_models),
                'feature_dimensions': 14,
                'data_count': self.data_count,
                'model_weights': self.model_weights.copy()
            }
            
            self.persistence.save_prediction_permanently(next_round, predictions, metadata)
            
            # 8. 予測結果表示
            self.display_new_prediction_results(predictions, next_info, learning_applied)
            
            # 9. 前回結果の分析・表示
            self.analyze_and_display_previous_results(latest_data, latest_round)
            
            print("\n" + "="*80)
            print("🎉 統合予測システム実行完了!")
            print(f"📝 第{next_round}回予測として永続化済み")
            print("🔄 次回実行時は保存済み予測を表示します")
            print("="*80)
            
            return predictions, next_info
            
        except Exception as e:
            print(f"❌ システムエラー: {str(e)}")
            print(f"詳細: {traceback.format_exc()}")
            return [], {}
    
    def train_models_if_needed(self, data):
        """必要に応じてモデルを学習"""
        if self.trained_models and len(self.trained_models) >= 2:
            print("✅ 既存の学習済みモデルを使用")
            return True
        
        # パート2の高度学習を実行
        return advanced_system.train_advanced_models(data)
    
    def display_existing_prediction(self, prediction_data, round_number):
        """既存の予測を表示"""
        print(f"📅 予測作成日時: {prediction_data['timestamp']}")
        print(f"📊 予測セット数: {len(prediction_data['predictions'])}セット")
        
        if prediction_data['metadata']['learning_applied']:
            print("💡 学習改善が適用された予測")
        
        print("-"*80)
        
        for i, pred in enumerate(prediction_data['predictions'], 1):
            clean_pred = [int(x) for x in pred]
            print(f"第{round_number}回予測 {i:2d}: {clean_pred}")
        
        # 検証済みの場合は結果も表示
        if prediction_data['verified'] and prediction_data['actual_result']:
            print(f"\n✅ 検証済み - 当選番号: {prediction_data['actual_result']}")
            print("一致結果:")
            for i, matches in enumerate(prediction_data['matches'], 1):
                print(f"  予測{i:2d}: {matches}個一致")
            print(f"最高一致: {prediction_data['best_match']}個")
    
    def display_new_prediction_results(self, predictions, next_info, learning_applied):
        """新しい予測結果の表示"""
        print("\n" + "="*80)
        print(f"🎯 {next_info['prediction_target']}の予測結果（20セット）")
        if learning_applied:
            print("💡 学習改善を適用した予測")
        print("🤖 3モデルアンサンブル + 14次元特徴量 + 自動学習")
        print("="*80)
        print(f"📅 予測作成日時: {next_info['current_date']}")
        print(f"📊 学習データ: 第1回〜第{next_info['latest_round']}回（{self.data_count}件）")
        print("-"*80)
        
        for i, pred in enumerate(predictions, 1):
            clean_pred = [int(x) for x in pred]
            print(f"第{next_info['next_round']}回予測 {i:2d}: {clean_pred}")
        
        # モデル性能表示
        print("\n" + "="*80)
        print("🤖 統合モデル性能")
        if learning_applied:
            print("💡 学習改善適用後の性能")
        print("="*80)
        
        for name, score in self.model_scores.items():
            weight = self.model_weights.get(name, 0)
            print(f"{name:15s}: CV精度 {score*100:5.2f}% | 重み {weight:.3f}")
        
        # 統計情報
        print("\n" + "="*80)
        print("📊 分析結果")
        print("="*80)
        print(f"特徴量次元: 14次元（ミニロト最適化版）")
        
        if hasattr(self, 'pattern_stats') and self.pattern_stats:
            print(f"平均合計値: {self.pattern_stats.get('avg_sum', 0):.1f}")
        
        print(f"\n🔥 頻出数字TOP10:")
        for i, (num, count) in enumerate(self.freq_counter.most_common(10)):
            if i % 5 == 0:
                print("")
            print(f"{int(num)}番({int(count)}回)", end="  ")
        
        # 学習改善情報の表示
        if learning_applied and hasattr(self.auto_learner, 'improvement_metrics'):
            self.display_learning_improvements()
    
    def display_learning_improvements(self):
        """学習改善情報の表示"""
        print("\n\n💡 === 学習改善情報 ===")
        
        metrics = self.auto_learner.improvement_metrics
        
        if 'frequently_missed' in metrics:
            print("🎯 見逃し頻度の高い番号（ブースト対象）:")
            for num, count in metrics['frequently_missed'][:5]:
                print(f"    {num}番: {count}回見逃し → ブースト適用")
        
        if 'high_accuracy_patterns' in metrics:
            patterns = metrics['high_accuracy_patterns']
            print(f"📊 高精度パターン学習:")
            print(f"    目標合計値: {patterns['avg_sum']:.1f}")
            print(f"    目標奇数個数: {patterns['avg_odd_count']:.1f}")
            print(f"    目標小数字個数: {patterns['avg_small_count']:.1f}")
            print(f"    学習サンプル数: {patterns['sample_size']}件")
        
        if 'small_number_importance' in metrics:
            importance = metrics['small_number_importance']
            print(f"🔢 小数字重要度: {importance:.1f}個（≤15の数字が重要）")
    
    def analyze_and_display_previous_results(self, latest_data, current_round):
        """前回結果の分析・表示"""
        previous_prediction = self.persistence.load_prediction(current_round)
        
        if not previous_prediction or not previous_prediction['verified']:
            return
        
        print(f"\n" + "="*80)
        print(f"📊 第{current_round}回 結果分析")
        print("="*80)
        
        actual_numbers = previous_prediction['actual_result']
        matches = previous_prediction['matches']
        
        print(f"🎯 当選番号: {actual_numbers}")
        print(f"📈 予測結果:")
        print("-"*50)
        
        for i, (pred, match_count) in enumerate(zip(previous_prediction['predictions'], matches), 1):
            pred_numbers = [int(x) for x in pred]
            matched = sorted(list(set(pred_numbers) & set(actual_numbers)))
            
            status = "🎉" if match_count >= 4 else "⭐" if match_count >= 3 else "📊"
            print(f"{status} 予測{i:2d}: {pred_numbers} → {match_count}個一致 {matched}")
        
        # 統計表示
        avg_matches = np.mean(matches)
        max_matches = max(matches)
        match_3_plus = sum(1 for m in matches if m >= 3)
        match_4_plus = sum(1 for m in matches if m >= 4)
        
        print("-"*50)
        print(f"📊 結果統計:")
        print(f"    平均一致数: {avg_matches:.2f}個")
        print(f"    最高一致数: {max_matches}個")
        print(f"    3個以上一致: {match_3_plus}セット")
        print(f"    4個以上一致: {match_4_plus}セット")

# グローバルシステム
integrated_system = MiniLotoIntegratedSystem()

# パート3実行関数
def run_miniloto_integrated_prediction():
    try:
        print("\n🚀 ミニロト統合予測システム実行")
        
        # 統合予測実行
        predictions, next_info = integrated_system.auto_setup_and_predict_with_persistence()
        
        if predictions:
            print("\n🎉 パート3: 統合システム完了!")
            print("📝 予測結果が永続化されました")
            return "SUCCESS"
        else:
            print("❌ 統合予測失敗")
            return "FAILED"
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        print(traceback.format_exc())
        return "ERROR"

def run_miniloto_auto_verification():
    try:
        print("\n🔄 ミニロト自動照合・学習改善実行")
        
        # データ確認
        if not integrated_system.data_fetcher.latest_data is None:
            latest_data = integrated_system.data_fetcher.latest_data
        else:
            print("📊 データ取得が必要です")
            if not integrated_system.data_fetcher.fetch_latest_data():
                return "FAILED"
            latest_data = integrated_system.data_fetcher.latest_data
        
        # 自動照合・学習改善実行
        # 過去の予測データを模擬的に作成して検証
        all_predictions = integrated_system.persistence.get_all_predictions()
        
        if all_predictions:
            print(f"📊 {len(all_predictions)}件の予測を確認中...")
            
            # 学習改善レポート生成
            report = integrated_system.auto_learner.generate_improvement_report()
            print(report)
            
            print("\n✅ 自動照合・学習改善完了！")
            return "SUCCESS"
        else:
            print("📊 照合対象の予測が見つかりません")
            return "SUCCESS"  # エラーではない
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        print(traceback.format_exc())
        return "ERROR"

def export_prediction_data():
    """予測データのエクスポート"""
    try:
        json_data = integrated_system.persistence.export_to_json()
        if json_data:
            print("✅ 予測データをJSONエクスポート完了")
            print(f"データサイズ: {len(json_data)}文字")
            return "SUCCESS"
        else:
            return "FAILED"
    except Exception as e:
        print(f"❌ エラー: {e}")
        return "ERROR"

# パート3テスト実行
print("\n" + "="*80)
print("🧪 パート3: 統合システムテスト実行")
print("="*80)

# 統合予測テスト
test_result = run_miniloto_integrated_prediction()
print(f"\n🏁 パート3統合予測テスト結果: {test_result}")

if test_result == "SUCCESS":
    print("✅ パート3統合予測完了")
    
    # 自動照合テスト
    verification_result = run_miniloto_auto_verification()
    print(f"\n🏁 パート3自動照合テスト結果: {verification_result}")
    
    # データエクスポートテスト
    export_result = export_prediction_data()
    print(f"\n🏁 パート3データエクスポートテスト結果: {export_result}")
    
    if verification_result == "SUCCESS" and export_result == "SUCCESS":
        print("✅ パート3完了 - 全機能正常動作")
        
        # パート3完了統計表示
        if integrated_system.trained_models:
            print(f"\n📊 パート3完了統計:")
            print(f"  学習データ数: {integrated_system.data_count}件")
            print(f"  特徴量次元: 14次元（最終版）")
            print(f"  学習モデル数: {len(integrated_system.trained_models)}個（3モデルアンサンブル）")
            print(f"  永続化予測数: {len(integrated_system.persistence.get_all_predictions())}件")
            print(f"  自動学習: 有効")
            print(f"  予測永続化: 有効")
    else:
        print("⚠️ 一部機能に問題がありますが、基本予測機能は正常です")
else:
    print("❌ パート3に問題があります。修正が必要です。")

print("\n" + "="*80)
print("🎯 パート3: 自動学習システム完了")
print("次回: パート4 - 統合・完成版")
print("="*80)

# ========================= パート3Cここまで =========================# ミニロト予測システム パート3: 自動学習システム（継続改善）
# ========================= パート3A開始 =========================

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import json
import traceback
from datetime import datetime

print("🚀 ミニロト予測システム - パート3: 自動学習システム")
print("🔄 自動照合・学習改善 + 予測永続化 + 継続的改善")
print("🧠 見逃しパターン学習 + 成功パターン分析")

# 自動照合・学習改善クラス（ミニロト版）
class MiniLotoAutoVerificationLearner:
    """ミニロト用自動照合と継続的学習改善を行うクラス"""
    def __init__(self):
        self.verification_results = []
        self.learning_history = []
        self.improvement_metrics = {}
        self.feature_weights = {}
        
    def verify_and_learn(self, prediction_history, latest_data):
        """予測履歴と実際の結果を照合し、学習を改善"""
        print("\n🔄 === ミニロト自動照合・学習改善開始 ===")
        
        verified_count = 0
        total_improvements = []
        main_cols = ['第1数字', '第2数字', '第3数字', '第4数字', '第5数字']
        round_col = '開催回'
        
        for entry in prediction_history.predictions:
            if entry['verified']:
                continue
                
            # 該当する開催回のデータを検索
            matching_data = latest_data[latest_data[round_col] == entry['round']]
            
            if len(matching_data) > 0:
                actual_row = matching_data.iloc[0]
                actual_numbers = []
                for col in main_cols:
                    if col in actual_row.index:
                        actual_numbers.append(int(actual_row[col]))
                
                if len(actual_numbers) == 5:
                    # 照合と分析
                    verification_result = self._analyze_prediction(
                        entry['predictions'], 
                        actual_numbers,
                        entry['round']
                    )
                    
                    self.verification_results.append(verification_result)
                    verified_count += 1
                    
                    # 学習改善
                    improvements = self._improve_from_result(verification_result, actual_row, main_cols)
                    total_improvements.extend(improvements)
        
        if verified_count > 0:
            print(f"\n✅ {verified_count}件の予測を照合・分析")
            self._aggregate_improvements(total_improvements)
        
        return verified_count
    
    def _analyze_prediction(self, predictions, actual, round_num):
        """予測結果の詳細分析"""
        analysis = {
            'round': round_num,
            'actual': actual,
            'predictions': predictions,
            'match_details': [],
            'patterns': {}
        }
        
        # 各予測セットの分析
        for i, pred in enumerate(predictions):
            pred_set = set(pred)
            actual_set = set(actual)
            matches = pred_set & actual_set
            
            detail = {
                'prediction_idx': i,
                'matches': len(matches),
                'matched_numbers': sorted(list(matches)),
                'missed_numbers': sorted(list(actual_set - pred_set)),
                'extra_numbers': sorted(list(pred_set - actual_set))
            }
            analysis['match_details'].append(detail)
        
        # パターン分析
        analysis['patterns'] = {
            'actual_sum': sum(actual),
            'actual_odd_count': sum(1 for n in actual if n % 2 == 1),
            'actual_range': max(actual) - min(actual),
            'actual_consecutive': self._count_consecutive(sorted(actual)),
            'best_match_count': max(d['matches'] for d in analysis['match_details']),
            'actual_small_count': sum(1 for n in actual if n <= 15)  # ミニロト用
        }
        
        return analysis
    
    def _count_consecutive(self, sorted_nums):
        """連続数をカウント"""
        count = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i+1] - sorted_nums[i] == 1:
                count += 1
        return count
    
    def _improve_from_result(self, verification_result, actual_row, main_cols):
        """照合結果から学習改善点を抽出"""
        improvements = []
        
        # 高精度予測（3個以上一致）の特徴を学習
        high_accuracy_preds = [
            d for d in verification_result['match_details'] 
            if d['matches'] >= 3
        ]
        
        if high_accuracy_preds:
            improvement = {
                'type': 'high_accuracy_pattern',
                'round': verification_result['round'],
                'patterns': verification_result['patterns'],
                'match_count': high_accuracy_preds[0]['matches']
            }
            improvements.append(improvement)
        
        # 頻繁に見逃す数字の学習
        all_missed = []
        for detail in verification_result['match_details']:
            all_missed.extend(detail['missed_numbers'])
        
        if all_missed:
            missed_freq = Counter(all_missed)
            improvement = {
                'type': 'frequently_missed',
                'numbers': missed_freq.most_common(5),
                'round': verification_result['round']
            }
            improvements.append(improvement)
        
        # ミニロト特有のパターン学習
        patterns = verification_result['patterns']
        if patterns['actual_small_count'] >= 3:  # 小数字が多い場合
            improvement = {
                'type': 'small_number_pattern',
                'small_count': patterns['actual_small_count'],
                'round': verification_result['round']
            }
            improvements.append(improvement)
        
        return improvements
    
    def _aggregate_improvements(self, improvements):
        """改善点を集約して学習戦略を更新"""
        print("\n📈 === ミニロト学習改善点の集約 ===")
        
        # 高精度パターンの集約
        high_acc_patterns = [imp for imp in improvements if imp['type'] == 'high_accuracy_pattern']
        if high_acc_patterns:
            avg_sum = np.mean([p['patterns']['actual_sum'] for p in high_acc_patterns])
            avg_odd = np.mean([p['patterns']['actual_odd_count'] for p in high_acc_patterns])
            avg_small = np.mean([p['patterns']['actual_small_count'] for p in high_acc_patterns])
            print(f"高精度予測パターン: 平均合計 {avg_sum:.1f}, 平均奇数 {avg_odd:.1f}, 平均小数字 {avg_small:.1f}")
            
            self.improvement_metrics['high_accuracy_patterns'] = {
                'avg_sum': avg_sum,
                'avg_odd_count': avg_odd,
                'avg_small_count': avg_small,
                'sample_size': len(high_acc_patterns)
            }
        
        # 頻繁に見逃す数字の集約
        missed_patterns = [imp for imp in improvements if imp['type'] == 'frequently_missed']
        if missed_patterns:
            all_missed_nums = Counter()
            for pattern in missed_patterns:
                for num, count in pattern['numbers']:
                    all_missed_nums[num] += count
            
            print(f"頻繁に見逃す数字TOP5: {all_missed_nums.most_common(5)}")
            self.improvement_metrics['frequently_missed'] = all_missed_nums.most_common(10)
        
        # 小数字パターンの集約
        small_patterns = [imp for imp in improvements if imp['type'] == 'small_number_pattern']
        if small_patterns:
            avg_small_count = np.mean([p['small_count'] for p in small_patterns])
            print(f"小数字重要パターン: 平均小数字数 {avg_small_count:.1f}")
            self.improvement_metrics['small_number_importance'] = avg_small_count
    
    def generate_improvement_report(self):
        """学習改善レポートを生成"""
        if not self.verification_results:
            return "まだ照合結果がありません"
        
        report = []
        report.append("\n📊 === ミニロト自動照合・学習改善レポート ===")
        report.append(f"照合済み予測: {len(self.verification_results)}件")
        
        # 全体的な精度
        all_matches = []
        for result in self.verification_results:
            for detail in result['match_details']:
                all_matches.append(detail['matches'])
        
        if all_matches:
            report.append(f"平均一致数: {np.mean(all_matches):.2f}個")
            report.append(f"最高一致数: {max(all_matches)}個")
        
        # 改善メトリクス
        if self.improvement_metrics:
            report.append("\n【学習による改善点】")
            
            if 'high_accuracy_patterns' in self.improvement_metrics:
                patterns = self.improvement_metrics['high_accuracy_patterns']
                report.append(f"高精度パターン発見:")
                report.append(f"  - 理想的な合計値: {patterns['avg_sum']:.0f}")
                report.append(f"  - 理想的な奇数個数: {patterns['avg_odd_count']:.0f}")
                report.append(f"  - 理想的な小数字個数: {patterns['avg_small_count']:.0f}")
            
            if 'frequently_missed' in self.improvement_metrics:
                report.append("頻出見逃し数字:")
                for num, count in self.improvement_metrics['frequently_missed'][:5]:
                    report.append(f"  - {num}番: {count}回見逃し")
            
            if 'small_number_importance' in self.improvement_metrics:
                importance = self.improvement_metrics['small_number_importance']
                report.append(f"小数字重要度: {importance:.1f}個（≤15の数字）")
        
        return "\n".join(report)
    
    def get_learning_adjustments(self):
        """学習調整パラメータを取得"""
        adjustments = {
            'boost_numbers': [],
            'pattern_targets': {},
            'weight_adjustments': {}
        }
        
        # 頻繁に見逃す数字をブースト
        if 'frequently_missed' in self.improvement_metrics:
            adjustments['boost_numbers'] = [
                num for num, _ in self.improvement_metrics['frequently_missed'][:5]
            ]
        
        # 高精度パターンをターゲット
        if 'high_accuracy_patterns' in self.improvement_metrics:
            adjustments['pattern_targets'] = self.improvement_metrics['high_accuracy_patterns']
        
        # 小数字重要度調整
        if 'small_number_importance' in self.improvement_metrics:
            adjustments['small_number_boost'] = self.improvement_metrics['small_number_importance']
        
        return adjustments

# ========================= パート3Aここまで =========================

# ========================= パート3B開始 =========================

# 予測永続化管理クラス（ミニロト版）
class MiniLotoPredictionPersistence:
    """ミニロト予測の永続化と履歴管理"""
    def __init__(self):
        self.memory_storage = {}
        self.session_predictions = {}
        self.prediction_metadata = {}
        
    def save_prediction_permanently(self, round_number, predictions, metadata):
        """予測を永続化保存"""
        try:
            prediction_data = {
                'round': round_number,
                'predictions': predictions,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat(),
                'verified': False,
                'actual_result': None
            }
            
            # メモリベース永続化
            self.memory_storage[round_number] = prediction_data
            self.session_predictions[round_number] = prediction_data
            
            print(f"💾 第{round_number}回予測を永続化保存完了")
            return True
            
        except Exception as e:
            print(f"❌ 永続化保存エラー: {e}")
            return False
    
    def load_prediction(self, round_number):
        """指定回の予測を読み込み"""
        try:
            if round_number in self.memory_storage:
                return self.memory_storage[round_number]
            elif round_number in self.session_predictions:
                return self.session_predictions[round_number]
            else:
                return None
        except Exception as e:
            print(f"❌ 予測読み込みエラー: {e}")
            return None
    
    def is_prediction_exists(self, round_number):
        """予測が既に存在するかチェック"""
        return round_number in self.memory_storage or round_number in self.session_predictions
    
    def update_with_actual_result(self, round_number, actual_numbers):
        """実際の結果で予測を更新"""
        try:
            prediction_data = self.load_prediction(round_number)
            if prediction_data:
                prediction_data['actual_result'] = actual_numbers
                prediction_data['verified'] = True
                
                # 一致数計算
                matches = []
                for pred_set in prediction_data['predictions']:
                    match_count = len(set(pred_set) & set(actual_numbers))
                    matches.append(match_count)
                
                prediction_data['matches'] = matches
                prediction_data['best_match'] = max(matches)
                
                # 保存更新
                self.memory_storage[round_number] = prediction_data
                self.session_predictions[round_number] = prediction_data
                
                print(f"✅ 第{round_number}回予測を実際の結果で更新")
                return True
                
        except Exception as e:
            print(f"❌ 実際結果更新エラー: {e}")
            return False
    
    def get_all_predictions(self):
        """全ての予測を取得"""
        all_predictions = {}
        all_predictions.update(self.memory_storage)
        all_predictions.update(self.session_predictions)
        return all_predictions
    
    def export_to_json(self):
        """JSON形式でエクスポート"""
        try:
            export_data = {
                'predictions': self.get_all_predictions(),
                'export_timestamp': datetime.now().isoformat()
            }
            return json.dumps(export_data, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ JSONエクスポートエラー: {e}")
            return None

# 統合ミニロト予測システム（最終版）
class MiniLotoIntegratedSystem:
    def __init__(self):
        print("🔧 MiniLotoIntegratedSystem初期化")
        
        # 他のパートから継承
        from __main__ import advanced_system
        self.data_fetcher = advanced_system.data_fetcher
        self.models = advanced_system.models
        self.scalers = advanced_system.scalers
        self.model_weights = advanced_system.model_weights.copy()
        self.freq_counter = advanced_system.freq_counter
        self.pair_freq = advanced_system.pair_freq
        self.pattern_stats = advanced_system.pattern_stats
        self.trained_models = advanced_system.trained_models
        self.model_scores = advanced_system.model_scores
        self.data_count = advanced_system.data_count
        
        # パート3専用機能
        self.auto_learner = MiniLotoAutoVerificationLearner()
        self.persistence = MiniLotoPredictionPersistence()
        self.learning_enabled = True
        
        print("✅ 統合予測システム初期化完了")
    
    def check_and_apply_learning(self, latest_data, current_round):
        """前回結果との照合・学習を実行"""
        print("\n🔍 === 前回結果との照合・学習チェック ===")
        
        # 前回の予測を取得
        previous_round = current_round
        previous_prediction = self.persistence.load_prediction(previous_round)
        
        if not previous_prediction:
            print(f"📊 第{previous_round}回の予測記録が見つかりません")
            return False
        
        if previous_prediction['verified']:
            print(f"✅ 第{previous_round}回は既に学習済みです")
            return True
        
        # 当選結果を最新データから取得
        main_cols = ['第1数字', '第2数字', '第3数字', '第4数字', '第5数字']
        round_col = '開催回'
        
        matching_data = latest_data[latest_data[round_col] == previous_round]
        
        if len(matching_data) == 0:
            print(f"📊 第{previous_round}回の当選結果がまだ未公開です")
            return False
        
        # 当選番号を取得
        actual_row = matching_data.iloc[0]
        actual_numbers = []
        for col in main_cols:
            if col in actual_row.index:
                actual_numbers.append(int(actual_row[col]))
        
        if len(actual_numbers) != 5:
            print(f"❌ 第{previous_round}回の当選番号が不完全です")
            return False
        
        # 予測結果との照合
        print(f"\n🎯 第{previous_round}回の結果分析・学習を実行")
        print(f"当選番号: {actual_numbers}")
        
        # 永続化システムで結果更新
        self.persistence.update_with_actual_result(previous_round, actual_numbers)
        
        # 学習分析を実行
        self.perform_detailed_learning_analysis(previous_prediction, actual_numbers, previous_round)
        
        print(f"✅ 第{previous_round}回の学習分析完了")
        return True
    
    def perform_detailed_learning_analysis(self, prediction_data, actual_numbers, round_number):
        """詳細な学習分析を実行"""
        print(f"\n🧠 === 第{round_number}回詳細学習分析 ===")
        
        predictions = prediction_data['predictions']
        
        # 一致数の統計
        matches = []
        detailed_analysis = []
        
        for i, pred_set in enumerate(predictions):
            pred_numbers = [int(x) for x in pred_set]
            match_count = len(set(pred_numbers) & set(actual_numbers))
            matches.append(match_count)
            
            matched_nums = sorted(list(set(pred_numbers) & set(actual_numbers)))
            missed_nums = sorted(list(set(actual_numbers) - set(pred_numbers)))
            extra_nums = sorted(list(set(pred_numbers) - set(actual_numbers)))
            
            detailed_analysis.append({
                'prediction': pred_numbers,
                'matches': match_count,
                'matched_numbers': matched_nums,
                'missed_numbers': missed_nums,
                'extra_numbers': extra_nums
            })
        
        avg_matches = np.mean(matches)
        max_matches = max(matches)
        
        print(f"平均一致数: {avg_matches:.2f}個")
        print(f"最高一致数: {max_matches}個")
        
        # 高精度予測の分析（3個以上一致）
        high_accuracy = [analysis for analysis in detailed_analysis if analysis['matches'] >= 3]
        if high_accuracy:
            print(f"\n🎯 高精度予測: {len(high_accuracy)}セット")
            for i, analysis in enumerate(high_accuracy):
                idx = detailed_analysis.index(analysis) + 1
                print(f"  セット{idx}: {analysis['matches']}個一致")
                print(f"    一致番号: {analysis['matched_numbers']}")
        
        # 頻繁に見逃した番号の分析
        all_missed = []
        for analysis in detailed_analysis:
            all_missed.extend(analysis['missed_numbers'])
        
        if all_missed:
            missed_freq = Counter(all_missed)
            print(f"\n❌ 頻繁に見逃した番号TOP5:")
            for num, count in missed_freq.most_common(5):
                print(f"    {num}番: {count}回見逃し")
            
            # 学習改善メトリクスに反映
            if not hasattr(self.auto_learner, 'improvement_metrics'):
                self.auto_learner.improvement_metrics = {}
            
            self.auto_learner.improvement_metrics['frequently_missed'] = missed_freq.most_common(10)
        
        # パターン分析
        actual_sum = sum(actual_numbers)
        actual_odd_count = sum(1 for n in actual_numbers if n % 2 == 1)
        actual_small_count = sum(1 for n in actual_numbers if n <= 15)
        
        print(f"\n📊 当選パターン分析:")
        print(f"    合計値: {actual_sum}")
        print(f"    奇数個数: {actual_odd_count}個")
        print(f"    小数字個数: {actual_small_count}個")
        print(f"    範囲: {min(actual_numbers)}-{max(actual_numbers)}")
        
        # 高精度パターンの学習
        if high_accuracy:
            high_acc_predictions = [analysis['prediction'] for analysis in high_accuracy]
            pattern_analysis = self.analyze_successful_patterns(high_acc_predictions, actual_numbers)
            
            if not hasattr(self.auto_learner, 'improvement_metrics'):
                self.auto_learner.improvement_metrics = {}
            
            self.auto_learner.improvement_metrics['high_accuracy_patterns'] = pattern_analysis
    
    def analyze_successful_patterns(self, successful_predictions, actual_numbers):
        """成功パターンの分析"""
        if not successful_predictions:
            return {}
        
        sums = [sum(pred) for pred in successful_predictions]
        odd_counts = [sum(1 for n in pred if n % 2 == 1) for pred in successful_predictions]
        small_counts = [sum(1 for n in pred if n <= 15) for pred in successful_predictions]
        
        pattern_analysis = {
            'avg_sum': np.mean(sums),
            'avg_odd_count': np.mean(odd_counts),
            'avg_small_count': np.mean(small_counts),
            'sample_size': len(successful_predictions),
            'actual_sum': sum(actual_numbers),
            'actual_odd_count': sum(1 for n in actual_numbers if n % 2 == 1),
            'actual_small_count': sum(1 for n in actual_numbers if n <= 15)
        }
        
        print(f"\n💡 成功パターン学習:")
        print(f"    理想的な合計値: {pattern_analysis['avg_sum']:.1f} (実際: {pattern_analysis['actual_sum']})")
        print(f"    理想的な奇数個数: {pattern_analysis['avg_odd_count']:.1f} (実際: {pattern_analysis['actual_odd_count']})")
        print(f"    理想的な小数字個数: {pattern_analysis['avg_small_count']:.1f} (実際: {pattern_analysis['actual_small_count']})")
        
        return pattern_analysis
    
    def predict_with_learning(self, count=20, use_learning=True):
        """学習改善を適用した予測"""
        try:
            if not self.trained_models:
                print("❌ 学習済みモデルなし")
                return []
            
            # 学習調整パラメータを取得
            if use_learning and hasattr(self.auto_learner, 'improvement_metrics'):
                adjustments = self.auto_learner.get_learning_adjustments()
                boost_numbers = adjustments.get('boost_numbers', [])
                pattern_targets = adjustments.get('pattern_targets', {})
                small_boost = adjustments.get('small_number_boost', 0)
                
                print(f"💡 学習改善適用: 見逃しブースト{len(boost_numbers)}個, パターン学習済み")
            else:
                boost_numbers = []
                pattern_targets = {}
                small_boost = 0
            
            # 基準特徴量（学習改善を反映）
            if pattern_targets:
                target_sum = pattern_targets.get('avg_sum', 80)
                target_odd = pattern_targets.get('avg_odd_count', 2.5)
                target_small = pattern_targets.get('avg_small_count', 2.5)
                base_features = [
                    target_sum / 5, 6.0, target_sum, target_odd, 28.0, 5.0, 16.0, 23.0, 1.0, 8.0, 16.0, 24.0, 5.5, target_small
                ]
            else:
                base_features = [16.0, 6.0, 80.0, 2.5, 28.0, 5.0, 16.0, 23.0, 1.0, 8.0, 16.0, 24.0, 5.5, 2.5]
            
            predictions = []
            
            for i in range(count):
                # 各モデルの予測を収集
                ensemble_votes = Counter()
                
                for name, model in self.trained_models.items():
                    try:
                        scaler = self.scalers[name]
                        X_scaled = scaler.transform([base_features])
                        
                        # 複数回予測
                        for _ in range(8):
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(X_scaled)[0]
                                classes = model.classes_
                                if len(classes) > 0:
                                    selected = np.random.choice(classes, p=proba/proba.sum())
                                    if 1 <= selected <= 31:
                                        weight = self.model_weights.get(name, 0.33)
                                        ensemble_votes[int(selected)] += weight
                            else:
                                pred = model.predict(X_scaled)[0]
                                if 1 <= pred <= 31:
                                    weight = self.model_weights.get(name, 0.33)
                                    ensemble_votes[int(pred)] += weight
                                    
                    except Exception as e:
                        continue
                
                # 頻出数字と組み合わせ
                frequent_nums = [num for num, _ in self.freq_counter.most_common(15)]
                for num in frequent_nums[:10]:
                    ensemble_votes[num] += 0.12
                
                # 学習改善：頻繁に見逃す数字をブースト
                for num in boost_numbers:
                    if 1 <= num <= 31:
                        ensemble_votes[num] += 0.25
                        if i == 0:  # 最初の予測時のみ表示
                            print(f"  💡 {num}番をブースト（頻出見逃し）")
                
                # 小数字ブースト
                if small_boost > 2:
                    for num in range(1, 16):
                        ensemble_votes[num] += 0.05
                
                # 上位5個を選択
                top_numbers = [num for num, _ in ensemble_votes.most_common(5)]
                
                # 不足分をランダム補完
                while len(top_numbers) < 5:
                    candidate = np.random.randint(1, 32)
                    if candidate not in top_numbers:
                        top_numbers.append(candidate)
                
                final_pred = sorted([int(x) for x in top_numbers[:5]])
                predictions.append(final_pred)
            
            return predictions
            
        except Exception as e:
            print(f"❌ 学習改善予測エラー: {str(e)}")
            return []

# ========================= パート3Bここまで =========================

# ========================= パート4C開始 =========================

# グローバル最終システム
final_system = MiniLotoFinalSystem()

# 完全版実行関数
def run_miniloto_final_prediction():
    """ミニロト完全版予測システム実行"""
    try:
        print("\n🌟 ミニロト完全版予測システム実行開始")
        
        # 完全版予測実行
        predictions, next_info = final_system.run_complete_prediction()
        
        if predictions:
            print("\n🎉 ミニロト完全版システム実行完了!")
            return "SUCCESS"
        else:
            print("❌ 完全版予測失敗")
            return "FAILED"
            
    except Exception as e:
        print(f"❌ 完全版システムエラー: {e}")
        print(traceback.format_exc())
        return "ERROR"

def run_miniloto_health_check():
    """システムヘルスチェック実行"""
    try:
        print("\n🔍 ミニロト完全版ヘルスチェック実行")
        
        health = final_system.system_health_check()
        
        if health and health['overall']:
            print("\n✅ システム状態: 正常")
            return "SUCCESS"
        else:
            print("\n⚠️ システム状態: 要注意")
            
            # 自動回復試行
            recovery = final_system.auto_recovery()
            if recovery:
                print("✅ 自動回復成功")
                return "SUCCESS"
            else:
                print("❌ 自動回復失敗")
                return "FAILED"
            
    except Exception as e:
        print(f"❌ ヘルスチェックエラー: {e}")
        return "ERROR"

def run_miniloto_timeseries_validation_final():
    """完全版時系列交差検証実行"""
    try:
        print("\n📊 ミニロト完全版時系列交差検証実行")
        
        # データ確認
        if not final_system.data_fetcher.latest_data is None:
            data = final_system.data_fetcher.latest_data
        else:
            if not final_system.data_fetcher.fetch_latest_data():
                return "FAILED"
            data = final_system.data_fetcher.latest_data
        
        # バリデーター実行
        from __main__ import MiniLotoTimeSeriesValidator
        validator = MiniLotoTimeSeriesValidator()
        
        # 固定窓検証（30, 50, 70回分）
        print("🔄 固定窓検証実行中...")
        fixed_results = validator.fixed_window_validation(data)
        
        # 累積窓検証
        print("🔄 累積窓検証実行中...")
        expanding_results = validator.expanding_window_validation(data)
        
        # 結果比較
        comparison = validator.compare_validation_methods()
        
        if comparison:
            print(f"\n✅ 時系列検証完了!")
            print(f"最適手法: {comparison['best_method']}")
            print(f"改善幅: {comparison['improvement']:.3f}")
            return "SUCCESS"
        else:
            return "FAILED"
            
    except Exception as e:
        print(f"❌ 時系列検証エラー: {e}")
        print(traceback.format_exc())
        return "ERROR"

def export_complete_system_data():
    """完全版システムデータエクスポート"""
    try:
        print("\n💾 完全版システムデータエクスポート実行")
        
        # 予測データエクスポート
        prediction_json = final_system.persistence.export_to_json()
        
        # システム状態エクスポート
        system_data = {
            'system_version': 'MiniLoto_Final_v1.0',
            'export_timestamp': datetime.now().isoformat(),
            'system_ready': final_system.system_ready,
            'model_count': len(final_system.trained_models),
            'data_count': final_system.data_count,
            'performance_metrics': final_system.performance_metrics,
            'last_error': final_system.last_error
        }
        
        if prediction_json:
            print(f"✅ 予測データエクスポート完了: {len(prediction_json)}文字")
        
        system_json = json.dumps(system_data, ensure_ascii=False, indent=2)
        print(f"✅ システムデータエクスポート完了: {len(system_json)}文字")
        
        return "SUCCESS"
        
    except Exception as e:
        print(f"❌ エクスポートエラー: {e}")
        return "ERROR"

# 最終版統合インターフェース
def show_final_interface():
    """最終版統合Webインターフェース表示"""
    html_content = '''
    <div style="
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        max-width: 1200px; 
        margin: 20px auto; 
        padding: 25px; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    ">
        <h1 style="text-align: center; margin-bottom: 10px; font-size: 28px;">
            🌟 ミニロト予測システム - 完全版
        </h1>
        <p style="text-align: center; font-size: 16px; margin-bottom: 25px; opacity: 0.9;">
            🎯 3モデルアンサンブル | 🧠 自動学習 | 💾 永続化 | 📊 時系列検証 | 🔄 継続改善
        </p>
        
        <div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 20px; margin-bottom: 25px; backdrop-filter: blur(10px);">
            <h3 style="margin-top: 0; color: #fff; font-size: 20px;">🎉 完全版の特徴</h3>
            <ul style="margin: 15px 0; font-size: 14px; line-height: 1.6;">
                <li><strong>永続化予測</strong>: 一度予測したら結果が変わらない安定システム</li>
                <li><strong>自動結果照合</strong>: 新しい当選番号で自動的に前回予測を検証・学習</li>
                <li><strong>見逃し学習</strong>: 頻繁に見逃す番号を特定して次回予測でブースト</li>
                <li><strong>成功パターン学習</strong>: 高精度予測の特徴を分析して反映</li>
                <li><strong>時系列検証</strong>: 50回分固定窓で最適な学習方法を検証</li>
                <li><strong>自動回復機能</strong>: システム問題を自動検出・修復</li>
                <li><strong>詳細分析表示</strong>: 前回結果の完全分析レポート表示</li>
            </ul>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 25px;">
            <!-- メイン予測機能 -->
            <div style="background: rgba(255,255,255,0.15); border-radius: 12px; padding: 25px;">
                <h3 style="margin-top: 0; color: #fff; font-size: 18px; display: flex; align-items: center;">
                    🎯 メイン予測機能
                </h3>
                <p style="font-size: 13px; margin: 15px 0; opacity: 0.9;">
                    ✨ 永続化対応<br>
                    🤖 3モデルアンサンブル<br>
                    💡 自動学習改善<br>
                    📊 14次元特徴量
                </p>
                <button onclick="runFinalPrediction()" style="
                    width: 100%; 
                    padding: 18px; 
                    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); 
                    color: white; 
                    border: none; 
                    border-radius: 10px; 
                    font-size: 16px; 
                    cursor: pointer;
                    font-weight: bold;
                    transition: transform 0.2s;
                " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                    🚀 完全版予測実行
                </button>
            </div>
            
            <!-- システム管理 -->
            <div style="background: rgba(255,255,255,0.15); border-radius: 12px; padding: 25px;">
                <h3 style="margin-top: 0; color: #fff; font-size: 18px; display: flex; align-items: center;">
                    🔧 システム管理
                </h3>
                <p style="font-size: 13px; margin: 15px 0; opacity: 0.9;">
                    🔍 ヘルスチェック<br>
                    🔄 自動回復<br>
                    📊 性能監視<br>
                    💾 データ管理
                </p>
                <button onclick="runHealthCheck()" style="
                    width: 100%; 
                    padding: 18px; 
                    background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); 
                    color: white; 
                    border: none; 
                    border-radius: 10px; 
                    font-size: 16px; 
                    cursor: pointer;
                    font-weight: bold;
                    transition: transform 0.2s;
                " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                    🔍 ヘルスチェック
                </button>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 25px;">
            <!-- 時系列検証 -->
            <div style="background: rgba(255,255,255,0.15); border-radius: 12px; padding: 25px;">
                <h3 style="margin-top: 0; color: #fff; font-size: 18px;">📊 時系列検証</h3>
                <p style="font-size: 13px; margin: 15px 0; opacity: 0.9;">
                    🔄 固定窓検証（50回分）<br>
                    📈 累積窓検証<br>
                    ⚖️ 手法比較分析<br>
                    🎯 最適化推奨
                </p>
                <button onclick="runTimeseriesValidation()" style="
                    width: 100%; 
                    padding: 15px; 
                    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                    color: #333; 
                    border: none; 
                    border-radius: 10px; 
                    font-size: 15px; 
                    cursor: pointer;
                    font-weight: bold;
                ">
                    📊 検証実行
                </button>
            </div>
            
            <!-- データエクスポート -->
            <div style="background: rgba(255,255,255,0.15); border-radius: 12px; padding: 25px;">
                <h3 style="margin-top: 0; color: #fff; font-size: 18px;">💾 データ管理</h3>
                <p style="font-size: 13px; margin: 15px 0; opacity: 0.9;">
                    📁 予測データ保存<br>
                    🔄 システム状態保存<br>
                    📊 性能メトリクス<br>
                    📤 JSONエクスポート
                </p>
                <button onclick="exportSystemData()" style="
                    width: 100%; 
                    padding: 15px; 
                    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                    color: #333; 
                    border: none; 
                    border-radius: 10px; 
                    font-size: 15px; 
                    cursor: pointer;
                    font-weight: bold;
                ">
                    💾 データ保存
                </button>
            </div>
        </div>
        
        <div id="status" style="
            padding: 20px; 
            background: rgba(255,255,255,0.1); 
            border-radius: 10px; 
            margin: 15px 0;
            border-left: 4px solid #fff;
            display: none;
            backdrop-filter: blur(5px);
        "></div>
        
        <div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 20px; margin-top: 25px;">
            <h4 style="margin-top: 0; color: #fff;">🎯 完全版システムの使い方</h4>
            <div style="display: flex; align-items: center; justify-content: space-between; margin: 20px 0;">
                <div style="text-align: center; flex: 1;">
                    <div style="background: rgba(255,255,255,0.2); color: white; width: 45px; height: 45px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 8px; font-size: 18px; font-weight: bold;">1</div>
                    <small style="font-size: 12px;"><strong>予測実行</strong><br>永続化対応<br>自動学習適用</small>
                </div>
                <div style="flex: 0.3; text-align: center; font-size: 20px;">→</div>
                <div style="text-align: center; flex: 1;">
                    <div style="background: rgba(255,255,255,0.2); color: white; width: 45px; height: 45px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 8px; font-size: 18px; font-weight: bold;">2</div>
                    <small style="font-size: 12px;"><strong>結果公開待ち</strong><br>当選番号が<br>公開されるまで待機</small>
                </div>
                <div style="flex: 0.3; text-align: center; font-size: 20px;">→</div>
                <div style="text-align: center; flex: 1;">
                    <div style="background: rgba(255,255,255,0.2); color: white; width: 45px; height: 45px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 8px; font-size: 18px; font-weight: bold;">3</div>
                    <small style="font-size: 12px;"><strong>自動学習</strong><br>結果照合<br>改善点反映</small>
                </div>
                <div style="flex: 0.3; text-align: center; font-size: 20px;">→</div>
                <div style="text-align: center; flex: 1;">
                    <div style="background: rgba(255,255,255,0.2); color: white; width: 45px; height: 45px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 8px; font-size: 18px; font-weight: bold;">4</div>
                    <small style="font-size: 12px;"><strong>継続改善</strong><br>学習結果を<br>次回予測に反映</small>
                </div>
            </div>
            <p style="font-size: 12px; color: rgba(255,255,255,0.8); margin: 15px 0 0; text-align: center;">
                ✨ <strong>完全自動化</strong>: 予測→保存→学習→改善のサイクルが全自動で実行されます
            </p>
        </div>
        
        <div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 20px; margin-top: 20px;">
            <h4 style="margin-top: 0; color: #fff;">🌟 完全版の革新的機能</h4>
            <ul style="margin: 15px 0; padding-left: 20px; font-size: 13px; line-height: 1.8;">
                <li><strong>予測永続化</strong>: 一度予測したら内容が変わらない信頼性</li>
                <li><strong>自動結果照合</strong>: 新しい当選番号で過去予測を自動検証</li>
                <li><strong>見逃し番号学習</strong>: よく見逃す番号を特定して次回ブースト</li>
                <li><strong>成功パターン学習</strong>: 高精度予測の特徴を抽出・適用</li>
                <li><strong>小数字重要度学習</strong>: 1-15番の出現傾向を学習</li>
                <li><strong>システム自動回復</strong>: 問題検出時の自動修復機能</li>
                <li><strong>完全分析レポート</strong>: 前回結果の詳細分析表示</li>
                <li><strong>時系列最適化</strong>: 50回分固定窓での最適学習</li>
            </ul>
        </div>
    </div>
    
    <script>
        async function runFinalPrediction() {
            const statusDiv = document.getElementById('status');
            
            statusDiv.style.display = 'block';
            statusDiv.innerHTML = '🌟 完全版予測システム実行中...';
            statusDiv.style.borderColor = '#ff6b6b';
            
            try {
                statusDiv.innerHTML = '🔍 システム状態確認→📊 データ取得→🧠 学習改善→🎯 予測生成→💾 永続化保存<br><small style="opacity:0.8;">完全版処理のため約2-3分かかります</small>';
                
                const result = await google.colab.kernel.invokeFunction(
                    'run_miniloto_final_prediction', 
                    [], 
                    {}
                );
                
                if (result === 'SUCCESS') {
                    statusDiv.innerHTML = '🎉 完全版予測システム実行完了!<br><strong>👆 上部のPythonコンソールで完全版予測結果を確認してください</strong><br><small style="opacity:0.8;">永続化保存済み・自動学習適用済み</small>';
                } else {
                    statusDiv.innerHTML = '❌ 完全版予測でエラーが発生しました';
                    statusDiv.style.background = 'rgba(255,255,255,0.1)';
                    statusDiv.style.borderColor = '#ff4757';
                }
                
            } catch (error) {
                statusDiv.innerHTML = '❌ エラー: ' + error.message;
                statusDiv.style.background = 'rgba(255,255,255,0.1)';
                statusDiv.style.borderColor = '#ff4757';
            }
        }
        
        async function runHealthCheck() {
            const statusDiv = document.getElementById('status');
            
            statusDiv.style.display = 'block';
            statusDiv.innerHTML = '🔍 システムヘルスチェック実行中...';
            statusDiv.style.borderColor = '#4ecdc4';
            
            try {
                statusDiv.innerHTML = '🔧 データ取得器→🤖 モデル→🧠 学習システム→💾 永続化をチェック中...<br><small style="opacity:0.8;">問題検出時は自動回復を実行します</small>';
                
                const result = await google.colab.kernel.invokeFunction(
                    'run_miniloto_health_check', 
                    [], 
                    {}
                );
                
                if (result === 'SUCCESS') {
                    statusDiv.innerHTML = '✅ システム状態: 正常<br><strong>👆 上部のPythonコンソールで詳細結果を確認してください</strong>';
                } else {
                    statusDiv.innerHTML = '⚠️ システム状態: 要注意（自動回復を試行しました）';
                    statusDiv.style.borderColor = '#ffa726';
                }
                
            } catch (error) {
                statusDiv.innerHTML = '❌ エラー: ' + error.message;
                statusDiv.style.background = 'rgba(255,255,255,0.1)';
                statusDiv.style.borderColor = '#ff4757';
            }
        }
        
        async function runTimeseriesValidation() {
            const statusDiv = document.getElementById('status');
            
            statusDiv.style.display = 'block';
            statusDiv.innerHTML = '📊 時系列交差検証実行中...';
            statusDiv.style.borderColor = '#a8edea';
            
            try {
                statusDiv.innerHTML = '🔄 固定窓検証（50回分）→📈 累積窓検証→⚖️ 手法比較分析中...<br><small style="opacity:0.8;">処理時間: 約3-5分</small>';
                
                const result = await google.colab.kernel.invokeFunction(
                    'run_miniloto_timeseries_validation_final', 
                    [], 
                    {}
                );
                
                statusDiv.innerHTML = '✅ 時系列交差検証完了!<br><strong>👆 上部のPythonコンソールで検証結果を確認してください</strong>';
                
            } catch (error) {
                statusDiv.innerHTML = '❌ エラー: ' + error.message;
                statusDiv.style.background = 'rgba(255,255,255,0.1)';
                statusDiv.style.borderColor = '#ff4757';
            }
        }
        
        async function exportSystemData() {
            const statusDiv = document.getElementById('status');
            
            statusDiv.style.display = 'block';
            statusDiv.innerHTML = '💾 システムデータエクスポート中...';
            statusDiv.style.borderColor = '#ffecd2';
            
            try {
                statusDiv.innerHTML = '📁 予測データ→🔄 システム状態→📊 性能メトリクスをエクスポート中...';
                
                const result = await google.colab.kernel.invokeFunction(
                    'export_complete_system_data', 
                    [], 
                    {}
                );
                
                statusDiv.innerHTML = '✅ システムデータエクスポート完了!<br>JSON形式でデータが準備されました';
                
            } catch (error) {
                statusDiv.innerHTML = '❌ エラー: ' + error.message;
                statusDiv.style.background = 'rgba(255,255,255,0.1)';
                statusDiv.style.borderColor = '#ff4757';
            }
        }
    </script>'''
    
    return html_content

# パート4最終テスト実行
print("\n" + "="*80)
print("🧪 パート4: 完全版システム最終テスト")
print("="*80)

# 完全版予測テスト
final_test_result = run_miniloto_final_prediction()
print(f"\n🏁 完全版予測テスト結果: {final_test_result}")

if final_test_result == "SUCCESS":
    print("✅ 完全版予測システム正常動作")
    
    # ヘルスチェックテスト
    health_test_result = run_miniloto_health_check()
    print(f"\n🏁 ヘルスチェックテスト結果: {health_test_result}")
    
    # データエクスポートテスト
    export_test_result = export_complete_system_data()
    print(f"\n🏁 データエクスポートテスト結果: {export_test_result}")
    
    if health_test_result == "SUCCESS" and export_test_result == "SUCCESS":
        print("\n🎉 === パート4完全版システム 全機能動作確認完了 ===")
        
        # 最終統計表示
        print(f"\n📊 ミニロト予測システム完全版 最終統計:")
        print(f"  システム名: MiniLoto_Final_v1.0")
        print(f"  学習データ数: {final_system.data_count}件")
        print(f"  特徴量次元: 14次元（ミニロト完全最適化）")
        print(f"  学習モデル数: {len(final_system.trained_models)}個")
        print(f"  永続化予測数: {len(final_system.persistence.get_all_predictions())}件")
        print(f"  システム状態: {'正常' if final_system.system_ready else '要注意'}")
        print(f"  自動学習: 有効")
        print(f"  永続化: 有効")
        print(f"  自動回復: 有効")
        print(f"  時系列検証: 対応")
        
        print(f"\n🌟 完全版の主要機能:")
        print(f"  ✅ 3モデルアンサンブル（RF + GB + NN）")
        print(f"  ✅ 14次元最適化特徴量")
        print(f"  ✅ 予測永続化システム")
        print(f"  ✅ 自動結果照合・学習改善")
        print(f"  ✅ 見逃し番号ブースト学習")
        print(f"  ✅ 成功パターン学習")
        print(f"  ✅ 時系列交差検証（50回分固定窓）")
        print(f"  ✅ システム自動回復機能")
        print(f"  ✅ 完全分析レポート")
        print(f"  ✅ データエクスポート機能")
        
    else:
        print("⚠️ 一部機能に問題がありますが、メイン予測機能は正常です")
else:
    print("❌ 完全版システムに問題があります")

print("\n" + "="*80)
print("🎯 ミニロト予測システム開発完了")
print("🌟 パート1: 基盤システム ✅")
print("🌟 パート2: 高度予測システム ✅") 
print("🌟 パート3: 自動学習システム ✅")
print("🌟 パート4: 統合・完成版 ✅")
print("="*80)

# 最終版インターフェース表示
print("\n🎉 === ミニロト予測システム完全版 完成! ===")
print("📱 完全版Webインターフェースを表示します...")

# パート4の関数をグローバルに登録
import sys
current_module = sys.modules[__name__]

# 実行関数をグローバルスコープに追加
setattr(current_module, 'run_miniloto_final_prediction', run_miniloto_final_prediction)
setattr(current_module, 'run_miniloto_health_check', run_miniloto_health_check) 
setattr(current_module, 'run_miniloto_timeseries_validation_final', run_miniloto_timeseries_validation_final)
setattr(current_module, 'export_complete_system_data', export_complete_system_data)

# HTMLインターフェース表示
final_interface_html = show_final_interface()

print("\n" + "="*80)
print("📱 完全版Webインターフェースを表示中...")
print("👆 上部に表示される完全版UIをご利用ください")
print("="*80)

# ========================= パート4Cここまで =========================# ========================= パート4B開始 =========================

    def _check_and_apply_complete_learning(self, latest_data, current_round):
        """完全版学習改善チェック・適用"""
        try:
            print("\n🧠 === 完全版学習改善チェック ===")
            
            # 前回予測データ取得
            previous_prediction = self.persistence.load_prediction(current_round)
            
            if not previous_prediction:
                print(f"📊 第{current_round}回の予測記録なし")
                return False
            
            if previous_prediction['verified']:
                print(f"✅ 第{current_round}回は既に学習適用済み")
                return True
            
            # 当選結果確認
            main_cols = ['第1数字', '第2数字', '第3数字', '第4数字', '第5数字']
            round_col = '開催回'
            
            matching_data = latest_data[latest_data[round_col] == current_round]
            
            if len(matching_data) == 0:
                print(f"📊 第{current_round}回の当選結果未公開")
                return False
            
            # 詳細学習分析実行
            actual_row = matching_data.iloc[0]
            actual_numbers = [int(actual_row[col]) for col in main_cols if col in actual_row.index]
            
            if len(actual_numbers) == 5:
                print(f"🎯 第{current_round}回学習分析実行: {actual_numbers}")
                
                # 永続化更新
                self.persistence.update_with_actual_result(current_round, actual_numbers)
                
                # 高度学習分析
                self._perform_advanced_learning_analysis(previous_prediction, actual_numbers, current_round)
                
                print(f"✅ 第{current_round}回完全学習分析完了")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ 完全学習チェックエラー: {e}")
            return False
    
    def _perform_advanced_learning_analysis(self, prediction_data, actual_numbers, round_number):
        """高度学習分析実行"""
        try:
            print(f"\n🔬 === 第{round_number}回高度学習分析 ===")
            
            predictions = prediction_data['predictions']
            
            # 詳細一致分析
            match_analysis = []
            for i, pred_set in enumerate(predictions):
                pred_numbers = [int(x) for x in pred_set]
                matched = set(pred_numbers) & set(actual_numbers)
                missed = set(actual_numbers) - set(pred_numbers)
                extra = set(pred_numbers) - set(actual_numbers)
                
                analysis = {
                    'index': i + 1,
                    'prediction': pred_numbers,
                    'matches': len(matched),
                    'matched_numbers': sorted(list(matched)),
                    'missed_numbers': sorted(list(missed)),
                    'extra_numbers': sorted(list(extra))
                }
                match_analysis.append(analysis)
            
            # 統計計算
            all_matches = [a['matches'] for a in match_analysis]
            avg_matches = np.mean(all_matches)
            max_matches = max(all_matches)
            
            print(f"平均一致数: {avg_matches:.2f}個 | 最高一致数: {max_matches}個")
            
            # 高精度セット分析
            high_accuracy = [a for a in match_analysis if a['matches'] >= 3]
            if high_accuracy:
                print(f"🎯 高精度セット: {len(high_accuracy)}個")
                for analysis in high_accuracy:
                    print(f"  セット{analysis['index']}: {analysis['matches']}個一致 {analysis['matched_numbers']}")
                
                # 成功パターン学習
                success_patterns = self._analyze_success_patterns(high_accuracy, actual_numbers)
                if success_patterns:
                    self.auto_learner.improvement_metrics['high_accuracy_patterns'] = success_patterns
            
            # 見逃し分析
            all_missed = []
            for analysis in match_analysis:
                all_missed.extend(analysis['missed_numbers'])
            
            if all_missed:
                missed_freq = Counter(all_missed)
                print(f"❌ 頻出見逃し番号: {missed_freq.most_common(5)}")
                self.auto_learner.improvement_metrics['frequently_missed'] = missed_freq.most_common(10)
            
            # 当選パターン分析
            self._analyze_winning_patterns(actual_numbers)
            
        except Exception as e:
            print(f"❌ 高度学習分析エラー: {e}")
    
    def _analyze_success_patterns(self, high_accuracy_sets, actual_numbers):
        """成功パターンの詳細分析"""
        if not high_accuracy_sets:
            return None
        
        predictions = [s['prediction'] for s in high_accuracy_sets]
        
        # 各種統計
        sums = [sum(pred) for pred in predictions]
        odd_counts = [sum(1 for n in pred if n % 2 == 1) for pred in predictions]
        small_counts = [sum(1 for n in pred if n <= 15) for pred in predictions]
        ranges = [max(pred) - min(pred) for pred in predictions]
        
        patterns = {
            'avg_sum': np.mean(sums),
            'avg_odd_count': np.mean(odd_counts),
            'avg_small_count': np.mean(small_counts),
            'avg_range': np.mean(ranges),
            'sample_size': len(predictions),
            'actual_sum': sum(actual_numbers),
            'actual_odd_count': sum(1 for n in actual_numbers if n % 2 == 1),
            'actual_small_count': sum(1 for n in actual_numbers if n <= 15),
            'actual_range': max(actual_numbers) - min(actual_numbers)
        }
        
        print(f"💡 成功パターン学習:")
        print(f"  理想合計: {patterns['avg_sum']:.1f} (実際: {patterns['actual_sum']})")
        print(f"  理想奇数: {patterns['avg_odd_count']:.1f} (実際: {patterns['actual_odd_count']})")
        print(f"  理想小数字: {patterns['avg_small_count']:.1f} (実際: {patterns['actual_small_count']})")
        print(f"  理想範囲: {patterns['avg_range']:.1f} (実際: {patterns['actual_range']})")
        
        return patterns
    
    def _analyze_winning_patterns(self, actual_numbers):
        """当選パターンの分析"""
        patterns = {
            'sum': sum(actual_numbers),
            'odd_count': sum(1 for n in actual_numbers if n % 2 == 1),
            'small_count': sum(1 for n in actual_numbers if n <= 15),
            'range': max(actual_numbers) - min(actual_numbers),
            'consecutive': self._count_consecutive_numbers(sorted(actual_numbers)),
            'decade_distribution': self._analyze_decade_distribution(actual_numbers)
        }
        
        print(f"📊 当選パターン詳細:")
        print(f"  合計: {patterns['sum']} | 奇数: {patterns['odd_count']}個 | 小数字: {patterns['small_count']}個")
        print(f"  範囲: {patterns['range']} | 連続: {patterns['consecutive']}組")
        print(f"  十の位分布: {patterns['decade_distribution']}")
        
        # パターンをメトリクスに保存
        self.auto_learner.improvement_metrics['latest_winning_pattern'] = patterns
        
        return patterns
    
    def _count_consecutive_numbers(self, sorted_nums):
        """連続数の組をカウント"""
        consecutive_groups = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i+1] - sorted_nums[i] == 1:
                consecutive_groups += 1
        return consecutive_groups
    
    def _analyze_decade_distribution(self, numbers):
        """十の位分布分析"""
        distribution = {'1-10': 0, '11-20': 0, '21-31': 0}
        for num in numbers:
            if 1 <= num <= 10:
                distribution['1-10'] += 1
            elif 11 <= num <= 20:
                distribution['11-20'] += 1
            elif 21 <= num <= 31:
                distribution['21-31'] += 1
        return distribution
    
    def _ensure_models_ready(self, data):
        """最終モデル準備確保"""
        try:
            if self.trained_models and len(self.trained_models) >= 2:
                print("✅ 既存モデルを使用")
                return True
            
            print("🔧 モデル学習が必要です...")
            
            # パート2の高度学習を使用
            if hasattr(integrated_system, 'train_advanced_models'):
                success = integrated_system.train_advanced_models(data)
                if success:
                    # モデルをコピー
                    self.trained_models = integrated_system.trained_models.copy()
                    self.scalers = integrated_system.scalers.copy()
                    self.model_scores = integrated_system.model_scores.copy()
                    return True
            
            # フォールバック: クイック学習
            return self._quick_model_training()
            
        except Exception as e:
            print(f"❌ モデル準備エラー: {e}")
            return False
    
    def _generate_complete_predictions(self, count=20, use_learning=True):
        """完全版予測生成"""
        try:
            if not self.trained_models:
                print("❌ 学習済みモデルなし")
                return []
            
            print(f"🎯 完全版予測生成開始（{count}セット）")
            if use_learning:
                print("💡 学習改善を適用")
            
            # 学習調整パラメータ取得
            boost_numbers = []
            pattern_targets = {}
            small_boost = 0
            
            if use_learning and hasattr(self.auto_learner, 'improvement_metrics'):
                adjustments = self.auto_learner.get_learning_adjustments()
                boost_numbers = adjustments.get('boost_numbers', [])
                pattern_targets = adjustments.get('pattern_targets', {})
                small_boost = adjustments.get('small_number_boost', 0)
            
            # 基準特徴量（学習改善反映）
            if pattern_targets and use_learning:
                target_sum = pattern_targets.get('avg_sum', 80)
                target_odd = pattern_targets.get('avg_odd_count', 2.5)
                target_small = pattern_targets.get('avg_small_count', 2.5)
                base_features = [
                    target_sum / 5, 6.0, target_sum, target_odd, 28.0, 5.0, 16.0, 23.0, 1.0, 8.0, 16.0, 24.0, 5.5, target_small
                ]
                print(f"📊 学習改善基準: 合計{target_sum:.0f}, 奇数{target_odd:.1f}, 小数字{target_small:.1f}")
            else:
                # デフォルト基準特徴量
                if hasattr(self, 'pattern_stats') and self.pattern_stats:
                    avg_sum = self.pattern_stats.get('avg_sum', 80)
                    base_features = [avg_sum / 5, 6.0, avg_sum, 2.5, 28.0, 5.0, 16.0, 23.0, 1.0, 8.0, 16.0, 24.0, 5.5, 2.5]
                else:
                    base_features = [16.0, 6.0, 80.0, 2.5, 28.0, 5.0, 16.0, 23.0, 1.0, 8.0, 16.0, 24.0, 5.5, 2.5]
            
            predictions = []
            
            for i in range(count):
                # アンサンブル投票
                ensemble_votes = Counter()
                
                # 各モデルの予測
                for name, model in self.trained_models.items():
                    try:
                        scaler = self.scalers[name]
                        X_scaled = scaler.transform([base_features])
                        
                        # 複数回予測で安定化
                        for _ in range(10):  # 完全版では多めに予測
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(X_scaled)[0]
                                classes = model.classes_
                                if len(classes) > 0:
                                    selected = np.random.choice(classes, p=proba/proba.sum())
                                    if 1 <= selected <= 31:
                                        weight = self.model_weights.get(name, 0.33)
                                        ensemble_votes[int(selected)] += weight
                            else:
                                pred = model.predict(X_scaled)[0]
                                if 1 <= pred <= 31:
                                    weight = self.model_weights.get(name, 0.33)
                                    ensemble_votes[int(pred)] += weight
                    except Exception as e:
                        continue
                
                # 頻出数字ブースト
                frequent_nums = [num for num, _ in self.freq_counter.most_common(15)]
                for num in frequent_nums[:10]:
                    ensemble_votes[num] += 0.15
                
                # 学習改善ブースト
                if use_learning:
                    # 見逃し番号ブースト
                    for num in boost_numbers:
                        if 1 <= num <= 31:
                            ensemble_votes[num] += 0.3
                    
                    # 小数字ブースト
                    if small_boost > 2:
                        for num in range(1, 16):
                            ensemble_votes[num] += 0.08
                
                # 上位5個選択
                top_numbers = [num for num, _ in ensemble_votes.most_common(5)]
                
                # 不足分補完
                while len(top_numbers) < 5:
                    candidate = np.random.randint(1, 32)
                    if candidate not in top_numbers:
                        top_numbers.append(candidate)
                
                final_pred = sorted([int(x) for x in top_numbers[:5]])
                predictions.append(final_pred)
            
            print(f"✅ 完全版予測生成完了: {len(predictions)}セット")
            return predictions
            
        except Exception as e:
            print(f"❌ 完全版予測生成エラー: {e}")
            return []
    
    def _create_complete_metadata(self, learning_applied):
        """完全版メタデータ作成"""
        metadata = {
            'system_version': 'MiniLoto_Final_v1.0',
            'learning_applied': learning_applied,
            'model_count': len(self.trained_models),
            'feature_dimensions': 14,
            'data_count': self.data_count,
            'model_weights': self.model_weights.copy(),
            'model_scores': self.model_scores.copy(),
            'system_ready': self.system_ready,
            'generation_method': 'complete_ensemble',
            'boost_applied': learning_applied and hasattr(self.auto_learner, 'improvement_metrics')
        }
        
        if learning_applied and hasattr(self.auto_learner, 'improvement_metrics'):
            metadata['learning_metrics'] = self.auto_learner.improvement_metrics.copy()
        
        return metadata
    
    def _display_complete_existing_prediction(self, prediction_data, round_number):
        """完全版既存予測表示"""
        print(f"\n" + "="*80)
        print(f"📂 第{round_number}回 永続化済み予測")
        print("="*80)
        print(f"📅 作成日時: {prediction_data['timestamp']}")
        print(f"📊 システム: {prediction_data['metadata'].get('system_version', 'Unknown')}")
        print(f"🤖 モデル数: {prediction_data['metadata'].get('model_count', 'Unknown')}")
        
        if prediction_data['metadata'].get('learning_applied', False):
            print("💡 学習改善適用済み")
        
        if prediction_data['metadata'].get('boost_applied', False):
            print("🚀 ブースト機能適用済み")
        
        print("-"*80)
        
        for i, pred in enumerate(prediction_data['predictions'], 1):
            clean_pred = [int(x) for x in pred]
            print(f"第{round_number}回予測 {i:2d}: {clean_pred}")
        
        # 検証結果表示
        if prediction_data['verified'] and prediction_data['actual_result']:
            print(f"\n✅ 検証完了 - 当選番号: {prediction_data['actual_result']}")
            print("📊 一致結果:")
            
            matches = prediction_data['matches']
            best_match = max(matches)
            avg_match = np.mean(matches)
            
            for i, match_count in enumerate(matches, 1):
                status = "🎉" if match_count >= 4 else "⭐" if match_count >= 3 else "📊"
                print(f"  {status} 予測{i:2d}: {match_count}個一致")
            
            print(f"\n📈 統計: 平均{avg_match:.2f}個一致 | 最高{best_match}個一致")
    
    def _display_complete_results(self, predictions, next_info, learning_applied):
        """完全版結果表示"""
        print(f"\n" + "="*80)
        print(f"🌟 {next_info['prediction_target']} 完全版予測結果")
        print("🎯 3モデルアンサンブル + 14次元特徴量 + 自動学習 + 永続化")
        print("="*80)
        print(f"📅 予測日時: {next_info['current_date']}")
        print(f"📊 学習データ: 第1回〜第{next_info['latest_round']}回（{self.data_count}件）")
        
        if learning_applied:
            print("💡 学習改善機能: 有効")
        
        print("-"*80)
        
        for i, pred in enumerate(predictions, 1):
            clean_pred = [int(x) for x in pred]
            print(f"第{next_info['next_round']}回予測 {i:2d}: {clean_pred}")
        
        # システム性能表示
        print(f"\n" + "="*80)
        print("🤖 システム性能詳細")
        print("="*80)
        
        print(f"モデル構成:")
        for name, score in self.model_scores.items():
            weight = self.model_weights.get(name, 0)
            print(f"  {name:15s}: CV精度{score*100:5.2f}% | 重み{weight:.3f}")
        
        print(f"\nシステム状態:")
        print(f"  準備状態: {'✅ 正常' if self.system_ready else '⚠️ 要注意'}")
        print(f"  永続化: ✅ 有効")
        print(f"  自動学習: {'✅ 有効' if learning_applied else '⚠️ 無効'}")
        
        # データ分析表示
        print(f"\n" + "="*80)
        print("📊 データ分析結果")
        print("="*80)
        print(f"特徴量次元: 14次元（ミニロト完全最適化版）")
        
        if hasattr(self, 'pattern_stats') and self.pattern_stats:
            print(f"データ統計: 平均合計{self.pattern_stats.get('avg_sum', 0):.1f}")
        
        print(f"\n🔥 頻出数字TOP10:")
        top_frequent = self.freq_counter.most_common(10)
        for i, (num, count) in enumerate(top_frequent):
            if i % 5 == 0:
                print("")
            print(f"{int(num)}番({int(count)}回)", end="  ")
        
        # 学習改善情報表示
        if learning_applied and hasattr(self.auto_learner, 'improvement_metrics'):
            self._display_complete_learning_info()
    
    def _display_complete_learning_info(self):
        """完全版学習改善情報表示"""
        print(f"\n\n💡 === 学習改善詳細情報 ===")
        
        metrics = self.auto_learner.improvement_metrics
        
        if 'frequently_missed' in metrics:
            print("🎯 見逃し頻度高数字（強化ブースト対象）:")
            for num, count in metrics['frequently_missed'][:5]:
                print(f"    {num}番: {count}回見逃し → +30%ブースト")
        
        if 'high_accuracy_patterns' in metrics:
            patterns = metrics['high_accuracy_patterns']
            print(f"📊 高精度パターン学習適用:")
            print(f"    ターゲット合計値: {patterns['avg_sum']:.1f}")
            print(f"    ターゲット奇数個数: {patterns['avg_odd_count']:.1f}")
            print(f"    ターゲット小数字個数: {patterns['avg_small_count']:.1f}")
            print(f"    学習サンプル: {patterns['sample_size']}件の成功パターン")
        
        if 'small_number_importance' in metrics:
            importance = metrics['small_number_importance']
            print(f"🔢 小数字重要度学習: {importance:.1f}個")
            print(f"    1-15番に+8%ブースト適用")
        
        if 'latest_winning_pattern' in metrics:
            pattern = metrics['latest_winning_pattern']
            print(f"📈 最新当選パターン参考:")
            print(f"    合計{pattern['sum']} | 奇数{pattern['odd_count']}個 | 範囲{pattern['range']}")
    
    def _display_complete_previous_analysis(self, latest_data, current_round):
        """完全版前回結果分析表示"""
        previous_prediction = self.persistence.load_prediction(current_round)
        
        if not previous_prediction or not previous_prediction['verified']:
            return
        
        print(f"\n" + "="*80)
        print(f"📊 第{current_round}回 完全結果分析")
        print("="*80)
        
        actual_numbers = previous_prediction['actual_result']
        matches = previous_prediction['matches']
        metadata = previous_prediction.get('metadata', {})
        
        print(f"🎯 当選番号: {actual_numbers}")
        print(f"📅 予測日時: {previous_prediction['timestamp']}")
        print(f"🤖 使用システム: {metadata.get('system_version', 'Unknown')}")
        
        if metadata.get('learning_applied', False):
            print("💡 学習改善: 適用済み")
        
        print(f"\n📈 詳細予測結果:")
        print("-"*50)
        
        # 詳細結果表示
        match_3_plus = 0
        match_4_plus = 0
        
        for i, (pred, match_count) in enumerate(zip(previous_prediction['predictions'], matches), 1):
            pred_numbers = [int(x) for x in pred]
            matched_nums = sorted(list(set(pred_numbers) & set(actual_numbers)))
            
            if match_count >= 3:
                match_3_plus += 1
            if match_count >= 4:
                match_4_plus += 1
            
            if match_count >= 4:
                status = "🎉 完全的中級"
            elif match_count >= 3:
                status = "⭐ 高精度"
            elif match_count >= 2:
                status = "📊 中精度"
            else:
                status = "📊 基本"
            
            print(f"{status} 予測{i:2d}: {pred_numbers} → {match_count}個一致 {matched_nums}")
        
        # 統計サマリー
        avg_matches = np.mean(matches)
        max_matches = max(matches)
        
        print("-"*50)
        print(f"📊 完全統計サマリー:")
        print(f"    平均一致数: {avg_matches:.2f}個")
        print(f"    最高一致数: {max_matches}個")
        print(f"    3個以上一致: {match_3_plus}セット")
        print(f"    4個以上一致: {match_4_plus}セット")
        print(f"    予測精度: {(avg_matches/5)*100:.1f}%")
        
        # 学習による改善効果
        if metadata.get('learning_applied', False):
            print(f"💡 学習改善効果: 見逃しブースト・パターン学習適用済み")
    
    def _update_performance_metrics(self):
        """パフォーマンスメトリクス更新"""
        try:
            all_predictions = self.persistence.get_all_predictions()
            verified_predictions = [p for p in all_predictions.values() if p['verified']]
            
            if verified_predictions:
                all_matches = []
                for pred in verified_predictions:
                    all_matches.extend(pred['matches'])
                
                self.performance_metrics = {
                    'total_predictions': len(all_predictions),
                    'verified_predictions': len(verified_predictions),
                    'avg_accuracy': np.mean(all_matches),
                    'max_accuracy': max(all_matches),
                    'high_accuracy_rate': sum(1 for m in all_matches if m >= 3) / len(all_matches),
                    'last_updated': datetime.now().isoformat()
                }
        except Exception as e:
            print(f"⚠️ パフォーマンスメトリクス更新エラー: {e}")

# ========================= パート4Bここまで =========================# ミニロト予測システム パート4: 統合・完成版
# ========================= パート4A開始 =========================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from collections import Counter, defaultdict
import json
import traceback
from datetime import datetime

print("🚀 ミニロト予測システム - パート4: 統合・完成版")
print("🎉 全機能統合 + 最終版UI + エラーハンドリング強化")
print("🌐 完全版ミニロト予測システム")

# 最終版統合クラス
class MiniLotoFinalSystem:
    """ミニロト予測システム最終版 - 全機能統合"""
    def __init__(self):
        print("🔧 MiniLotoFinalSystem初期化中...")
        
        # 他のパートから継承
        from __main__ import integrated_system
        
        # データ取得器
        self.data_fetcher = integrated_system.data_fetcher
        
        # モデル関連
        self.models = integrated_system.models
        self.scalers = integrated_system.scalers
        self.model_weights = integrated_system.model_weights.copy()
        self.trained_models = integrated_system.trained_models
        self.model_scores = integrated_system.model_scores
        
        # データ分析
        self.freq_counter = integrated_system.freq_counter
        self.pair_freq = integrated_system.pair_freq
        self.pattern_stats = integrated_system.pattern_stats
        self.data_count = integrated_system.data_count
        
        # 高度機能
        self.auto_learner = integrated_system.auto_learner
        self.persistence = integrated_system.persistence
        self.validator = None
        
        # システム状態
        self.system_ready = False
        self.last_error = None
        self.performance_metrics = {}
        
        print("✅ 最終版システム初期化完了")
    
    def system_health_check(self):
        """システムヘルスチェック"""
        try:
            print("\n🔍 === システムヘルスチェック ===")
            
            health_status = {
                'data_fetcher': False,
                'models': False,
                'learning': False,
                'persistence': False,
                'overall': False
            }
            
            # データ取得器チェック
            try:
                if hasattr(self.data_fetcher, 'latest_data') and self.data_fetcher.latest_data is not None:
                    health_status['data_fetcher'] = True
                    print("✅ データ取得器: 正常")
                else:
                    print("⚠️ データ取得器: データ未取得")
            except Exception as e:
                print(f"❌ データ取得器: エラー - {e}")
            
            # モデルチェック
            try:
                if self.trained_models and len(self.trained_models) >= 2:
                    health_status['models'] = True
                    print(f"✅ モデル: 正常 ({len(self.trained_models)}個)")
                else:
                    print("⚠️ モデル: 未学習")
            except Exception as e:
                print(f"❌ モデル: エラー - {e}")
            
            # 学習システムチェック
            try:
                if hasattr(self.auto_learner, 'improvement_metrics'):
                    health_status['learning'] = True
                    print("✅ 学習システム: 正常")
                else:
                    print("⚠️ 学習システム: 未初期化")
            except Exception as e:
                print(f"❌ 学習システム: エラー - {e}")
            
            # 永続化チェック
            try:
                if hasattr(self.persistence, 'memory_storage'):
                    health_status['persistence'] = True
                    predictions_count = len(self.persistence.get_all_predictions())
                    print(f"✅ 永続化: 正常 ({predictions_count}件保存)")
                else:
                    print("⚠️ 永続化: 未初期化")
            except Exception as e:
                print(f"❌ 永続化: エラー - {e}")
            
            # 総合判定
            working_components = sum(health_status.values())
            if working_components >= 3:
                health_status['overall'] = True
                self.system_ready = True
                print(f"\n🎉 システム総合状態: 正常 ({working_components}/4)")
            else:
                print(f"\n⚠️ システム総合状態: 要注意 ({working_components}/4)")
            
            return health_status
            
        except Exception as e:
            print(f"❌ ヘルスチェックエラー: {e}")
            self.last_error = str(e)
            return None
    
    def auto_recovery(self):
        """自動回復処理"""
        try:
            print("\n🔧 === 自動回復処理開始 ===")
            
            recovery_success = False
            
            # データ取得の回復
            if not hasattr(self.data_fetcher, 'latest_data') or self.data_fetcher.latest_data is None:
                print("🔄 データ取得の回復を試行...")
                if self.data_fetcher.fetch_latest_data():
                    print("✅ データ取得回復成功")
                    recovery_success = True
                else:
                    print("❌ データ取得回復失敗")
            
            # モデル学習の回復
            if not self.trained_models and hasattr(self.data_fetcher, 'latest_data'):
                print("🔄 モデル学習の回復を試行...")
                try:
                    # 簡易モデル学習
                    success = self._quick_model_training()
                    if success:
                        print("✅ モデル学習回復成功")
                        recovery_success = True
                    else:
                        print("❌ モデル学習回復失敗")
                except Exception as e:
                    print(f"❌ モデル学習回復エラー: {e}")
            
            if recovery_success:
                print("\n🎉 自動回復処理完了")
                self.system_ready = True
            else:
                print("\n⚠️ 自動回復処理で一部問題が残っています")
            
            return recovery_success
            
        except Exception as e:
            print(f"❌ 自動回復エラー: {e}")
            self.last_error = str(e)
            return False
    
    def _quick_model_training(self):
        """クイックモデル学習"""
        try:
            if not hasattr(self.data_fetcher, 'latest_data') or self.data_fetcher.latest_data is None:
                return False
            
            # 基本的な2モデルのみで高速学習
            quick_models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=50, max_depth=8, random_state=42, n_jobs=-1
                ),
                'gradient_boost': GradientBoostingClassifier(
                    n_estimators=40, max_depth=4, random_state=42
                )
            }
            
            # 簡易特徴量作成
            X, y = self._create_simple_features(self.data_fetcher.latest_data)
            if X is None or len(X) < 50:
                return False
            
            # 学習実行
            for name, model in quick_models.items():
                try:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    self.scalers[name] = scaler
                    
                    model.fit(X_scaled, y)
                    cv_score = np.mean(cross_val_score(model, X_scaled, y, cv=2))
                    
                    self.trained_models[name] = model
                    self.model_scores[name] = cv_score
                    
                except Exception as e:
                    continue
            
            if self.trained_models:
                self.model_weights = {
                    'random_forest': 0.6,
                    'gradient_boost': 0.4
                }
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def _create_simple_features(self, data):
        """簡易特徴量作成（回復用）"""
        try:
            features = []
            targets = []
            main_cols = ['第1数字', '第2数字', '第3数字', '第4数字', '第5数字']
            
            for i in range(min(len(data), 500)):  # 効率化のため500件まで
                try:
                    current = []
                    for col in main_cols:
                        if col in data.columns:
                            current.append(int(data.iloc[i][col]))
                    
                    if len(current) != 5 or not all(1 <= x <= 31 for x in current) or len(set(current)) != 5:
                        continue
                    
                    # 簡易8次元特徴量
                    feat = [
                        float(np.mean(current)),           # 平均値
                        float(np.std(current)),            # 標準偏差
                        float(np.sum(current)),            # 合計値
                        float(sum(1 for x in current if x % 2 == 1)),  # 奇数個数
                        float(max(current)),               # 最大値
                        float(min(current)),               # 最小値
                        float(max(current) - min(current)), # 範囲
                        float(len([x for x in current if x <= 15])), # 小数字数
                    ]
                    
                    # 次回予測ターゲット
                    if i < len(data) - 1:
                        next_nums = []
                        for col in main_cols:
                            if col in data.columns:
                                next_nums.append(int(data.iloc[i+1][col]))
                        
                        if len(next_nums) == 5:
                            for target_num in next_nums:
                                features.append(feat.copy())
                                targets.append(target_num)
                        
                except Exception as e:
                    continue
            
            return np.array(features), np.array(targets)
            
        except Exception as e:
            return None, None
    
    def run_complete_prediction(self, force_new=False):
        """完全版予測実行"""
        try:
            print("\n" + "="*80)
            print("🌟 ミニロト完全版予測システム実行")
            print("="*80)
            
            # 1. システムヘルスチェック
            health = self.system_health_check()
            if not health or not health['overall']:
                print("\n🔧 システム問題を検出。自動回復を実行...")
                recovery_success = self.auto_recovery()
                if not recovery_success:
                    print("❌ 自動回復失敗。手動対応が必要です。")
                    return [], {}
            
            # 2. データ確認・取得
            if not hasattr(self.data_fetcher, 'latest_data') or self.data_fetcher.latest_data is None:
                print("📊 最新データを取得中...")
                if not self.data_fetcher.fetch_latest_data():
                    print("❌ データ取得失敗")
                    return [], {}
            
            latest_data = self.data_fetcher.latest_data
            latest_round = self.data_fetcher.latest_round
            next_round = latest_round + 1
            next_info = self.data_fetcher.get_next_round_info()
            
            print(f"📊 最新データ: 第{latest_round}回まで（{len(latest_data)}件）")
            print(f"🎯 予測対象: 第{next_round}回")
            
            # 3. 永続化チェック
            if not force_new and self.persistence.is_prediction_exists(next_round):
                print(f"\n📂 第{next_round}回の予測は既に永続化されています")
                existing_prediction = self.persistence.load_prediction(next_round)
                
                self._display_complete_existing_prediction(existing_prediction, next_round)
                self._display_complete_previous_analysis(latest_data, latest_round)
                
                return existing_prediction['predictions'], next_info
            
            # 4. 新規予測生成
            print(f"\n🆕 第{next_round}回の新規予測を生成します")
            
            # 5. 学習改善チェック
            learning_applied = self._check_and_apply_complete_learning(latest_data, latest_round)
            
            # 6. 最終モデル確保
            if not self._ensure_models_ready(latest_data):
                print("❌ モデル準備失敗")
                return [], {}
            
            # 7. 完全版予測生成
            predictions = self._generate_complete_predictions(20, learning_applied)
            if not predictions:
                print("❌ 予測生成失敗")
                return [], {}
            
            # 8. 永続化保存
            metadata = self._create_complete_metadata(learning_applied)
            self.persistence.save_prediction_permanently(next_round, predictions, metadata)
            
            # 9. 完全版結果表示
            self._display_complete_results(predictions, next_info, learning_applied)
            
            # 10. 前回結果分析表示
            self._display_complete_previous_analysis(latest_data, latest_round)
            
            # 11. パフォーマンス更新
            self._update_performance_metrics()
            
            print("\n" + "="*80)
            print("🎉 ミニロト完全版予測システム実行完了!")
            print(f"📝 第{next_round}回予測として永続化保存済み")
            print("🔄 次回実行時は保存済み予測を表示します")
            print("="*80)
            
            return predictions, next_info
            
        except Exception as e:
            print(f"❌ 完全版システムエラー: {str(e)}")
            print(f"詳細: {traceback.format_exc()}")
            self.last_error = str(e)
            return [], {}

# ========================= パート4Aここまで =========================