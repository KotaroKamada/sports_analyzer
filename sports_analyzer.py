import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import streamlit as st
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# ページ設定
st.set_page_config(
    page_title="コンディション管理アプリ",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SportsDataAnalyzer:
    def __init__(self):
        self.data = None
        self.players = []
        self.dates = []
        self.training_times = []
        
    def load_excel_data(self, uploaded_file):
        """エクセルファイルを読み込みデータを構造化（外傷・障害データも含む）"""
        try:
            # エクセルファイルを読み込み
            df = pd.read_excel(uploaded_file, header=None, sheet_name=None)
            
            # Sheet1の処理（既存のデータ）
            sheet1 = df.get('Sheet1', df[list(df.keys())[0]])  # 最初のシートを取得
            
            # 日付データを取得（1行目、C列以降）
            date_row = sheet1.iloc[0, 2:]  # 1行目のC列以降
            self.dates = pd.to_datetime(date_row.dropna()).tolist()
            
            # チーム平均データ（2-4行目）※活動時間が個別化されたため、チーム平均の練習時間は削除
            team_avg = {
                'sleep': sheet1.iloc[1, 2:2+len(self.dates)].tolist(),
                'weight': sheet1.iloc[2, 2:2+len(self.dates)].tolist(),
                'rpe': sheet1.iloc[3, 2:2+len(self.dates)].tolist()
            }
            
            # 選手データを抽出（各選手4行構成：睡眠時間、体重、RPE、活動時間）
            players_data = {}
            row = 4  # A5から開始（0ベースなので4）
            
            while row < len(sheet1):
                if pd.notna(sheet1.iloc[row, 0]) and sheet1.iloc[row, 0] != '':
                    player_name = sheet1.iloc[row, 0]
                    
                    # 選手の4行分のデータを取得
                    sleep_data = sheet1.iloc[row, 2:2+len(self.dates)].tolist()
                    weight_data = sheet1.iloc[row+1, 2:2+len(self.dates)].tolist()
                    rpe_data = sheet1.iloc[row+2, 2:2+len(self.dates)].tolist()
                    activity_time_data = sheet1.iloc[row+3, 2:2+len(self.dates)].tolist()
                    
                    players_data[player_name] = {
                        'sleep': sleep_data,
                        'weight': weight_data,
                        'rpe': rpe_data,
                        'activity_time': activity_time_data  # 個別の活動時間を追加
                    }
                    
                    row += 4  # 次の選手へ（4行構成）
                else:
                    row += 1
            
            # Sheet2の処理（外傷・障害データ）- より詳細な読み込み処理
            injury_data = {}
            injury_sheet_found = False
            
            # 外傷・障害シートを探す（「外傷・障害」を最優先に）
            injury_sheet_names = ['外傷・障害', '外傷障害', 'Sheet2', 'injury', 'Injury']
            injury_sheet = None
            
            for sheet_name in injury_sheet_names:
                if sheet_name in df:
                    injury_sheet = df[sheet_name]
                    injury_sheet_found = True
                    print(f"外傷・障害シートを発見: {sheet_name}")
                    break
            
            if injury_sheet is not None:
                print(f"外傷・障害シートの形状: {injury_sheet.shape}")
                
                # 1行目の日付データを確認（C列以降、またはB列以降）
                # まずはB列以降をチェック
                date_start_col = 1  # B列から開始
                if pd.notna(injury_sheet.iloc[0, 1]):
                    # B列に日付がある場合
                    injury_dates = injury_sheet.iloc[0, 1:1+len(self.dates)].tolist()
                elif pd.notna(injury_sheet.iloc[0, 2]):
                    # C列に日付がある場合
                    date_start_col = 2
                    injury_dates = injury_sheet.iloc[0, 2:2+len(self.dates)].tolist()
                
                print(f"外傷・障害シートの日付開始列: {date_start_col}")
                
                # 2行目以降の選手データを読み込み
                for row_idx in range(1, len(injury_sheet)):
                    if pd.notna(injury_sheet.iloc[row_idx, 0]) and injury_sheet.iloc[row_idx, 0] != '':
                        player_name = str(injury_sheet.iloc[row_idx, 0]).strip()
                        
                        # 日付列から外傷・障害データを取得
                        injury_counts = injury_sheet.iloc[row_idx, date_start_col:date_start_col+len(self.dates)].tolist()
                        
                        # NaNや空文字を0に置換（より厳密な処理）
                        processed_injuries = []
                        for i, x in enumerate(injury_counts):
                            if pd.notna(x) and x != '' and x is not None:
                                try:
                                    # 文字列の場合は数値に変換
                                    if isinstance(x, str):
                                        x = x.strip()
                                        if x == '' or x.lower() in ['none', 'nan', 'null']:
                                            processed_injuries.append(0)
                                            continue
                                    
                                    val = float(x)
                                    if val != val:  # NaNチェック
                                        processed_injuries.append(0)
                                    else:
                                        processed_injuries.append(int(max(0, val)))  # 負の値は0にする
                                except (ValueError, TypeError):
                                    processed_injuries.append(0)
                            else:
                                processed_injuries.append(0)
                        
                        # 足りない日付分は0で埋める
                        while len(processed_injuries) < len(self.dates):
                            processed_injuries.append(0)
                        
                        injury_data[player_name] = processed_injuries
                        
                        # デバッグ用：外傷・障害データの読み込み確認
                        injury_count = sum([x for x in processed_injuries if x > 0])
                        if injury_count > 0:
                            print(f"外傷・障害データ読み込み: {player_name} - 総件数: {injury_count}")
                            # 発生日も表示
                            injury_days = [i for i, x in enumerate(processed_injuries) if x > 0]
                            print(f"  発生日インデックス: {injury_days[:5]}...")  # 最初の5つだけ表示
            
            else:
                print("外傷・障害シートが見つかりませんでした。利用可能なシート:", list(df.keys()))
            
            # 選手データに外傷・障害データを追加
            for player in players_data:
                if player in injury_data:
                    players_data[player]['injuries'] = injury_data[player]
                    # デバッグ用
                    total_injuries = sum([x for x in injury_data[player] if x > 0])
                    if total_injuries > 0:
                        print(f"選手データに外傷・障害追加: {player} - 総件数: {total_injuries}")
                else:
                    # 外傷・障害データがない場合は0で埋める
                    players_data[player]['injuries'] = [0] * len(self.dates)
            
            # Sheet3の処理（目標体重データ）
            target_weight_data = {}
            target_sheet = None
            
            # シート名「目標体重」または「Sheet3」を検索
            if '目標体重' in df:
                target_sheet = df['目標体重']
            elif 'Sheet3' in df:
                target_sheet = df['Sheet3']
            
            if target_sheet is not None:
                # 目標体重シートの1行目は日付、2行目以降は選手データ
                for row_idx in range(1, len(target_sheet)):
                    if pd.notna(target_sheet.iloc[row_idx, 0]):
                        player_name = target_sheet.iloc[row_idx, 0]
                        # 日付列（1列目以降）から目標体重データを取得
                        target_weights = target_sheet.iloc[row_idx, 1:1+len(self.dates)].tolist()
                        # NaNや空文字は前の値で埋める（目標体重は継続するため）
                        processed_targets = []
                        last_valid = None
                        for weight in target_weights:
                            if pd.notna(weight) and weight != '':
                                try:
                                    last_valid = float(weight)
                                    processed_targets.append(last_valid)
                                except (ValueError, TypeError):
                                    processed_targets.append(last_valid)
                            else:
                                processed_targets.append(last_valid)
                        target_weight_data[player_name] = processed_targets
            
            # 選手データに目標体重データを追加
            for player in players_data:
                if player in target_weight_data:
                    players_data[player]['target_weight'] = target_weight_data[player]
                else:
                    # 目標体重データがない場合はNoneで埋める
                    players_data[player]['target_weight'] = [None] * len(self.dates)
            
            # デバッグ情報を追加
            if target_weight_data:
                print(f"目標体重データを読み込みました: {len(target_weight_data)}選手分")
                for player, weights in target_weight_data.items():
                    valid_weights = [w for w in weights if w is not None]
                    if valid_weights:
                        print(f"  {player}: {len(valid_weights)}日分の目標体重データ")
            else:
                print("目標体重データが見つかりませんでした")
            
            # 外傷・障害データの読み込み結果をサマリー表示
            if injury_data:
                total_injury_cases = sum([sum([x for x in injuries if x > 0]) for injuries in injury_data.values()])
                print(f"外傷・障害データ読み込み完了: {len(injury_data)}選手、総件数: {total_injury_cases}")
            else:
                print("外傷・障害データが見つかりませんでした（Sheet2を確認してください）")
            
            self.data = {
                'team_avg': team_avg,
                'players': players_data
            }
            
            self.players = list(players_data.keys())
            
            # sRPE計算（個別の活動時間を使用）
            self.calculate_srpe()
            
            # ACWR計算
            self.calculate_acwr()
            
            # Zスコア計算
            self.calculate_zscores()
            
            return True
            
        except Exception as e:
            st.error(f"ファイル読み込みエラー: {str(e)}")
            return False
    
    def calculate_srpe(self):
        """sRPE（session RPE）を計算（個別の活動時間を使用）"""
        for player in self.players:
            player_data = self.data['players'][player]
            rpe_values = player_data['rpe']
            activity_time_values = player_data['activity_time']
            srpe_values = []
            
            for i, (rpe, activity_time) in enumerate(zip(rpe_values, activity_time_values)):
                if pd.notna(rpe) and pd.notna(activity_time) and activity_time != '' and rpe != '':
                    try:
                        srpe = float(rpe) * float(activity_time)
                        srpe_values.append(srpe)
                    except (ValueError, TypeError):
                        srpe_values.append(np.nan)
                else:
                    srpe_values.append(np.nan)
            
            self.data['players'][player]['srpe'] = srpe_values
    
    def calculate_acwr(self):
        """Acute:Chronic Workload Ratio (ACWR)を計算"""
        for player in self.players:
            srpe_values = self.data['players'][player]['srpe']
            chronic_rpe = []  # 28日間の平均
            acute_rpe = []    # 7日間の平均
            acwr = []
            
            for i in range(len(srpe_values)):
                # Chronic RPE (28日間)
                start_chronic = max(0, i - 27)
                chronic_data = srpe_values[start_chronic:i+1]
                chronic_data = [x for x in chronic_data if pd.notna(x)]
                chronic_avg = np.mean(chronic_data) if chronic_data else np.nan
                chronic_rpe.append(chronic_avg)
                
                # Acute RPE (7日間)
                start_acute = max(0, i - 6)
                acute_data = srpe_values[start_acute:i+1]
                acute_data = [x for x in acute_data if pd.notna(x)]
                acute_avg = np.mean(acute_data) if acute_data else np.nan
                acute_rpe.append(acute_avg)
                
                # ACWR
                if pd.notna(chronic_avg) and chronic_avg != 0:
                    acwr_value = acute_avg / chronic_avg
                    acwr.append(acwr_value)
                else:
                    acwr.append(np.nan)
            
            self.data['players'][player]['chronic_rpe'] = chronic_rpe
            self.data['players'][player]['acute_rpe'] = acute_rpe
            self.data['players'][player]['acwr'] = acwr
    
    def calculate_zscores(self):
        """各指標のZスコアを計算"""
        # 全選手の有効データを収集
        all_sleep = []
        all_weight = []
        all_srpe = []
        all_acwr = []
        
        for player in self.players:
            player_data = self.data['players'][player]
            
            # 有効な値のみ抽出
            sleep_values = [x for x in player_data.get('sleep', []) if pd.notna(x)]
            weight_values = [x for x in player_data.get('weight', []) if pd.notna(x)]
            srpe_values = [x for x in player_data.get('srpe', []) if pd.notna(x)]
            acwr_values = [x for x in player_data.get('acwr', []) if pd.notna(x)]
            
            all_sleep.extend(sleep_values)
            all_weight.extend(weight_values)
            all_srpe.extend(srpe_values)
            all_acwr.extend(acwr_values)
        
        # チーム全体の統計値を計算
        self.team_stats = {
            'sleep': {'mean': np.mean(all_sleep) if all_sleep else 0, 'std': np.std(all_sleep) if len(all_sleep) > 1 else 1},
            'weight': {'mean': np.mean(all_weight) if all_weight else 0, 'std': np.std(all_weight) if len(all_weight) > 1 else 1},
            'srpe': {'mean': np.mean(all_srpe) if all_srpe else 0, 'std': np.std(all_srpe) if len(all_srpe) > 1 else 1},
            'acwr': {'mean': np.mean(all_acwr) if all_acwr else 0, 'std': np.std(all_acwr) if len(all_acwr) > 1 else 1}
        }
        
        # 各選手のZスコアを計算
        for player in self.players:
            player_data = self.data['players'][player]
            
            # 睡眠時間のZスコア
            sleep_zscores = []
            for val in player_data.get('sleep', []):
                if pd.notna(val):
                    z = (val - self.team_stats['sleep']['mean']) / self.team_stats['sleep']['std']
                    sleep_zscores.append(z)
                else:
                    sleep_zscores.append(np.nan)
            
            # 体重のZスコア
            weight_zscores = []
            for val in player_data.get('weight', []):
                if pd.notna(val):
                    z = (val - self.team_stats['weight']['mean']) / self.team_stats['weight']['std']
                    weight_zscores.append(z)
                else:
                    weight_zscores.append(np.nan)
            
            # sRPEのZスコア
            srpe_zscores = []
            for val in player_data.get('srpe', []):
                if pd.notna(val):
                    z = (val - self.team_stats['srpe']['mean']) / self.team_stats['srpe']['std']
                    srpe_zscores.append(z)
                else:
                    srpe_zscores.append(np.nan)
            
            # ACWRのZスコア
            acwr_zscores = []
            for val in player_data.get('acwr', []):
                if pd.notna(val):
                    z = (val - self.team_stats['acwr']['mean']) / self.team_stats['acwr']['std']
                    acwr_zscores.append(z)
                else:
                    acwr_zscores.append(np.nan)
            
            # Zスコアをデータに追加
            self.data['players'][player]['sleep_zscore'] = sleep_zscores
            self.data['players'][player]['weight_zscore'] = weight_zscores
            self.data['players'][player]['srpe_zscore'] = srpe_zscores
            self.data['players'][player]['acwr_zscore'] = acwr_zscores

def create_improved_trend_charts_with_all_zscores(dates, sleep_data, weight_data, rpe_data, 
                                                 srpe_data, acwr_data, player_name, selected_date,
                                                 sleep_zscore, weight_zscore, srpe_zscore, acwr_zscore, injury_data=None, target_weight_data=None):
    """全てのZスコアを含む改良された推移チャート（外傷・障害データ表示強化版）"""
    
    # データの型安全性を確保（改良版）
    def safe_data(data_list):
        """データリストを安全に変換（空欄や無効値はNoneに変換）"""
        result = []
        for x in data_list:
            if pd.notna(x) and x != '' and x is not None:
                try:
                    # 文字列の場合は数値に変換を試行
                    if isinstance(x, str):
                        x = x.strip()  # 前後の空白を除去
                        if x == '':
                            result.append(None)
                            continue
                    float_val = float(x)
                    # 極端な値や無効な値をチェック
                    if np.isfinite(float_val):
                        result.append(float_val)
                    else:
                        result.append(None)
                except (ValueError, TypeError):
                    result.append(None)
            else:
                result.append(None)
        return result
    
    # 安全にデータを変換
    sleep_data = safe_data(sleep_data)
    weight_data = safe_data(weight_data)
    rpe_data = safe_data(rpe_data)
    srpe_data = safe_data(srpe_data)
    acwr_data = safe_data(acwr_data)
    
    sleep_zscore = safe_data(sleep_zscore) if sleep_zscore else [None] * len(dates)
    weight_zscore = safe_data(weight_zscore) if weight_zscore else [None] * len(dates)
    srpe_zscore = safe_data(srpe_zscore) if srpe_zscore else [None] * len(dates)
    acwr_zscore = safe_data(acwr_zscore) if acwr_zscore else [None] * len(dates)
    
    # 目標体重データの処理
    target_weight_data = safe_data(target_weight_data) if target_weight_data else [None] * len(dates)
    
    # 外傷・障害データの処理（強化版）
    processed_injury_data = []
    if injury_data:
        for x in injury_data:
            if pd.notna(x) and x != '' and x is not None:
                try:
                    val = int(float(x))
                    processed_injury_data.append(max(0, val))
                except (ValueError, TypeError):
                    processed_injury_data.append(0)
            else:
                processed_injury_data.append(0)
    else:
        processed_injury_data = [0] * len(dates)
    
    # 外傷・障害データがあるかチェック
    has_injury_data = any(x > 0 for x in processed_injury_data)
    if has_injury_data:
        print(f"外傷・障害データ確認: {player_name} - 総件数: {sum(processed_injury_data)}")
    
    # 5行1列のサブプロットを作成
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=[
            '<b>睡眠時間</b>', 
            '<b>体重</b>', 
            '<b>sRPE</b>', 
            '<b>ACWR</b>',
            '<b>sRPE Zスコア</b>'
        ],
        vertical_spacing=0.06
    )
    
    # 統一されたシックなカラーパレット
    colors = {
        'sleep': '#2C3E50',      # ダークスレート
        'weight': '#2C3E50',     # ダークスレート（他と統一）
        'rpe': '#2C3E50',        # ダークスレート
        'srpe': '#34495E',       # ダークグレー
        'acwr': '#2C3E50',       # ダークスレート
        'neutral': '#7F8C8D',    # グレー
        'zscore_srpe': '#F39C12',    # オレンジ
        'zscore_acwr': '#3498DB',    # ブルー
        'injury': '#E74C3C'      # 赤（外傷・障害用）
    }
    
    # 1. 睡眠時間
    if sleep_data and any(x is not None for x in sleep_data):
        fig.add_trace(
            go.Scatter(
                x=dates, y=sleep_data, 
                name='睡眠時間',
                line=dict(color=colors['sleep'], width=4, shape='spline', smoothing=0.8),
                marker=dict(size=8, line=dict(width=2, color='white'), symbol='circle'),
                hovertemplate='<b>睡眠時間</b><br>日付: %{x}<br>値: %{y:.2f}時間<extra></extra>',
                connectgaps=True,
            ),
            row=1, col=1
        )
    
    # 2. 体重（目標体重線付き）
    if weight_data and any(x is not None for x in weight_data):
        fig.add_trace(
            go.Scatter(
                x=dates, y=weight_data, 
                name='体重',
                line=dict(color=colors['weight'], width=4, shape='spline', smoothing=0.8),
                marker=dict(size=8, line=dict(width=2, color='white'), symbol='circle'),
                hovertemplate='<b>体重</b><br>日付: %{x}<br>値: %{y:.2f}kg<extra></extra>',
                connectgaps=True,
                mode='lines+markers'
            ),
            row=2, col=1
        )
        
        # 目標体重線を追加
        if target_weight_data and any(x is not None for x in target_weight_data):
            fig.add_trace(
                go.Scatter(
                    x=dates, y=target_weight_data,
                    name='目標体重',
                    line=dict(color='#E67E22', width=3, dash='dash'),
                    marker=dict(size=6, symbol='diamond'),
                    hovertemplate='<b>目標体重</b><br>日付: %{x}<br>値: %{y:.2f}kg<extra></extra>',
                    connectgaps=True
                ),
                row=2, col=1
            )
    
    # 3. sRPE（外傷・障害マーカー付き - 強化版）
    if srpe_data and any(x is not None for x in srpe_data):
        fig.add_trace(
            go.Scatter(
                x=dates, y=srpe_data, 
                name='sRPE',
                line=dict(color=colors['srpe'], width=4, shape='spline', smoothing=0.8),
                marker=dict(size=8, line=dict(width=2, color='white'), symbol='square'),
                hovertemplate='<b>sRPE</b><br>日付: %{x}<br>値: %{y:.1f}<extra></extra>',
                connectgaps=True
            ),
            row=3, col=1
        )
        
        # 外傷・障害を棒グラフで表示（sRPEグラフ - 強化版）
        if has_injury_data:
            # 最大sRPE値を取得してスケール調整
            max_srpe = max([val for val in srpe_data if val is not None]) if any(val is not None for val in srpe_data) else 100
            
            # 外傷・障害データを棒グラフ用に調整（より目立つように）
            injury_bar_data = []
            injury_dates = []
            injury_counts = []
            
            for i, (date, count) in enumerate(zip(dates, processed_injury_data)):
                if count > 0:
                    # 外傷・障害の棒の高さをsRPEの40%程度に設定（より目立つように）
                    bar_height = max_srpe * 0.4 * min(count, 3)  # 最大3件まで表示
                    injury_bar_data.append(bar_height)
                    injury_dates.append(date)
                    injury_counts.append(count)
                else:
                    injury_bar_data.append(0)
                    injury_dates.append(date)
                    injury_counts.append(0)
            
            fig.add_trace(
                go.Bar(
                    x=injury_dates,
                    y=injury_bar_data,
                    name='外傷・障害',
                    marker=dict(color=colors['injury'], opacity=0.8),
                    hovertemplate='<b>外傷・障害</b><br>日付: %{x}<br>件数: %{customdata}<extra></extra>',
                    customdata=injury_counts,
                    width=86400000 * 0.5  # 棒の幅を少し細く
                ),
                row=3, col=1
            )
    
    # 4. ACWR
    if acwr_data and any(x is not None for x in acwr_data):
        fig.add_trace(
            go.Scatter(
                x=dates, y=acwr_data, 
                name='ACWR',
                line=dict(color=colors['acwr'], width=4, shape='spline', smoothing=0.8),
                marker=dict(size=8, line=dict(width=2, color='white'), symbol='diamond'),
                hovertemplate='<b>ACWR</b><br>日付: %{x}<br>値: %{y:.2f}<extra></extra>',
                connectgaps=True
            ),
            row=4, col=1
        )
        
        # ACWR理想値と境界線を追加
        fig.add_hline(y=1.0, line_dash="dash", line_color=colors['neutral'], 
                     annotation_text="理想値 (1.0)", row=4, col=1)
        fig.add_hline(y=0.8, line_dash="dot", line_color=colors['neutral'], 
                     annotation_text="低リスク境界", row=4, col=1)
        fig.add_hline(y=1.3, line_dash="dot", line_color=colors['neutral'], 
                     annotation_text="高リスク境界", row=4, col=1)
    
    # 5. sRPE Zスコア（外傷・障害マーカー付き - 強化版）
    if srpe_zscore and any(x is not None for x in srpe_zscore):
        fig.add_trace(
            go.Scatter(
                x=dates, y=srpe_zscore,
                name='sRPE Z-score',
                line=dict(color=colors['zscore_srpe'], width=4, shape='spline', smoothing=0.8),
                marker=dict(size=8, line=dict(width=2, color='white'), symbol='square'),
                hovertemplate='<b>sRPE Z-score</b><br>日付: %{x}<br>値: %{y:.2f}<extra></extra>',
                connectgaps=True
            ),
            row=5, col=1
        )
        
        # 外傷・障害を棒グラフで表示（sRPE Zスコアグラフ - 強化版）
        if has_injury_data:
            # Zスコアの範囲に合わせて棒の高さを調整（より目立つように）
            injury_z_bar_data = []
            injury_z_dates = []
            injury_z_counts = []
            
            for i, (date, count) in enumerate(zip(dates, processed_injury_data)):
                if count > 0:
                    # 外傷・障害の棒の高さを固定値（Zスコア範囲内）- より目立つように
                    bar_height = 0.8 * min(count, 3)  # Zスコア0.8の高さ × 件数（最大3件）
                    injury_z_bar_data.append(bar_height)
                    injury_z_dates.append(date)
                    injury_z_counts.append(count)
                else:
                    injury_z_bar_data.append(0)
                    injury_z_dates.append(date)
                    injury_z_counts.append(0)
            
            fig.add_trace(
                go.Bar(
                    x=injury_z_dates,
                    y=injury_z_bar_data,
                    name='外傷・障害 (Z)',
                    marker=dict(color=colors['injury'], opacity=0.8),
                    hovertemplate='<b>外傷・障害</b><br>日付: %{x}<br>件数: %{customdata}<extra></extra>',
                    customdata=injury_z_counts,
                    showlegend=False,
                    width=86400000 * 0.5  # 棒の幅を少し細く
                ),
                row=5, col=1
            )
        
        # Zスコアの基準線を追加（sRPE）
        fig.add_hline(y=0, line_dash="solid", line_color=colors['neutral'], line_width=2,
                     annotation_text="平均値 (Z=0)", row=5, col=1)
        fig.add_hline(y=1, line_dash="dot", line_color="#27AE60", 
                     annotation_text="標準偏差+1", row=5, col=1)
        fig.add_hline(y=-1, line_dash="dot", line_color="#27AE60", 
                     annotation_text="標準偏差-1", row=5, col=1)
        fig.add_hline(y=2, line_dash="dot", line_color="#E74C3C", 
                     annotation_text="標準偏差+2", row=5, col=1)
        fig.add_hline(y=-2, line_dash="dot", line_color="#E74C3C", 
                     annotation_text="標準偏差-2", row=5, col=1)
    
    # 選択された日付に縦線を追加
    try:
        for row in range(1, 6):  # 5行に変更
            fig.add_vline(
                x=selected_date, 
                line_dash="solid", 
                line_color="rgba(0, 0, 0, 0.6)", 
                line_width=3,
                annotation_text="選択日",
                annotation_position="top",
                row=row, col=1
            )
    except Exception as e:
        # 縦線の追加でエラーが発生した場合はスキップ
        pass
    
    # Y軸ラベルを設定
    fig.update_yaxes(
        title_text="睡眠時間 (時間)", 
        row=1, col=1,
        gridcolor='rgba(52, 73, 94, 0.1)',
        linecolor='rgba(52, 73, 94, 0.3)',
        title_font=dict(size=12, color='#2C3E50')
    )
    
    fig.update_yaxes(
        title_text="体重 (kg)", 
        row=2, col=1,
        gridcolor='rgba(52, 73, 94, 0.1)',
        linecolor='rgba(52, 73, 94, 0.3)',
        title_font=dict(size=12, color='#2C3E50')
    )
    
    fig.update_yaxes(
        title_text="sRPE", 
        row=3, col=1,
        gridcolor='rgba(52, 73, 94, 0.1)',
        linecolor='rgba(52, 73, 94, 0.3)',
        title_font=dict(size=12, color='#2C3E50')
    )
    
    fig.update_yaxes(
        title_text="ACWR", 
        row=4, col=1,
        gridcolor='rgba(52, 73, 94, 0.1)',
        linecolor='rgba(52, 73, 94, 0.3)',
        title_font=dict(size=12, color='#2C3E50')
    )
    
    # Zスコア軸の設定（sRPE）
    fig.update_yaxes(
        title_text="Z-score", 
        row=5, col=1,
        gridcolor='rgba(52, 73, 94, 0.1)',
        linecolor='rgba(52, 73, 94, 0.3)',
        title_font=dict(size=12, color='#2C3E50'),
        range=[-3, 3]  # Zスコアの範囲を-3から3に固定
    )
    
    # レイアウト設定
    fig.update_layout(
        title=dict(
            text=f"<b>{player_name} - データ推移</b>",
            x=0.5,
            font=dict(size=20, color='#2C3E50', family='Arial Black')
        ),
        height=1400,  # 5行になったので高さを調整
        showlegend=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)'
        ),
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=120, b=50),
        font=dict(family="Arial", color='#2C3E50')
    )
    
    # X軸の設定
    fig.update_xaxes(
        gridcolor='rgba(52, 73, 94, 0.1)',
        linecolor='rgba(52, 73, 94, 0.3)',
        tickfont=dict(color='#2C3E50')
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def create_team_analysis(analyzer):
    """チーム分析グラフを作成（指定した日の各指標の棒グラフ + チーム平均推移）"""
    if not analyzer.data:
        st.error("データが読み込まれていません")
        return
    
    # 日付選択
    st.markdown("### 📅 分析日選択")
    available_dates = analyzer.dates
    
    selected_date = st.selectbox(
        "分析する日付を選択してください",
        options=available_dates,
        index=len(available_dates)-1 if available_dates else 0,  # 最新日を初期選択
        format_func=lambda x: x.strftime('%Y年%m月%d日 (%a)'),
        key="team_analysis_date"
    )
    
    if not selected_date:
        st.warning("日付を選択してください")
        return
    
    # 選択された日付のインデックスを取得
    try:
        selected_index = available_dates.index(selected_date)
    except ValueError:
        st.error("選択された日付のデータが見つかりません")
        return
    
    # 各選手の指定日データを収集
    players_data = []
    player_names = []
    
    for player in analyzer.players:
        player_data = analyzer.data['players'][player]
        
        # データ取得の安全な関数
        def get_safe_value(data_list, index):
            if index < len(data_list):
                val = data_list[index]
                if pd.notna(val) and val != '':
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return None
            return None
        
        # 各指標のデータを取得
        sleep_val = get_safe_value(player_data.get('sleep', []), selected_index)
        weight_val = get_safe_value(player_data.get('weight', []), selected_index)
        srpe_val = get_safe_value(player_data.get('srpe', []), selected_index)
        acwr_val = get_safe_value(player_data.get('acwr', []), selected_index)
        srpe_zscore_val = get_safe_value(player_data.get('srpe_zscore', []), selected_index)
        
        players_data.append({
            'player': player,
            'sleep': sleep_val,
            'weight': weight_val,
            'srpe': srpe_val,
            'acwr': acwr_val,
            'srpe_zscore': srpe_zscore_val
        })
        
        # 表示用の短縮名前（苗字のみ、またはスペース後の部分）
        display_name = player.split()[-1] if ' ' in player else player
        player_names.append(display_name)
    
    # 統一されたシックなカラーパレット
    colors = {
        'sleep': '#2C3E50',
        'weight': '#34495E', 
        'srpe': '#5D6D7E',
        'acwr': '#85929E',
        'srpe_zscore': '#F39C12'
    }
    
    # =============================================================================
    # 指定日の各指標棒グラフ（3行2列 + sRPE Zスコアは1行占有）
    # =============================================================================
    st.markdown(f"### 📊 {selected_date.strftime('%Y年%m月%d日')} の各指標比較")
    
    fig_bars = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            f'<b>睡眠時間</b>', 
            f'<b>体重</b>', 
            f'<b>sRPE</b>',
            f'<b>ACWR</b>',
            f'<b>sRPE Zスコア</b>',
            ''
        ],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar", "colspan": 2}, None]],  # sRPE Zスコアを横一面に
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. 睡眠時間
    sleep_values = [d['sleep'] if d['sleep'] is not None else 0 for d in players_data]
    sleep_colors = [colors['sleep'] if d['sleep'] is not None else '#BDC3C7' for d in players_data]
    sleep_text = [f'{v:.1f}h' if v > 0 else 'N/A' for v in sleep_values]
    
    fig_bars.add_trace(
        go.Bar(
            x=player_names, 
            y=sleep_values, 
            name='睡眠時間',
            marker=dict(color=sleep_colors),
            text=sleep_text,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>睡眠時間: %{text}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. 体重
    weight_values = [d['weight'] if d['weight'] is not None else 0 for d in players_data]
    weight_colors = [colors['weight'] if d['weight'] is not None else '#BDC3C7' for d in players_data]
    weight_text = [f'{v:.1f}kg' if v > 0 else 'N/A' for v in weight_values]
    
    fig_bars.add_trace(
        go.Bar(
            x=player_names, 
            y=weight_values, 
            name='体重',
            marker=dict(color=weight_colors),
            text=weight_text,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>体重: %{text}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. sRPE
    srpe_values = [d['srpe'] if d['srpe'] is not None else 0 for d in players_data]
    srpe_colors = [colors['srpe'] if d['srpe'] is not None else '#BDC3C7' for d in players_data]
    srpe_text = [f'{v:.1f}' if v > 0 else 'N/A' for v in srpe_values]
    
    fig_bars.add_trace(
        go.Bar(
            x=player_names, 
            y=srpe_values, 
            name='sRPE',
            marker=dict(color=srpe_colors),
            text=srpe_text,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>sRPE: %{text}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. ACWR（色分けあり）
    acwr_values = [d['acwr'] if d['acwr'] is not None else 0 for d in players_data]
    acwr_colors = []
    for d in players_data:
        if d['acwr'] is not None:
            if d['acwr'] > 1.3:
                acwr_colors.append('#E74C3C')  # 赤（高リスク）
            elif d['acwr'] < 0.8:
                acwr_colors.append('#F39C12')  # オレンジ（低負荷）
            else:
                acwr_colors.append('#27AE60')  # 緑（適正）
        else:
            acwr_colors.append('#BDC3C7')  # グレー（データなし）
    
    acwr_text = [f'{v:.2f}' if v > 0 else 'N/A' for v in acwr_values]
    
    fig_bars.add_trace(
        go.Bar(
            x=player_names, 
            y=acwr_values, 
            name='ACWR',
            marker=dict(color=acwr_colors),
            text=acwr_text,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>ACWR: %{text}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # ACWR理想値と境界線を追加
    fig_bars.add_hline(y=1.0, line_dash="dash", line_color="#7F8C8D", 
                      annotation_text="理想値 (1.0)", row=2, col=2)
    fig_bars.add_hline(y=0.8, line_dash="dot", line_color="#F39C12", 
                      annotation_text="低リスク境界", row=2, col=2)
    fig_bars.add_hline(y=1.3, line_dash="dot", line_color="#E74C3C", 
                      annotation_text="高リスク境界", row=2, col=2)
    
    # 5. sRPE Zスコア（横一面使用）
    zscore_values = [d['srpe_zscore'] if d['srpe_zscore'] is not None else 0 for d in players_data]
    zscore_colors = []
    for d in players_data:
        if d['srpe_zscore'] is not None:
            if abs(d['srpe_zscore']) > 2:
                zscore_colors.append('#E74C3C')  # 赤（±2σ超）
            elif abs(d['srpe_zscore']) > 1:
                zscore_colors.append('#F39C12')  # オレンジ（±1σ超）
            else:
                zscore_colors.append('#27AE60')  # 緑（±1σ内）
        else:
            zscore_colors.append('#BDC3C7')  # グレー（データなし）
    
    zscore_text = [f'{v:.2f}' if d['srpe_zscore'] is not None else 'N/A' 
                  for v, d in zip(zscore_values, players_data)]
    
    fig_bars.add_trace(
        go.Bar(
            x=player_names, 
            y=zscore_values, 
            name='sRPE Zスコア',
            marker=dict(color=zscore_colors),
            text=zscore_text,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>sRPE Z-score: %{text}<extra></extra>'
        ),
        row=3, col=1  # colspanで横一面使用
    )
    
    # Zスコアの基準線を追加
    fig_bars.add_hline(y=0, line_dash="solid", line_color="#7F8C8D", line_width=2,
                      annotation_text="平均値 (Z=0)", row=3, col=1)
    fig_bars.add_hline(y=1, line_dash="dot", line_color="#27AE60", 
                      annotation_text="標準偏差+1", row=3, col=1)
    fig_bars.add_hline(y=-1, line_dash="dot", line_color="#27AE60", 
                      annotation_text="標準偏差-1", row=3, col=1)
    fig_bars.add_hline(y=2, line_dash="dot", line_color="#E74C3C", 
                      annotation_text="標準偏差+2", row=3, col=1)
    fig_bars.add_hline(y=-2, line_dash="dot", line_color="#E74C3C", 
                      annotation_text="標準偏差-2", row=3, col=1)
    
    # Y軸の設定
    fig_bars.update_yaxes(title_text="時間", row=1, col=1)
    fig_bars.update_yaxes(title_text="kg", row=1, col=2)
    fig_bars.update_yaxes(title_text="sRPE", row=2, col=1)
    fig_bars.update_yaxes(title_text="ACWR", row=2, col=2)
    fig_bars.update_yaxes(
        title_text="Z-score", 
        row=3, col=1,
        range=[-3, 3]  # Zスコアの範囲を-3から3に固定
    )
    
    # レイアウト設定
    fig_bars.update_layout(
        height=900,  # 3行になったので高さを増加
        title=dict(
            text=f"<b>各指標比較 - {selected_date.strftime('%Y年%m月%d日')}</b>",
            x=0.5,
            font=dict(size=18, color='#2C3E50', family='Arial Black')
        ),
        showlegend=False,
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        font=dict(family="Arial", color='#2C3E50'),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # X軸の設定（角度調整）
    fig_bars.update_xaxes(
        tickangle=45,
        gridcolor='rgba(52, 73, 94, 0.1)',
        linecolor='rgba(52, 73, 94, 0.3)',
        tickfont=dict(color='#2C3E50', size=10)
    )
    
    fig_bars.update_yaxes(
        gridcolor='rgba(52, 73, 94, 0.1)',
        linecolor='rgba(52, 73, 94, 0.3)',
        tickfont=dict(color='#2C3E50')
    )
    
    st.plotly_chart(fig_bars, use_container_width=True, config={'displayModeBar': False})
    
    # =============================================================================
    # チーム平均sRPE Zスコア推移 + 障害発生数グラフ（強化版）
    # =============================================================================
    st.markdown("### 📈 チーム平均sRPE Zスコア推移 & 障害発生状況")
    
    # 各日のチーム平均sRPE Zスコアと障害発生数を計算（強化版）
    dates = analyzer.dates
    team_avg_zscore = []
    daily_injury_totals = []
    
    for day_idx in range(len(dates)):
        day_zscores = []
        daily_injuries = 0
        
        for player in analyzer.players:
            player_data = analyzer.data['players'][player]
            
            # sRPE Zスコアを取得
            if day_idx < len(player_data.get('srpe_zscore', [])):
                zscore = player_data['srpe_zscore'][day_idx]
                if pd.notna(zscore):
                    day_zscores.append(zscore)
            
            # 外傷・障害データを集計（強化版）
            if day_idx < len(player_data.get('injuries', [])):
                injury_count = player_data['injuries'][day_idx]
                if pd.notna(injury_count) and injury_count != '' and injury_count is not None:
                    try:
                        count = int(float(injury_count))
                        if count > 0:
                            daily_injuries += count
                    except (ValueError, TypeError):
                        pass
        
        if day_zscores:
            team_avg_zscore.append(np.mean(day_zscores))
        else:
            team_avg_zscore.append(None)
        
        daily_injury_totals.append(daily_injuries)
    
    # デバッグ用：チーム全体の外傷・障害数を確認
    total_team_injuries = sum(daily_injury_totals)
    injury_days = sum([1 for x in daily_injury_totals if x > 0])
    if total_team_injuries > 0:
        print(f"チーム外傷・障害データ: 総件数{total_team_injuries}件、発生日数{injury_days}日")
    
    # チーム平均推移グラフを作成（個別分析と同じスタイル）
    fig_trend = make_subplots(
        rows=1, cols=1,
        subplot_titles=['<b>チーム平均sRPE Zスコア推移</b>'],
        vertical_spacing=0.1
    )
    
    # チーム平均sRPE Zスコア推移（折れ線グラフ）- 個別分析と同じスタイル
    if team_avg_zscore and any(x is not None for x in team_avg_zscore):
        fig_trend.add_trace(
            go.Scatter(
                x=dates, 
                y=team_avg_zscore,
                name='チーム平均sRPE Z-score',
                line=dict(color='#F39C12', width=4, shape='spline', smoothing=0.8),
                marker=dict(size=8, line=dict(width=2, color='white'), symbol='square'),
                hovertemplate='<b>チーム平均sRPE Z-score</b><br>日付: %{x}<br>値: %{y:.2f}<extra></extra>',
                connectgaps=True
            ),
            row=1, col=1
        )
    
    # 外傷・障害を棒グラフで表示（強化版）
    if daily_injury_totals and any(count > 0 for count in daily_injury_totals):
        # Zスコアの範囲に合わせて棒の高さを調整（より目立つように）
        injury_bar_data = []
        injury_dates = []
        injury_counts = []
        
        for date, count in zip(dates, daily_injury_totals):
            if count > 0:
                # 外傷・障害の棒の高さを固定値（Zスコア範囲内）- より目立つように
                bar_height = 0.8 * min(count, 5)  # Zスコア0.8の高さ × 件数（最大5件）
                injury_bar_data.append(bar_height)
                injury_dates.append(date)
                injury_counts.append(count)
            else:
                injury_bar_data.append(0)
                injury_dates.append(date)
                injury_counts.append(0)
        
        fig_trend.add_trace(
            go.Bar(
                x=injury_dates,
                y=injury_bar_data,
                name='外傷・障害',
                marker=dict(color='#E74C3C', opacity=0.8),
                hovertemplate='<b>外傷・障害</b><br>日付: %{x}<br>件数: %{customdata}<extra></extra>',
                customdata=injury_counts,
                width=86400000 * 0.5  # 棒の幅調整
            ),
            row=1, col=1
        )
    
    # 選択日に縦線を追加
    try:
        fig_trend.add_vline(
            x=selected_date, 
            line_dash="solid", 
            line_color="rgba(0, 0, 0, 0.6)", 
            line_width=3,
            annotation_text="選択日",
            annotation_position="top",
            row=1, col=1
        )
    except Exception as e:
        # 縦線の追加でエラーが発生した場合はスキップ
        pass
    
    # Zスコアの基準線を追加（個別分析と同じ）
    fig_trend.add_hline(y=0, line_dash="solid", line_color="#7F8C8D", line_width=2,
                       annotation_text="平均値 (Z=0)", row=1, col=1)
    fig_trend.add_hline(y=1, line_dash="dot", line_color="#27AE60", 
                       annotation_text="標準偏差+1", row=1, col=1)
    fig_trend.add_hline(y=-1, line_dash="dot", line_color="#27AE60", 
                       annotation_text="標準偏差-1", row=1, col=1)
    fig_trend.add_hline(y=2, line_dash="dot", line_color="#E74C3C", 
                       annotation_text="標準偏差+2", row=1, col=1)
    fig_trend.add_hline(y=-2, line_dash="dot", line_color="#E74C3C", 
                       annotation_text="標準偏差-2", row=1, col=1)
    
    # Y軸の設定（個別分析と同じ）
    fig_trend.update_yaxes(
        title_text="Z-score", 
        row=1, col=1,
        range=[-3, 3],  # Zスコアの範囲を-3から3に固定
        gridcolor='rgba(52, 73, 94, 0.1)',
        linecolor='rgba(52, 73, 94, 0.3)',
        title_font=dict(size=12, color='#2C3E50')
    )
    
    # X軸の設定
    fig_trend.update_xaxes(
        gridcolor='rgba(52, 73, 94, 0.1)',
        linecolor='rgba(52, 73, 94, 0.3)',
        tickfont=dict(color='#2C3E50')
    )
    
    # レイアウト設定（個別分析と同じスタイル）
    fig_trend.update_layout(
        height=500,
        title=dict(
            text="<b>チーム平均sRPE Zスコア推移 & 外傷・障害発生状況</b>",
            x=0.5,
            font=dict(size=16, color='#2C3E50', family='Arial Black')
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)'
        ),
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        font=dict(family="Arial", color='#2C3E50'),
        margin=dict(l=50, r=50, t=120, b=50)
    )
    
    st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
    
    # 外傷・障害データのサマリー表示
    if total_team_injuries > 0:
        st.info(f"💡 期間中の外傷・障害発生状況: 総件数 {total_team_injuries}件、発生日数 {injury_days}日")
    else:
        st.success("✅ 期間中に外傷・障害の発生はありませんでした")
    
    # =============================================================================
    # データの詳細表示
    # =============================================================================
    st.markdown(f"### 📋 {selected_date.strftime('%Y年%m月%d日')} の詳細データ")
    
    # 表示用データフレームを作成
    table_data = []
    for i, (data, name) in enumerate(zip(players_data, player_names)):
        def format_value(val, decimal=2):
            if val is not None:
                return f"{val:.{decimal}f}"
            return "N/A"
        
        # リスク判定
        risk_status = "N/A"
        if data['acwr'] is not None:
            if data['acwr'] > 1.3:
                risk_status = "🔴 高リスク"
            elif data['acwr'] < 0.8:
                risk_status = "🟡 低負荷"
            else:
                risk_status = "🟢 適正"
        
        # Zスコア判定
        zscore_status = "N/A"
        if data['srpe_zscore'] is not None:
            if abs(data['srpe_zscore']) > 2:
                zscore_status = "🔴 注意"
            elif abs(data['srpe_zscore']) > 1:
                zscore_status = "🟡 監視"
            else:
                zscore_status = "🟢 正常"
        
        # 選手の外傷・障害データを取得
        player_injury_count = 0
        if selected_index < len(analyzer.data['players'][data['player']].get('injuries', [])):
            injury_val = analyzer.data['players'][data['player']]['injuries'][selected_index]
            if pd.notna(injury_val) and injury_val != '' and injury_val is not None:
                try:
                    player_injury_count = int(float(injury_val))
                except (ValueError, TypeError):
                    player_injury_count = 0
        
        table_data.append({
            "選手名": data['player'],
            "睡眠時間": format_value(data['sleep'], 1) + "h" if data['sleep'] is not None else "N/A",
            "体重": format_value(data['weight'], 1) + "kg" if data['weight'] is not None else "N/A",
            "sRPE": format_value(data['srpe'], 1) if data['srpe'] is not None else "N/A",
            "ACWR": format_value(data['acwr'], 2) if data['acwr'] is not None else "N/A",
            "ACWR判定": risk_status,
            "sRPE Z値": format_value(data['srpe_zscore'], 2) if data['srpe_zscore'] is not None else "N/A",
            "Z値判定": zscore_status,
            "外傷・障害": f"{player_injury_count}件" if player_injury_count > 0 else "-"
        })
    
    # データフレームを作成して表示
    df_team_table = pd.DataFrame(table_data)
    
    st.dataframe(
        df_team_table,
        use_container_width=True,
        height=400,
        column_config={
            "選手名": st.column_config.TextColumn("選手名", width="medium"),
            "睡眠時間": st.column_config.TextColumn("睡眠時間", width="small"),
            "体重": st.column_config.TextColumn("体重", width="small"),
            "sRPE": st.column_config.TextColumn("sRPE", width="small"),
            "ACWR": st.column_config.TextColumn("ACWR", width="small"),
            "ACWR判定": st.column_config.TextColumn("ACWR判定", width="medium"),
            "sRPE Z値": st.column_config.TextColumn("sRPE Z値", width="small"),
            "Z値判定": st.column_config.TextColumn("Z値判定", width="medium"),
            "外傷・障害": st.column_config.TextColumn("外傷・障害", width="small")
        }
    )

def generate_summary_report(analyzer):
    """サマリーレポートを生成"""
    if not analyzer.data:
        st.error("データが読み込まれていません")
        return
    
    st.markdown("## 📊 サマリーレポート")
    
    # 基本情報
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("分析期間", f"{analyzer.dates[0].strftime('%Y-%m-%d')} ～ {analyzer.dates[-1].strftime('%Y-%m-%d')}")
    with col2:
        st.metric("分析選手数", f"{len(analyzer.players)}人")
    with col3:
        total_days = len(analyzer.dates)
        st.metric("分析日数", f"{total_days}日")
    
    # 外傷・障害の統計を追加
    st.markdown("### 🚑 外傷・障害発生状況")
    
    # 全期間の外傷・障害データを集計
    total_injuries = 0
    player_injury_stats = {}
    
    for player in analyzer.players:
        player_data = analyzer.data['players'][player]
        injury_data = player_data.get('injuries', [])
        
        player_total = 0
        for injury_count in injury_data:
            if pd.notna(injury_count) and injury_count != '' and injury_count is not None:
                try:
                    count = int(float(injury_count))
                    if count > 0:
                        player_total += count
                        total_injuries += count
                except (ValueError, TypeError):
                    pass
        
        player_injury_stats[player] = player_total
    
    # 外傷・障害の統計表示
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("総外傷・障害件数", f"{total_injuries}件")
    with col2:
        injury_players = sum([1 for count in player_injury_stats.values() if count > 0])
        st.metric("外傷・障害発生選手数", f"{injury_players}人")
    with col3:
        if total_injuries > 0:
            avg_per_player = total_injuries / len(analyzer.players)
            st.metric("選手平均外傷・障害件数", f"{avg_per_player:.1f}件")
        else:
            st.metric("選手平均外傷・障害件数", "0件")
    
    # 外傷・障害が多い選手のリスト
    if total_injuries > 0:
        high_injury_players = [(player, count) for player, count in player_injury_stats.items() if count > 0]
        high_injury_players.sort(key=lambda x: x[1], reverse=True)
        
        st.markdown("**外傷・障害発生選手:**")
        for player, count in high_injury_players[:5]:  # 上位5名まで表示
            st.write(f"- {player}: {count}件")
    
    # 選手別詳細統計
    st.markdown("### 👥 選手別統計")
    
    stats_data = []
    high_risk_players = []
    low_risk_players = []
    
    for player in analyzer.players:
        player_data = analyzer.data['players'][player]
        
        # 基本統計（空文字も除外）
        sleep_values = [x for x in player_data.get('sleep', []) if pd.notna(x) and x != '']
        weight_values = [x for x in player_data.get('weight', []) if pd.notna(x) and x != '']
        srpe_values = [x for x in player_data.get('srpe', []) if pd.notna(x) and x != '']
        acwr_values = [x for x in player_data.get('acwr', []) if pd.notna(x) and x != '']
        
        # ACWR リスク判定
        risk_status = "データ不足"
        latest_acwr = 0
        if acwr_values:
            latest_acwr = acwr_values[-1]
            if latest_acwr > 1.3:
                risk_status = "🔴 高リスク"
                high_risk_players.append(f"{player} (ACWR: {latest_acwr:.2f})")
            elif latest_acwr < 0.8:
                risk_status = "🟡 低負荷"
                low_risk_players.append(f"{player} (ACWR: {latest_acwr:.2f})")
            else:
                risk_status = "🟢 適正範囲"
        
        stats_data.append({
            "選手名": player,
            "平均睡眠時間": f"{np.mean(sleep_values):.2f}±{np.std(sleep_values):.2f}" if sleep_values else "N/A",
            "平均体重": f"{np.mean(weight_values):.2f}±{np.std(weight_values):.2f}" if weight_values else "N/A",
            "平均sRPE": f"{np.mean(srpe_values):.2f}±{np.std(srpe_values):.2f}" if srpe_values else "N/A",
            "最新ACWR": f"{latest_acwr:.2f}" if latest_acwr > 0 else "N/A",
            "リスクステータス": risk_status,
            "外傷・障害件数": f"{player_injury_stats.get(player, 0)}件"
        })
    
    # 統計テーブル表示
    df_stats = pd.DataFrame(stats_data)
    st.dataframe(df_stats, use_container_width=True)
    
    # リスク分析
    st.markdown("### ⚠️ リスク分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if high_risk_players:
            st.markdown("**🔴 高負荷リスク選手:**")
            for player in high_risk_players:
                st.write(f"- {player}")
        else:
            st.success("高負荷リスク選手はいません")
    
    with col2:
        if low_risk_players:
            st.markdown("**🟡 低負荷選手:**")
            for player in low_risk_players:
                st.write(f"- {player}")
        else:
            st.info("低負荷選手はいません")
    
    if not high_risk_players and not low_risk_players:
        st.success("🎉 全選手が適正範囲内です！")

def create_individual_csv(analyzer, selected_player, filtered_dates, filtered_data):
    """個別選手データのCSV生成"""
    csv_data = []
    for i, date in enumerate(filtered_dates):
        def format_value(val):
            if val is not None and pd.notna(val) and val != '':
                return val
            return ""
        
        row = {
            "日付": date.strftime('%Y-%m-%d'),
            "選手名": selected_player,
            "睡眠時間": format_value(filtered_data['sleep'][i] if i < len(filtered_data['sleep']) else None),
            "体重": format_value(filtered_data['weight'][i] if i < len(filtered_data['weight']) else None),
            "RPE": format_value(filtered_data['rpe'][i] if i < len(filtered_data['rpe']) else None),
            "活動時間": format_value(filtered_data['activity_time'][i] if i < len(filtered_data['activity_time']) else None),
            "sRPE": format_value(filtered_data['srpe'][i] if i < len(filtered_data['srpe']) else None),
            "ACWR": format_value(filtered_data['acwr'][i] if i < len(filtered_data['acwr']) else None),
            "睡眠時間_Zスコア": format_value(filtered_data['sleep_zscore'][i] if i < len(filtered_data['sleep_zscore']) else None),
            "体重_Zスコア": format_value(filtered_data['weight_zscore'][i] if i < len(filtered_data['weight_zscore']) else None),
            "sRPE_Zスコア": format_value(filtered_data['srpe_zscore'][i] if i < len(filtered_data['srpe_zscore']) else None),
            "ACWR_Zスコア": format_value(filtered_data['acwr_zscore'][i] if i < len(filtered_data['acwr_zscore']) else None),
            "外傷障害件数": format_value(filtered_data['injuries'][i] if i < len(filtered_data['injuries']) else 0),
            "目標体重": format_value(filtered_data['target_weight'][i] if i < len(filtered_data['target_weight']) else None),
            "体重差": format_value(filtered_data['target_weight'][i] - filtered_data['weight'][i] if (i < len(filtered_data['target_weight']) and i < len(filtered_data['weight']) and filtered_data['target_weight'][i] is not None and filtered_data['weight'][i] is not None) else None)
        }
        csv_data.append(row)
    
    return pd.DataFrame(csv_data)

def create_all_players_csv(analyzer, start_date, end_date):
    """全選手の指定期間データをまとめてCSV生成"""
    all_data = []
    
    # 日付範囲を設定
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    
    for player in analyzer.players:
        player_data = analyzer.data['players'][player]
        dates = analyzer.dates[:len(player_data.get('sleep', []))]
        
        # 期間フィルタリング
        mask = [(d >= start_datetime and d < end_datetime) for d in dates]
        filtered_dates = [d for d, m in zip(dates, mask) if m]
        
        if filtered_dates:
            # データをフィルタリング
            def filter_data(data_list, mask):
                return [data_list[i] if i < len(data_list) else np.nan for i, m in enumerate(mask) if m]
            
            filtered_data = {
                'sleep': filter_data(player_data.get('sleep', []), mask),
                'weight': filter_data(player_data.get('weight', []), mask),
                'rpe': filter_data(player_data.get('rpe', []), mask),
                'activity_time': filter_data(player_data.get('activity_time', []), mask),
                'srpe': filter_data(player_data.get('srpe', []), mask),
                'acwr': filter_data(player_data.get('acwr', []), mask),
                'sleep_zscore': filter_data(player_data.get('sleep_zscore', []), mask),
                'weight_zscore': filter_data(player_data.get('weight_zscore', []), mask),
                'srpe_zscore': filter_data(player_data.get('srpe_zscore', []), mask),
                'acwr_zscore': filter_data(player_data.get('acwr_zscore', []), mask),
                'injuries': filter_data(player_data.get('injuries', []), mask)
            }
            
            # 各日のデータを追加
            for i, date in enumerate(filtered_dates):
                def format_value(val):
                    if val is not None and pd.notna(val) and val != '':
                        return val
                    return ""
                
                row = {
                    "日付": date.strftime('%Y-%m-%d'),
                    "選手名": player,
                    "睡眠時間": format_value(filtered_data['sleep'][i] if i < len(filtered_data['sleep']) else None),
                    "体重": format_value(filtered_data['weight'][i] if i < len(filtered_data['weight']) else None),
                    "RPE": format_value(filtered_data['rpe'][i] if i < len(filtered_data['rpe']) else None),
                    "活動時間": format_value(filtered_data['activity_time'][i] if i < len(filtered_data['activity_time']) else None),
                    "sRPE": format_value(filtered_data['srpe'][i] if i < len(filtered_data['srpe']) else None),
                    "ACWR": format_value(filtered_data['acwr'][i] if i < len(filtered_data['acwr']) else None),
                    "睡眠時間_Zスコア": format_value(filtered_data['sleep_zscore'][i] if i < len(filtered_data['sleep_zscore']) else None),
                    "体重_Zスコア": format_value(filtered_data['weight_zscore'][i] if i < len(filtered_data['weight_zscore']) else None),
                    "sRPE_Zスコア": format_value(filtered_data['srpe_zscore'][i] if i < len(filtered_data['srpe_zscore']) else None),
                    "ACWR_Zスコア": format_value(filtered_data['acwr_zscore'][i] if i < len(filtered_data['acwr_zscore']) else None),
                    "外傷障害件数": format_value(filtered_data['injuries'][i] if i < len(filtered_data['injuries']) else 0)
                }
                all_data.append(row)
    
    return pd.DataFrame(all_data)

def create_team_comparison_csv(analyzer):
    """チーム比較データのCSV生成"""
    stats_data = []
    
    for player in analyzer.players:
        player_data = analyzer.data['players'][player]
        
        # 基本統計
        sleep_values = [x for x in player_data.get('sleep', []) if pd.notna(x) and x != '']
        weight_values = [x for x in player_data.get('weight', []) if pd.notna(x) and x != '']
        srpe_values = [x for x in player_data.get('srpe', []) if pd.notna(x) and x != '']
        acwr_values = [x for x in player_data.get('acwr', []) if pd.notna(x) and x != '']
        injury_values = [x for x in player_data.get('injuries', []) if pd.notna(x)]
        
        # ACWR リスク判定
        risk_status = "データ不足"
        latest_acwr = ""
        if acwr_values:
            latest_acwr = acwr_values[-1]
            if latest_acwr > 1.3:
                risk_status = "高リスク"
            elif latest_acwr < 0.8:
                risk_status = "低負荷"
            else:
                risk_status = "適正範囲"
        
        # 外傷・障害の総件数を計算
        total_injuries = 0
        for injury_count in injury_values:
            if pd.notna(injury_count) and injury_count != '' and injury_count is not None:
                try:
                    count = int(float(injury_count))
                    if count > 0:
                        total_injuries += count
                except (ValueError, TypeError):
                    pass
        
        stats_data.append({
            "選手名": player,
            "平均睡眠時間": np.mean(sleep_values) if sleep_values else "",
            "睡眠時間_標準偏差": np.std(sleep_values) if len(sleep_values) > 1 else "",
            "平均体重": np.mean(weight_values) if weight_values else "",
            "体重_標準偏差": np.std(weight_values) if len(weight_values) > 1 else "",
            "平均sRPE": np.mean(srpe_values) if srpe_values else "",
            "sRPE_標準偏差": np.std(srpe_values) if len(srpe_values) > 1 else "",
            "最新ACWR": latest_acwr,
            "リスクステータス": risk_status,
            "総外傷障害件数": total_injuries,
            "データ期間_開始": analyzer.dates[0].strftime('%Y-%m-%d') if analyzer.dates else "",
            "データ期間_終了": analyzer.dates[-1].strftime('%Y-%m-%d') if analyzer.dates else "",
            "分析日数": len(analyzer.dates)
        })
    
    return pd.DataFrame(stats_data)

def create_daily_team_csv(analyzer):
    """チーム全体の日別データCSV生成"""
    daily_data = []
    
    for day_idx, date in enumerate(analyzer.dates):
        # 各日のチームデータを集計
        daily_sleep = []
        daily_weight = []
        daily_srpe = []
        daily_acwr = []
        daily_injuries = 0
        daily_zscore_srpe = []
        
        for player in analyzer.players:
            player_data = analyzer.data['players'][player]
            
            # 各指標のデータを取得
            if day_idx < len(player_data.get('sleep', [])):
                sleep_val = player_data['sleep'][day_idx]
                if pd.notna(sleep_val) and sleep_val != '':
                    daily_sleep.append(sleep_val)
            
            if day_idx < len(player_data.get('weight', [])):
                weight_val = player_data['weight'][day_idx]
                if pd.notna(weight_val) and weight_val != '':
                    daily_weight.append(weight_val)
            
            if day_idx < len(player_data.get('srpe', [])):
                srpe_val = player_data['srpe'][day_idx]
                if pd.notna(srpe_val) and srpe_val != '':
                    daily_srpe.append(srpe_val)
            
            if day_idx < len(player_data.get('acwr', [])):
                acwr_val = player_data['acwr'][day_idx]
                if pd.notna(acwr_val) and acwr_val != '':
                    daily_acwr.append(acwr_val)
            
            if day_idx < len(player_data.get('srpe_zscore', [])):
                zscore_val = player_data['srpe_zscore'][day_idx]
                if pd.notna(zscore_val) and zscore_val != '':
                    daily_zscore_srpe.append(zscore_val)
            
            # 外傷・障害データを集計
            if day_idx < len(player_data.get('injuries', [])):
                injury_count = player_data['injuries'][day_idx]
                if pd.notna(injury_count) and injury_count != '' and injury_count is not None:
                    try:
                        count = int(float(injury_count))
                        if count > 0:
                            daily_injuries += count
                    except (ValueError, TypeError):
                        pass
        
        daily_data.append({
            "日付": date.strftime('%Y-%m-%d'),
            "チーム平均睡眠時間": np.mean(daily_sleep) if daily_sleep else "",
            "チーム平均体重": np.mean(daily_weight) if daily_weight else "",
            "チーム平均sRPE": np.mean(daily_srpe) if daily_srpe else "",
            "チーム平均ACWR": np.mean(daily_acwr) if daily_acwr else "",
            "チーム平均sRPE_Zスコア": np.mean(daily_zscore_srpe) if daily_zscore_srpe else "",
            "チーム総外傷障害件数": daily_injuries,
            "データ有効選手数_睡眠": len(daily_sleep),
            "データ有効選手数_体重": len(daily_weight),
            "データ有効選手数_sRPE": len(daily_srpe),
            "データ有効選手数_ACWR": len(daily_acwr)
        })
    
    return pd.DataFrame(daily_data)

def main():
    """メイン関数"""
    st.title("🏀 コンディション管理アプリ")
    st.markdown("---")
    
    # セッション状態の初期化
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SportsDataAnalyzer()
    
    # サイドバー
    with st.sidebar:
        st.header("📁 ファイル読み込み")
        uploaded_file = st.file_uploader(
            "エクセルファイルを選択してください",
            type=['xlsx', 'xls'],
            help="A列に選手名、B列に項目名、C列以降に日別データがあるファイルを選択してください"
        )
        
        if uploaded_file is not None:
            if st.button("📊 データ読み込み", type="primary"):
                with st.spinner("データを読み込み中..."):
                    if st.session_state.analyzer.load_excel_data(uploaded_file):
                        st.success(f"✅ データを正常に読み込みました！\n選手数: {len(st.session_state.analyzer.players)}人")
                        
                        # 目標体重データの確認
                        target_weight_found = False
                        for player in st.session_state.analyzer.players:
                            player_data = st.session_state.analyzer.data['players'][player]
                            if 'target_weight' in player_data:
                                target_weights = [w for w in player_data['target_weight'] if w is not None]
                                if target_weights:
                                    target_weight_found = True
                                    break
                        
                        if target_weight_found:
                            st.info("📊 目標体重データも読み込まれました")
                        else:
                            st.warning("⚠️ 目標体重データが見つかりませんでした。「目標体重」または「Sheet3」シートを確認してください。")
                        
                        # 外傷・障害データの確認
                        injury_found = False
                        total_injuries = 0
                        for player in st.session_state.analyzer.players:
                            player_data = st.session_state.analyzer.data['players'][player]
                            if 'injuries' in player_data:
                                injury_counts = [x for x in player_data['injuries'] if pd.notna(x) and x != '' and x != 0]
                                if injury_counts:
                                    injury_found = True
                                    for count in injury_counts:
                                        try:
                                            total_injuries += int(float(count))
                                        except (ValueError, TypeError):
                                            pass
                        
                        if injury_found and total_injuries > 0:
                            st.success(f"🚑 外傷・障害データも読み込まれました（総件数: {total_injuries}件）")
                            
                            # 外傷・障害データの詳細を表示
                            injury_details = []
                            for player in st.session_state.analyzer.players:
                                player_data = st.session_state.analyzer.data['players'][player]
                                if 'injuries' in player_data:
                                    player_total = sum([x for x in player_data['injuries'] if pd.notna(x) and x != '' and x != 0])
                                    if player_total > 0:
                                        injury_details.append(f"{player}: {player_total}件")
                            
                            if injury_details:
                                st.info(f"📋 選手別内訳: " + ", ".join(injury_details[:3]) + ("..." if len(injury_details) > 3 else ""))
                        else:
                            st.warning("⚠️ 外傷・障害データが見つかりませんでした。「外傷・障害」シートを確認してください。")
                        
                        st.rerun()
                    else:
                        st.error("❌ データの読み込みに失敗しました")
        
        # 機能説明
        st.markdown("---")
        st.header("📖 機能説明")
        st.markdown("""
        **計算項目:**
        - **sRPE**: RPE × 個別活動時間
        - **Chronic RPE**: 28日間のsRPE平均
        - **Acute RPE**: 7日間のsRPE平均  
        - **ACWR**: Acute ÷ Chronic (理想値: 0.8-1.3)
        - **Zスコア**: チーム平均からの偏差を標準化
        
        **分析項目:**
        - 睡眠時間の推移
        - 体重の推移
        - RPE、sRPE
        - ACWR (傷害リスク指標)
        - 各指標のZスコア表示
        - チーム全体の比較分析
        - 詳細サマリーレポート
        - 外傷・障害データとの関係分析
        """)
    
    # メインコンテンツ
    if st.session_state.analyzer.data is None:
        st.info("👆 サイドバーからエクセルファイルを読み込んでください")
        
        # サンプルデータ形式の説明
        st.markdown("## 📋 必要なデータ形式")
        st.markdown("""
        エクセルファイルは以下の形式で作成してください：
        
        **Sheet1（メインデータ）:**
        - **A列**: 選手名
        - **B列**: 項目名（睡眠時間、体重、RPE、活動時間）
        - **C列以降**: 日別データ
        - **各選手4行構成**：睡眠時間、体重、RPE、活動時間
        - **1行目**: 日付データ
        - **2-4行目**: チーム平均データ（睡眠時間、体重、RPE）
        
        **Sheet2（外傷・障害データ）:**
        - **A列**: 選手名
        - **B列以降**: 日別外傷・障害件数（0または件数）
        - **シート名**: 「外傷・障害」、「外傷障害」、または「Sheet2」
        
        **Sheet3（目標体重データ）:**
        - **A列**: 選手名
        - **B列以降**: 日別目標体重
        """)
        
    else:
        # タブ作成
        tab1, tab2, tab3 = st.tabs(["👤 個別分析", "👥 チーム分析", "📊 サマリーレポート"])
        
        with tab1:
            st.header("👤 個別選手分析")
            
            # 選手選択
            selected_player = st.selectbox(
                "分析する選手を選択してください",
                st.session_state.analyzer.players,
                key="player_select"
            )
            
            if selected_player:
                player_data = st.session_state.analyzer.data['players'][selected_player]
                dates = st.session_state.analyzer.dates[:len(player_data.get('sleep', []))]
                
                # 日付選択機能
                st.markdown("### 📅 日付選択")
                col_date1, col_date2 = st.columns(2)
                
                with col_date1:
                    selected_date = st.selectbox(
                        "表示する日付を選択",
                        options=dates,
                        index=len(dates)-1 if dates else 0,  # 最新日を初期選択
                        format_func=lambda x: x.strftime('%Y-%m-%d')
                    )
                
                with col_date2:
                    # 期間表示用
                    start_date = st.date_input(
                        "推移グラフ開始日",
                        value=dates[0].date() if dates else datetime.now().date(),
                        min_value=dates[0].date() if dates else datetime.now().date(),
                        max_value=dates[-1].date() if dates else datetime.now().date()
                    )
                
                # 選択日のデータを取得
                selected_index = dates.index(selected_date)
                
                # その日の数値表示
                st.markdown(f"### 📊 {selected_date.strftime('%Y年%m月%d日')}のデータ")
                
                # 統一されたシックなカラーパレット
                colors = {
                    'sleep': '#2C3E50',      # ダークスレート
                    'weight': '#2C3E50',     # ダークスレート（他と統一）
                    'srpe': '#34495E',       # ダークグレー
                    'acwr': '#2C3E50'        # ダークスレート
                }
                
                # 主要指標を4つのカラムで表示（元の値とZスコアを統合）
                col1, col2, col3, col4 = st.columns(4)
                
                # データ取得（空文字もチェック）
                def get_safe_value(data_list, index):
                    if index < len(data_list):
                        val = data_list[index]
                        if pd.notna(val) and val != '':
                            return val
                    return None
                
                sleep_val = get_safe_value(player_data.get('sleep', []), selected_index)
                weight_val = get_safe_value(player_data.get('weight', []), selected_index)
                rpe_val = get_safe_value(player_data.get('rpe', []), selected_index)
                srpe_val = get_safe_value(player_data.get('srpe', []), selected_index)
                acwr_val = get_safe_value(player_data.get('acwr', []), selected_index)
                
                # Zスコア取得
                sleep_z = get_safe_value(player_data.get('sleep_zscore', []), selected_index)
                weight_z = get_safe_value(player_data.get('weight_zscore', []), selected_index)
                srpe_z = get_safe_value(player_data.get('srpe_zscore', []), selected_index)
                acwr_z = get_safe_value(player_data.get('acwr_zscore', []), selected_index)
                
                # 目標体重データ取得
                target_weight_val = get_safe_value(player_data.get('target_weight', []), selected_index)
                # 目標体重との差を計算
                weight_diff = None
                if weight_val is not None and target_weight_val is not None:
                    weight_diff = weight_val - target_weight_val
                
                # 外傷・障害データ取得
                injury_val = get_safe_value(player_data.get('injuries', []), selected_index)
                injury_count = 0
                if injury_val is not None:
                    try:
                        injury_count = int(float(injury_val))
                    except (ValueError, TypeError):
                        injury_count = 0
                
                with col1:
                    sleep_str = f"{sleep_val:.2f}h" if sleep_val is not None else "N/A"
                    sleep_z_str = f"Z: {sleep_z:.2f}" if sleep_z is not None else "Z: N/A"
                    z_indicator = "🟢" if sleep_z is not None and abs(sleep_z) <= 1 else "🔴" if sleep_z is not None else "⚪"
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {colors['sleep']} 0%, {colors['sleep']}DD 100%);
                        padding: 1.5rem;
                        border-radius: 15px;
                        color: white;
                        text-align: center;
                        margin: 0.5rem 0;
                        box-shadow: 0 10px 40px rgba(44, 62, 80, 0.4);
                        border: 1px solid rgba(255, 255, 255, 0.1);
                    ">
                        <div style="font-size: 1.1rem; margin-bottom: 0.5rem; opacity: 0.9; color: #BDC3C7;">睡眠時間</div>
                        <div style="font-size: 2.5rem; font-weight: 800; margin: 0.5rem 0; color: #ECF0F1;">{sleep_str}</div>
                        <div style="font-size: 1.3rem; font-weight: 600; opacity: 0.95; margin-top: 0.8rem;">{z_indicator} {sleep_z_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    weight_str = f"{weight_val:.2f}kg" if weight_val is not None else "N/A"
                    target_str = f"目標: {target_weight_val:.1f}kg" if target_weight_val is not None else "目標: N/A"
                    
                    # 目標体重との差を表示
                    if weight_diff is not None:
                        diff_str = f"差: {weight_diff:+.1f}kg"
                        if weight_diff > 0:
                            diff_indicator = "🔴"  # 目標より重い
                        elif weight_diff < -0.5:
                            diff_indicator = "🟡"  # 目標より軽すぎる
                        else:
                            diff_indicator = "🟢"  # 適正範囲
                    else:
                        diff_str = "差: N/A"
                        diff_indicator = "⚪"
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {colors['weight']} 0%, {colors['weight']}DD 100%);
                        padding: 1.5rem;
                        border-radius: 15px;
                        color: white;
                        text-align: center;
                        margin: 0.5rem 0;
                        box-shadow: 0 10px 40px rgba(44, 62, 80, 0.4);
                        border: 1px solid rgba(255, 255, 255, 0.1);
                    ">
                        <div style="font-size: 1.1rem; margin-bottom: 0.5rem; opacity: 0.9;">体重</div>
                        <div style="font-size: 2.5rem; font-weight: 800; margin: 0.5rem 0;">{weight_str}</div>
                        <div style="font-size: 0.9rem; opacity: 0.85; margin-bottom: 0.5rem;">{target_str}</div>
                        <div style="font-size: 1.3rem; font-weight: 600; opacity: 0.95; margin-top: 0.8rem;">{diff_indicator} {diff_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    rpe_str = f"{rpe_val:.1f}" if rpe_val is not None else "N/A"
                    srpe_str = f"sRPE: {srpe_val:.1f}" if srpe_val is not None else "sRPE: N/A"
                    srpe_z_str = f"Z: {srpe_z:.2f}" if srpe_z is not None else "Z: N/A"
                    z_indicator = "🟢" if srpe_z is not None and abs(srpe_z) <= 1 else "🔴" if srpe_z is not None else "⚪"
                    
                    # 外傷・障害表示を追加
                    injury_str = f"外傷: {injury_count}件" if injury_count > 0 else "外傷: なし"
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {colors['srpe']} 0%, {colors['srpe']}DD 100%);
                        padding: 1.5rem;
                        border-radius: 15px;
                        color: white;
                        text-align: center;
                        margin: 0.5rem 0;
                        box-shadow: 0 10px 40px rgba(52, 73, 94, 0.4);
                        border: 1px solid rgba(255, 255, 255, 0.1);
                    ">
                        <div style="font-size: 1.1rem; margin-bottom: 0.5rem; opacity: 0.9;">RPE</div>
                        <div style="font-size: 2.5rem; font-weight: 800; margin: 0.5rem 0;">{rpe_str}</div>
                        <div style="font-size: 0.9rem; opacity: 0.85; margin-bottom: 0.5rem;">{srpe_str}</div>
                        <div style="font-size: 0.85rem; opacity: 0.8; margin-bottom: 0.5rem; color: {'#E74C3C' if injury_count > 0 else '#95A5A6'};">{injury_str}</div>
                        <div style="font-size: 1.3rem; font-weight: 600; opacity: 0.95;">{z_indicator} {srpe_z_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    acwr_str = f"{acwr_val:.2f}" if acwr_val is not None else "N/A"
                    acwr_z_str = f"Z: {acwr_z:.2f}" if acwr_z is not None else "Z: N/A"
                    z_indicator = "🟢" if acwr_z is not None and abs(acwr_z) <= 1 else "🔴" if acwr_z is not None else "⚪"
                    
                    # ACWR色分け
                    if acwr_val is not None:
                        if acwr_val > 1.3:
                            risk_text = "🔴 高リスク"
                        elif acwr_val < 0.8:
                            risk_text = "🟡 低負荷"
                        else:
                            risk_text = "🟢 適正"
                    else:
                        risk_text = "データなし"
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {colors['acwr']} 0%, {colors['acwr']}DD 100%);
                        padding: 1.5rem;
                        border-radius: 15px;
                        color: white;
                        text-align: center;
                        margin: 0.5rem 0;
                        box-shadow: 0 10px 40px rgba(44, 62, 80, 0.4);
                        border: 1px solid rgba(255,255,255,0.15);
                    ">
                        <div style="font-size: 1.1rem; margin-bottom: 0.5rem; opacity: 0.9;">ACWR</div>
                        <div style="font-size: 2.5rem; font-weight: 800; margin: 0.5rem 0;">{acwr_str}</div>
                        <div style="font-size: 0.95rem; opacity: 0.9; margin-bottom: 0.5rem;">{risk_text}</div>
                        <div style="font-size: 1.3rem; font-weight: 600; opacity: 0.95;">{z_indicator} {acwr_z_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 推移グラフ表示
                st.markdown("### 📈 推移グラフ")
                
                # 期間フィルタリング
                start_datetime = pd.to_datetime(start_date)
                end_datetime = pd.to_datetime(selected_date) + pd.Timedelta(days=1)
                
                mask = [(d >= start_datetime and d < end_datetime) for d in dates]
                filtered_dates = [d for d, m in zip(dates, mask) if m]
                
                if filtered_dates:
                    # データをフィルタリング
                    def filter_data(data_list, mask):
                        return [data_list[i] if i < len(data_list) else np.nan for i, m in enumerate(mask) if m]
                    
                    filtered_sleep = filter_data(player_data.get('sleep', []), mask)
                    filtered_weight = filter_data(player_data.get('weight', []), mask)
                    filtered_rpe = filter_data(player_data.get('rpe', []), mask)
                    filtered_srpe = filter_data(player_data.get('srpe', []), mask)
                    filtered_acwr = filter_data(player_data.get('acwr', []), mask)
                    
                    # 全てのZスコアをフィルタリング
                    filtered_sleep_zscore = filter_data(player_data.get('sleep_zscore', []), mask)
                    filtered_weight_zscore = filter_data(player_data.get('weight_zscore', []), mask)
                    filtered_srpe_zscore = filter_data(player_data.get('srpe_zscore', []), mask)
                    
                    # 外傷・障害データもフィルタリング
                    filtered_injury_data = filter_data(player_data.get('injuries', []), mask)
                    
                    # 目標体重データもフィルタリング
                    filtered_target_weight = filter_data(player_data.get('target_weight', []), mask)
                    
                    # フィルタリングされたデータを保存（CSV出力用）
                    filtered_data = {
                        'sleep': filtered_sleep,
                        'weight': filtered_weight,
                        'rpe': filtered_rpe,
                        'activity_time': filter_data(player_data.get('activity_time', []), mask),
                        'srpe': filtered_srpe,
                        'acwr': filtered_acwr,
                        'sleep_zscore': filtered_sleep_zscore,
                        'weight_zscore': filtered_weight_zscore,
                        'srpe_zscore': filtered_srpe_zscore,
                        'acwr_zscore': filter_data(player_data.get('acwr_zscore', []), mask),
                        'injuries': filtered_injury_data,
                        'target_weight': filtered_target_weight
                    }
                    
                    # 改良されたグラフ表示（外傷・障害データ強化版）
                    create_improved_trend_charts_with_all_zscores(
                        filtered_dates, filtered_sleep, filtered_weight, filtered_rpe, 
                        filtered_srpe, filtered_acwr, selected_player, selected_date,
                        filtered_sleep_zscore, filtered_weight_zscore, 
                        filtered_srpe_zscore, None, filtered_injury_data, filtered_target_weight
                    )
                    
                    # 日別データ一覧表を追加
                    st.markdown("### 📋 日別データ一覧")
                    
                    # 表示用データフレームを作成
                    table_data = []
                    for i, date in enumerate(filtered_dates):
                        def format_value(val):
                            if val is not None and pd.notna(val) and val != '':
                                return f"{val:.2f}"
                            return "-"
                        
                        def format_value_1f(val):
                            if val is not None and pd.notna(val) and val != '':
                                return f"{val:.1f}"
                            return "-"
                        
                        # 目標体重との差を計算
                        target_weight = filtered_target_weight[i] if i < len(filtered_target_weight) and filtered_target_weight[i] is not None else None
                        current_weight = filtered_weight[i] if filtered_weight[i] is not None else None
                        weight_diff = None
                        if current_weight is not None and target_weight is not None:
                            weight_diff = current_weight - target_weight
                        
                        # 外傷・障害データを取得
                        injury_count = 0
                        if i < len(filtered_injury_data) and filtered_injury_data[i] is not None:
                            try:
                                injury_count = int(float(filtered_injury_data[i]))
                            except (ValueError, TypeError):
                                injury_count = 0
                        
                        row = {
                            "日付": date.strftime('%Y-%m-%d'),
                            "睡眠時間": format_value(filtered_sleep[i]),
                            "体重": format_value(filtered_weight[i]),
                            "目標体重": format_value(target_weight) if target_weight is not None else "-",
                            "体重差": f"{weight_diff:+.1f}" if weight_diff is not None else "-",
                            "RPE": format_value_1f(filtered_rpe[i]),
                            "sRPE": format_value_1f(filtered_srpe[i]),
                            "ACWR": format_value(filtered_acwr[i]),
                            "睡眠Z": format_value(filtered_sleep_zscore[i] if filtered_sleep_zscore else None),
                            "sRPE Z": format_value(filtered_srpe_zscore[i] if filtered_srpe_zscore else None),
                            "外傷・障害": f"{injury_count}件" if injury_count > 0 else "-"
                        }
                        table_data.append(row)
                    
                    # データフレームを作成して表示
                    df_table = pd.DataFrame(table_data)
                    
                    # スタイル付きで表示
                    st.dataframe(
                        df_table,
                        use_container_width=True,
                        height=400,
                        column_config={
                            "日付": st.column_config.DateColumn(
                                "日付",
                                format="YYYY-MM-DD"
                            ),
                            "睡眠時間": st.column_config.TextColumn("睡眠時間(h)"),
                            "体重": st.column_config.TextColumn("体重(kg)"),
                            "目標体重": st.column_config.TextColumn("目標体重(kg)"),
                            "体重差": st.column_config.TextColumn("目標差(kg)"),
                            "RPE": st.column_config.TextColumn("RPE"),
                            "sRPE": st.column_config.TextColumn("sRPE"),
                            "ACWR": st.column_config.TextColumn("ACWR"),
                            "睡眠Z": st.column_config.TextColumn("睡眠Z値"),
                            "sRPE Z": st.column_config.TextColumn("sRPE Z値"),
                            "外傷・障害": st.column_config.TextColumn("外傷・障害")
                        }
                    )
                    
                    # CSV出力ボタン（個別分析用）
                    st.markdown("### 📥 データエクスポート")
                    
                    # 2つのボタンを横並びで表示
                    col_csv1, col_csv2 = st.columns(2)
                    
                    with col_csv1:
                        # 個別選手のCSV出力
                        csv_individual = create_individual_csv(st.session_state.analyzer, selected_player, filtered_dates, filtered_data)
                        filename = f"{selected_player}_個別データ_{start_date.strftime('%Y%m%d')}_{selected_date.strftime('%Y%m%d')}.csv"
                        
                        st.download_button(
                            label="📥 この選手のデータをCSVでダウンロード",
                            data=csv_individual.to_csv(index=False, encoding='utf-8-sig'),
                            file_name=filename,
                            mime='text/csv',
                            help=f"{selected_player}の{start_date}から{selected_date}までのデータをCSVファイルでダウンロードします"
                        )
                    
                    with col_csv2:
                        # 全選手の一括CSV出力
                        csv_all_players = create_all_players_csv(st.session_state.analyzer, start_date, selected_date)
                        filename_all = f"全選手データ_{start_date.strftime('%Y%m%d')}_{selected_date.strftime('%Y%m%d')}.csv"
                        
                        st.download_button(
                            label="📥 全選手のデータを一括CSVでダウンロード",
                            data=csv_all_players.to_csv(index=False, encoding='utf-8-sig'),
                            file_name=filename_all,
                            mime='text/csv',
                            help=f"全{len(st.session_state.analyzer.players)}選手の{start_date}から{selected_date}までのデータを一括でCSVファイルでダウンロードします",
                            type="secondary"
                        )
                    
                    # 補足情報
                    st.info(f"💡 全選手一括ダウンロードには{len(st.session_state.analyzer.players)}選手分のデータが含まれます（期間: {start_date} ～ {selected_date}）")
        
        with tab2:
            st.header("👥 チーム分析")
            create_team_analysis(st.session_state.analyzer)
            
            # CSV出力ボタン（チーム分析用）
            st.markdown("### 📥 データエクスポート")
            
            col_csv1, col_csv2 = st.columns(2)
            
            with col_csv1:
                # チーム統計データのCSV出力
                csv_team_stats = create_team_comparison_csv(st.session_state.analyzer)
                filename_stats = f"チーム統計データ_{datetime.now().strftime('%Y%m%d')}.csv"
                
                st.download_button(
                    label="📥 チーム統計データをCSVでダウンロード",
                    data=csv_team_stats.to_csv(index=False, encoding='utf-8-sig'),
                    file_name=filename_stats,
                    mime='text/csv',
                    help="各選手の平均値、標準偏差、リスク状況等の統計データをCSVファイルでダウンロードします"
                )
            
            with col_csv2:
                # チーム日別データのCSV出力
                csv_daily_team = create_daily_team_csv(st.session_state.analyzer)
                filename_daily = f"チーム日別データ_{datetime.now().strftime('%Y%m%d')}.csv"
                
                st.download_button(
                    label="📥 チーム日別データをCSVでダウンロード",
                    data=csv_daily_team.to_csv(index=False, encoding='utf-8-sig'),
                    file_name=filename_daily,
                    mime='text/csv',
                    help="各日のチーム平均値、外傷・障害件数等の日別データをCSVファイルでダウンロードします"
                )
        
        with tab3:
            generate_summary_report(st.session_state.analyzer)
            
            # CSV出力ボタン（サマリーレポート用）
            st.markdown("### 📥 データエクスポート")
            
            # サマリーレポートのCSV（選手別統計と同じ）
            csv_summary = create_team_comparison_csv(st.session_state.analyzer)
            filename_summary = f"サマリーレポート_{datetime.now().strftime('%Y%m%d')}.csv"
            
            st.download_button(
                label="📥 サマリーレポートをCSVでダウンロード",
                data=csv_summary.to_csv(index=False, encoding='utf-8-sig'),
                file_name=filename_summary,
                mime='text/csv',
                help="サマリーレポートに表示されている選手別統計データをCSVファイルでダウンロードします"
            )

if __name__ == "__main__":
    main()