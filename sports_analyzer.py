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
            
            # Sheet2の処理（外傷・障害データ）
            injury_data = {}
            if 'Sheet2' in df:
                sheet2 = df['Sheet2']
                # Sheet2の1行目は日付、2行目以降は選手データ
                for row_idx in range(1, len(sheet2)):
                    if pd.notna(sheet2.iloc[row_idx, 0]):
                        player_name = sheet2.iloc[row_idx, 0]
                        # 日付列（1列目以降）から外傷・障害データを取得
                        injury_counts = sheet2.iloc[row_idx, 1:1+len(self.dates)].tolist()
                        # NaNや空文字を0に置換
                        injury_counts = [int(x) if pd.notna(x) and x != '' else 0 for x in injury_counts]
                        injury_data[player_name] = injury_counts
            
            # 選手データに外傷・障害データを追加
            for player in players_data:
                if player in injury_data:
                    players_data[player]['injuries'] = injury_data[player]
                else:
                    # 外傷・障害データがない場合は0で埋める
                    players_data[player]['injuries'] = [0] * len(self.dates)
            
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
                                                 sleep_zscore, weight_zscore, srpe_zscore, acwr_zscore, injury_data=None):
    """全てのZスコアを含む改良された推移チャート"""
    
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
    
    # 外傷・障害データの処理
    if injury_data:
        injury_data = [int(x) if pd.notna(x) and x != '' and x != 0 else 0 for x in injury_data]
    else:
        injury_data = [0] * len(dates)
    
    # デバッグ用：データの長さを確認
    st.write(f"デバッグ情報 - データ長: dates={len(dates)}, sleep={len(sleep_data)}, weight={len(weight_data)}")
    
    # 5行1列のサブプロットを作成（元のまま）
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
                connectgaps=True,  # 空欄があってもグラフを繋げる
            ),
            row=1, col=1
        )
    
    # 2. 体重（修正版）
    if weight_data and any(x is not None for x in weight_data):
        # 有効なデータポイントのみを取得
        valid_dates = []
        valid_weights = []
        for i, (date, weight) in enumerate(zip(dates, weight_data)):
            if weight is not None:
                valid_dates.append(date)
                valid_weights.append(weight)
        
        st.write(f"デバッグ情報 - 体重の有効データ数: {len(valid_weights)}")
        
        if valid_weights:  # 有効なデータが存在する場合のみグラフを描画
            fig.add_trace(
                go.Scatter(
                    x=dates, y=weight_data, 
                    name='体重',
                    line=dict(color=colors['weight'], width=4, shape='spline', smoothing=0.8),
                    marker=dict(size=8, line=dict(width=2, color='white'), symbol='circle'),
                    hovertemplate='<b>体重</b><br>日付: %{x}<br>値: %{y:.2f}kg<extra></extra>',
                    connectgaps=True,  # これが重要：空欄があっても線を繋げる
                    mode='lines+markers'  # 明示的にモードを指定
                ),
                row=2, col=1
            )
    
    # 3. sRPE（外傷・障害マーカー付き）
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
        
        # 外傷・障害を棒グラフで表示（sRPEグラフ）
        if injury_data and any(count > 0 for count in injury_data):
            # 最大sRPE値を取得してスケール調整
            max_srpe = max([val for val in srpe_data if val is not None]) if any(val is not None for val in srpe_data) else 1
            
            # 外傷・障害データを棒グラフ用に調整
            injury_bar_data = []
            for count in injury_data:
                if count > 0:
                    # 外傷・障害の棒の高さをsRPEの30%程度に設定
                    injury_bar_data.append(max_srpe * 0.3 * count)
                else:
                    injury_bar_data.append(0)
            
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=injury_bar_data,
                    name='外傷・障害',
                    marker=dict(color=colors['injury'], opacity=0.7),
                    hovertemplate='<b>外傷・障害</b><br>日付: %{x}<br>件数: %{customdata}<extra></extra>',
                    customdata=injury_data,
                    width=86400000 * 0.6  # 棒の幅調整
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
    
    # 5. sRPE Zスコア（外傷・障害マーカー付き）
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
        
        # 外傷・障害を棒グラフで表示（sRPE Zスコアグラフ）
        if injury_data and any(count > 0 for count in injury_data):
            # Zスコアの範囲に合わせて棒の高さを調整
            injury_z_bar_data = []
            for count in injury_data:
                if count > 0:
                    # 外傷・障害の棒の高さを固定値（Zスコア範囲内）
                    injury_z_bar_data.append(0.5 * count)  # Zスコア0.5の高さ × 件数
                else:
                    injury_z_bar_data.append(0)
            
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=injury_z_bar_data,
                    name='外傷・障害 (Z)',
                    marker=dict(color=colors['injury'], opacity=0.7),
                    hovertemplate='<b>外傷・障害</b><br>日付: %{x}<br>件数: %{customdata}<extra></extra>',
                    customdata=injury_data,
                    showlegend=False,
                    width=86400000 * 0.6  # 棒の幅調整
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

def create_team_comparison(analyzer):
    """チーム全体の比較グラフを作成（Plotly版）"""
    if not analyzer.data:
        st.error("データが読み込まれていません")
        return
    
    # 統一されたシックなカラーパレット
    colors = {
        'sleep': '#2C3E50',
        'weight': '#2C3E50',     # 他と統一
        'srpe': '#34495E',
        'acwr': '#2C3E50',
        'injury': '#E74C3C'
    }
    
    # 各選手の統計値を取得
    player_stats = {}
    for player in analyzer.players:
        player_data = analyzer.data['players'][player]
        
        # 最新の値を取得（欠損値を除く）
        sleep_values = [x for x in player_data['sleep'] if pd.notna(x) and x != '']
        weight_values = [x for x in player_data['weight'] if pd.notna(x) and x != '']
        srpe_values = [x for x in player_data['srpe'] if pd.notna(x) and x != '']
        acwr_values = [x for x in player_data['acwr'] if pd.notna(x) and x != '']
        
        player_stats[player] = {
            'avg_sleep': np.mean(sleep_values) if sleep_values else 0,
            'avg_weight': np.mean(weight_values) if weight_values else 0,
            'avg_srpe': np.mean(srpe_values) if srpe_values else 0,
            'latest_acwr': acwr_values[-1] if acwr_values else 0
        }
    
    # 2行2列のサブプロットを作成（上部に比較グラフ、下部にZスコア推移）
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['平均睡眠時間', '平均体重', '平均sRPE', '最新ACWR', '', 'チーム平均sRPE Zスコア推移'],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}],
               [{"colspan": 2}, None]],
        vertical_spacing=0.12
    )
    
    players = list(player_stats.keys())
    player_names = [p.split()[-1] if ' ' in p else p for p in players]
    
    # 1. 平均睡眠時間
    sleep_avgs = [player_stats[p]['avg_sleep'] for p in players]
    fig.add_trace(
        go.Bar(x=player_names, y=sleep_avgs, name='睡眠時間', 
               marker_color=colors['sleep'], text=[f'{v:.1f}' for v in sleep_avgs],
               textposition='outside'),
        row=1, col=1
    )
    
    # 2. 平均体重
    weight_avgs = [player_stats[p]['avg_weight'] for p in players]
    fig.add_trace(
        go.Bar(x=player_names, y=weight_avgs, name='体重', 
               marker_color=colors['weight'], text=[f'{v:.1f}' for v in weight_avgs],
               textposition='outside'),
        row=1, col=2
    )
    
    # 3. 平均sRPE
    srpe_avgs = [player_stats[p]['avg_srpe'] for p in players]
    fig.add_trace(
        go.Bar(x=player_names, y=srpe_avgs, name='sRPE', 
               marker_color=colors['srpe'], text=[f'{v:.1f}' for v in srpe_avgs],
               textposition='outside'),
        row=2, col=1
    )
    
    # 4. 最新ACWR
    acwr_latest = [player_stats[p]['latest_acwr'] for p in players]
    acwr_colors = []
    for x in acwr_latest:
        if x > 1.3:
            acwr_colors.append('#E74C3C')  # 赤
        elif x < 0.8:
            acwr_colors.append('#F39C12')  # オレンジ
        else:
            acwr_colors.append('#27AE60')  # 緑
    
    fig.add_trace(
        go.Bar(x=player_names, y=acwr_latest, name='ACWR', 
               marker_color=acwr_colors, text=[f'{v:.2f}' if v > 0 else '' for v in acwr_latest],
               textposition='outside'),
        row=2, col=2
    )
    
    # ACWR理想値と境界線を追加
    fig.add_hline(y=1.0, line_dash="dash", line_color="blue", 
                 annotation_text="理想値", row=2, col=2)
    fig.add_hline(y=0.8, line_dash="dot", line_color="orange", 
                 annotation_text="低リスク境界", row=2, col=2)
    fig.add_hline(y=1.3, line_dash="dot", line_color="red", 
                 annotation_text="高リスク境界", row=2, col=2)
    
    # 5. チーム平均sRPE Zスコア推移を追加
    dates = analyzer.dates
    
    # 各日のチーム平均sRPE Zスコアを計算
    team_avg_zscore = []
    daily_injury_totals = []
    for day_idx in range(len(dates)):
        day_zscores = []
        daily_injuries = 0
        for player in analyzer.players:
            player_data = analyzer.data['players'][player]
            if day_idx < len(player_data.get('srpe_zscore', [])):
                zscore = player_data['srpe_zscore'][day_idx]
                if pd.notna(zscore):
                    day_zscores.append(zscore)
            
            # 外傷・障害データを集計
            if day_idx < len(player_data.get('injuries', [])):
                injury_count = player_data['injuries'][day_idx]
                if pd.notna(injury_count) and injury_count > 0:
                    daily_injuries += injury_count
        
        if day_zscores:
            team_avg_zscore.append(np.mean(day_zscores))
        else:
            team_avg_zscore.append(None)
        
        daily_injury_totals.append(daily_injuries)
    
    # チーム平均Zスコア推移グラフ
    fig.add_trace(
        go.Scatter(
            x=dates, 
            y=team_avg_zscore,
            name='チーム平均sRPE Z-score',
            line=dict(color='#E67E22', width=4, shape='spline', smoothing=0.8),
            marker=dict(size=8, line=dict(width=2, color='white'), symbol='circle'),
            hovertemplate='<b>チーム平均sRPE Z-score</b><br>日付: %{x}<br>値: %{y:.2f}<extra></extra>',
            connectgaps=True
        ),
        row=3, col=1
    )
    
    # 外傷・障害発生日にマーカーを追加
    injury_dates = []
    injury_zscore_values = []
    for date, injury_count, zscore_val in zip(dates, daily_injury_totals, team_avg_zscore):
        if injury_count > 0 and zscore_val is not None:
            injury_dates.append(date)
            injury_zscore_values.append(zscore_val)
    
    # 外傷・障害を棒グラフで表示
    if injury_dates:
        fig.add_trace(
            go.Bar(
                x=injury_dates, 
                y=[1] * len(injury_dates),  # 固定の高さ
                name='外傷・障害発生日',
                marker=dict(color='#E74C3C', opacity=0.6),
                hovertemplate='<b>チーム外傷・障害発生</b><br>日付: %{x}<br>件数: 1<extra></extra>',
                yaxis='y4'
            ),
            row=3, col=1
        )
    
    # Zスコアの基準線を追加
    fig.add_hline(y=0, line_dash="solid", line_color="#7F8C8D", line_width=2,
                 annotation_text="平均値 (Z=0)", row=3, col=1)
    fig.add_hline(y=1, line_dash="dot", line_color="#27AE60", 
                 annotation_text="標準偏差+1", row=3, col=1)
    fig.add_hline(y=-1, line_dash="dot", line_color="#27AE60", 
                 annotation_text="標準偏差-1", row=3, col=1)
    fig.add_hline(y=2, line_dash="dot", line_color="#E74C3C", 
                 annotation_text="標準偏差+2", row=3, col=1)
    fig.add_hline(y=-2, line_dash="dot", line_color="#E74C3C", 
                 annotation_text="標準偏差-2", row=3, col=1)
    
    # Y軸ラベルを設定
    fig.update_yaxes(title_text="時間", row=1, col=1)
    fig.update_yaxes(title_text="kg", row=1, col=2)
    fig.update_yaxes(title_text="sRPE", row=2, col=1)
    fig.update_yaxes(title_text="ACWR", row=2, col=2)
    fig.update_yaxes(
        title_text="Z-score", 
        row=3, col=1,
        range=[-3, 3],  # Zスコアの範囲を-3から3に固定
        gridcolor='rgba(52, 73, 94, 0.1)',
        linecolor='rgba(52, 73, 94, 0.3)'
    )
    
    fig.update_layout(
        height=1200, 
        title_text="チーム比較分析 & sRPE負荷推移", 
        title_x=0.5, 
        showlegend=False,
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        font=dict(family="Arial", color='#2C3E50')
    )
    
    # X軸の設定
    fig.update_xaxes(
        gridcolor='rgba(52, 73, 94, 0.1)',
        linecolor='rgba(52, 73, 94, 0.3)',
        tickfont=dict(color='#2C3E50')
    )
    
    st.plotly_chart(fig, use_container_width=True)

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
            "リスクステータス": risk_status
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
            "外傷障害件数": format_value(filtered_data['injuries'][i] if i < len(filtered_data['injuries']) else 0)
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
            "総外傷障害件数": sum(injury_values) if injury_values else 0,
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
                if pd.notna(injury_count) and injury_count > 0:
                    daily_injuries += injury_count
        
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
        """)
        
    else:
        # タブ作成
        tab1, tab2, tab3 = st.tabs(["👤 個別分析", "👥 チーム比較", "📊 サマリーレポート"])
        
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
                    weight_z_str = f"Z: {weight_z:.2f}" if weight_z is not None else "Z: N/A"
                    z_indicator = "🟢" if weight_z is not None and abs(weight_z) <= 1 else "🔴" if weight_z is not None else "⚪"
                    
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
                        <div style="font-size: 1.3rem; font-weight: 600; opacity: 0.95; margin-top: 0.8rem;">{z_indicator} {weight_z_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    rpe_str = f"{rpe_val:.1f}" if rpe_val is not None else "N/A"
                    srpe_str = f"sRPE: {srpe_val:.1f}" if srpe_val is not None else "sRPE: N/A"
                    srpe_z_str = f"Z: {srpe_z:.2f}" if srpe_z is not None else "Z: N/A"
                    z_indicator = "🟢" if srpe_z is not None and abs(srpe_z) <= 1 else "🔴" if srpe_z is not None else "⚪"
                    
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
                    
                    # 全てのZスコアをフィルタリング（ACWR Zスコアを除去）
                    filtered_sleep_zscore = filter_data(player_data.get('sleep_zscore', []), mask)
                    filtered_weight_zscore = filter_data(player_data.get('weight_zscore', []), mask)
                    filtered_srpe_zscore = filter_data(player_data.get('srpe_zscore', []), mask)
                    
                    # 外傷・障害データもフィルタリング
                    filtered_injury_data = filter_data(player_data.get('injuries', []), mask)
                    
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
                        'injuries': filtered_injury_data
                    }
                    
                    # 改良されたグラフ表示（ACWR Zスコアを除去）
                    create_improved_trend_charts_with_all_zscores(
                        filtered_dates, filtered_sleep, filtered_weight, filtered_rpe, 
                        filtered_srpe, filtered_acwr, selected_player, selected_date,
                        filtered_sleep_zscore, filtered_weight_zscore, 
                        filtered_srpe_zscore, None, filtered_injury_data  # acwr_zscoreをNoneに
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
                        
                        row = {
                            "日付": date.strftime('%Y-%m-%d'),
                            "睡眠時間": format_value(filtered_sleep[i]),
                            "体重": format_value(filtered_weight[i]),
                            "RPE": format_value_1f(filtered_rpe[i]),
                            "sRPE": format_value_1f(filtered_srpe[i]),
                            "ACWR": format_value(filtered_acwr[i]),
                            "睡眠Z": format_value(filtered_sleep_zscore[i] if filtered_sleep_zscore else None),
                            "体重Z": format_value(filtered_weight_zscore[i] if filtered_weight_zscore else None),
                            "sRPE Z": format_value(filtered_srpe_zscore[i] if filtered_srpe_zscore else None)
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
                            "RPE": st.column_config.TextColumn("RPE"),
                            "sRPE": st.column_config.TextColumn("sRPE"),
                            "ACWR": st.column_config.TextColumn("ACWR"),
                            "睡眠Z": st.column_config.TextColumn("睡眠Z値"),
                            "体重Z": st.column_config.TextColumn("体重Z値"),
                            "sRPE Z": st.column_config.TextColumn("sRPE Z値")
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
            st.header("👥 チーム比較分析")
            create_team_comparison(st.session_state.analyzer)
            
            # CSV出力ボタン（チーム比較用）
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