"""
대시보드 시각화 모듈
"""
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt  


class Visualizer:
    """
    대시보드 차트 및 시각화를 담당하는 클래스
    """
    
    def __init__(self):
        """
        Visualizer 클래스 초기화
        """
        # 기본 차트 색상
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'buy': '#2ca02c',
            'sell': '#d62728',
            'hold': '#7f7f7f',
            'profit': '#2ca02c',
            'loss': '#d62728',
            'background': '#ffffff',
            'grid': '#e6e6e6'
        }
        
        # 기본 차트 레이아웃
        self.default_layout = {
            'template': 'plotly_white',
            'font': {'family': 'Arial, sans-serif'},
            'xaxis': {'gridcolor': self.colors['grid']},
            'yaxis': {'gridcolor': self.colors['grid']},
            'paper_bgcolor': self.colors['background'],
            'plot_bgcolor': self.colors['background'],
            'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40}
        }
    
    def create_portfolio_value_chart(
        self, 
        portfolio_values: List[float], 
        timestamps: List[str],
        benchmark_values: Optional[List[float]] = None,
        title: str = "포트폴리오 가치 변화"
    ) -> go.Figure:
        """
        포트폴리오 가치 변화 차트 생성
        
        Args:
            portfolio_values: 포트폴리오 가치 목록
            timestamps: 타임스탬프 목록
            benchmark_values: 벤치마크 가치 목록 (옵션)
            title: 차트 제목
            
        Returns:
            Plotly Figure 객체
        """
        fig = go.Figure()
        
        # 포트폴리오 가치 라인 추가
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=portfolio_values,
            mode='lines',
            name='포트폴리오',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        # 벤치마크 추가 (있는 경우)
        if benchmark_values is not None and len(benchmark_values) == len(timestamps):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=benchmark_values,
                mode='lines',
                name='벤치마크',
                line=dict(color=self.colors['secondary'], width=2, dash='dash')
            ))
        
        # 시작점과 종료점 강조
        if len(portfolio_values) > 0:
            fig.add_trace(go.Scatter(
                x=[timestamps[0]],
                y=[portfolio_values[0]],
                mode='markers',
                name='시작',
                marker=dict(color=self.colors['primary'], size=10),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[timestamps[-1]],
                y=[portfolio_values[-1]],
                mode='markers',
                name='종료',
                marker=dict(color=self.colors['primary'], size=10),
                showlegend=False
            ))
        
        # 레이아웃 설정
        layout = self.default_layout.copy()
        layout.update({
            'title': title,
            'xaxis_title': '날짜',
            'yaxis_title': '가치 (원)',
            'hovermode': 'x unified'
        })
        
        fig.update_layout(layout)
        
        return fig
    
    # 수익률 변화 차트
    def create_returns_chart(
        self, 
        returns: List[float], 
        timestamps: List[str],
        benchmark_returns: Optional[List[float]] = None,
        title: str = "수익률 변화"
    ) -> go.Figure:
        """
        수익률 변화 차트 생성
        
        Args:
            returns: 수익률 목록
            timestamps: 타임스탬프 목록
            benchmark_returns: 벤치마크 수익률 목록 (옵션)
            title: 차트 제목
            
        Returns:
            Plotly Figure 객체
        """
        fig = go.Figure()
        
        # 수익률 라인 추가
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[r * 100 for r in returns],  # 퍼센트로 변환
            mode='lines',
            name='전략 수익률',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        # 벤치마크 추가 (있는 경우)
        if benchmark_returns is not None and len(benchmark_returns) == len(timestamps):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=[r * 100 for r in benchmark_returns],  # 퍼센트로 변환
                mode='lines',
                name='벤치마크 수익률',
                line=dict(color=self.colors['secondary'], width=2, dash='dash')
            ))
        
        # 0% 라인 추가
        fig.add_trace(go.Scatter(
            x=[timestamps[0], timestamps[-1]],
            y=[0, 0],
            mode='lines',
            name='기준선',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False
        ))
        
        # 레이아웃 설정
        layout = self.default_layout.copy()
        layout.update({
            'title': title,
            'xaxis_title': '날짜',
            'yaxis_title': '수익률 (%)',
            'hovermode': 'x unified'
        })
        
        fig.update_layout(layout)
        
        return fig
    
    # 위험 대비 수익률 차트
    def create_risk_return_chart(self, results: Dict[str, Any]) -> go.Figure:
        model_ids = []
        sharpe_ratios = []
        drawdowns = []
        total_returns = []

        for model_id, data in results.items():
            metrics = data.get("metrics", {})
            model_ids.append(model_id)
            sharpe_ratios.append(metrics.get("sharpe_ratio", 0))
            drawdowns.append(metrics.get("max_drawdown", 0))
            total_returns.append(metrics.get("total_return", 0))

        # hover text
        hover_texts = [
            f"<b>{mid}</b><br>Sharpe: {s:.2f}<br>Drawdown: {d:.2%}<br>Return: {r:.2f}%"
            for mid, s, d, r in zip(model_ids, sharpe_ratios, drawdowns, total_returns)
        ]

        
        bubble_sizes = [12 + abs(r) * 2 for r in total_returns]

        fig = go.Figure(data=[
            go.Scatter(
                x=drawdowns,
                y=sharpe_ratios,
                mode='markers',
                text=hover_texts,
                hoverinfo='text',
                marker=dict(
                    size=bubble_sizes,
                    color=total_returns,
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title="Total Return (%)"),
                    line=dict(width=1, color='darkgray')  # 버블 외곽선 추가
                )
            )
        ])

        fig.update_layout(
            title="Sharpe vs Max Drawdown",
            xaxis_title="Max Drawdown",
            yaxis_title="Sharpe Ratio",
            template="plotly_white",
            hovermode="closest",
            height=400
        )

        return fig


        fig.update_layout(
            title="위험 대비 수익률 평가 (Sharpe vs Max Drawdown)",
            xaxis_title="Max Drawdown",
            yaxis_title="Sharpe Ratio",
            template="plotly_white",
            hovermode="closest"
        )
        return fig
            
    # 초기자산 vs 최종자산 차트
    def create_initial_vs_final_balance_chart(
        self,
        model_ids: List[str],
        initial_balances: List[float],
        final_balances: List[float],
        title: str = "초기 자산 vs 최종 자산 비교"
    ) -> go.Figure:
        """
        모델별 초기 vs 최종 자산 비교 차트 생성

        Args:
            model_ids: 모델 ID 목록
            initial_balances: 초기 자산 목록
            final_balances: 최종 자산 목록
            title: 차트 제목

        Returns:
            Plotly Figure 객체
        """
        fig = go.Figure()

        # 초기 자산 (회색)
        fig.add_trace(go.Bar(
            x=model_ids,
            y=initial_balances,
            name="초기 자산",
            marker_color="#d1d5db"  # 회색
        ))

        # 최종 자산 (파란색)
        fig.add_trace(go.Bar(
            x=model_ids,
            y=final_balances,
            name="최종 자산",
            marker_color= "#003eb1"  # 파란색
        ))

        layout = self.default_layout.copy()
        layout.update({
            'title': title,
            'xaxis_title': "모델 ID",
            'yaxis_title': "자산 ($)",
            'barmode': 'group'
        })
        
        fig.update_layout(layout)
        return fig

    
    
    # 낙폭 차트
    def create_drawdown_chart(
        self, 
        drawdowns: List[float], 
        timestamps: List[str],
        title: str = "낙폭(Drawdown) 변화"
    ) -> go.Figure:
        """
        낙폭(Drawdown) 차트 생성

        Args:
            drawdowns: 낙폭 목록 (0~1 사이 실수)
            timestamps: 타임스탬프 목록
            title: 차트 제목

        Returns:
            Plotly Figure 객체
        """
        fig = go.Figure()

        # 1. 낙폭 선 그래프
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[d * 100 for d in drawdowns],  # 퍼센트 단위로 변환
            mode='lines',
            name='낙폭',
            line=dict(color=self.colors['danger'], width=2),
            fill='tozeroy',
            fillcolor=f'rgba({int(self.colors["danger"][1:3], 16)}, {int(self.colors["danger"][3:5], 16)}, {int(self.colors["danger"][5:7], 16)}, 0.2)'
        ))

        # 2. 최대 낙폭 표시 (점 + 텍스트 툴팁)
        if drawdowns:
            max_drawdown_idx = np.argmin(drawdowns)
            max_drawdown = drawdowns[max_drawdown_idx]
            max_drawdown_pct = max_drawdown * 100

            fig.add_trace(go.Scatter(
                x=[timestamps[max_drawdown_idx]],
                y=[max_drawdown_pct],
                mode='markers+text',
                name=f'최대 낙폭: {max_drawdown_pct:.2f}%',
                marker=dict(color=self.colors['danger'], size=10),
                text=[f'최대 낙폭: {max_drawdown_pct:.2f}%'],
                textposition='top right'
            ))

        # 3. 레이아웃 설정
        fig.update_layout(
            title=title,
            xaxis_title='날짜',
            yaxis_title='낙폭 (%)',
            hovermode='x unified',
            template='plotly_white'
        )

        return fig
    
    
    # 거래 신호 차트
    def create_trade_chart(
        self, 
        prices: List[float], 
        timestamps: List[str],
        buy_signals: List[Tuple[str, float]],
        sell_signals: List[Tuple[str, float]],
        title: str = "거래 신호"
    ) -> go.Figure:
        """
        거래 신호 차트 생성
        
        Args:
            prices: 가격 목록
            timestamps: 타임스탬프 목록
            buy_signals: 매수 신호 목록 (타임스탬프, 가격)
            sell_signals: 매도 신호 목록 (타임스탬프, 가격)
            title: 차트 제목
            
        Returns:
            Plotly Figure 객체
        """
        fig = go.Figure()
        
        # 가격 라인 추가
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=prices,
            mode='lines',
            name='가격',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        # 매수 신호 추가
        if buy_signals:
            buy_timestamps, buy_prices = zip(*buy_signals)
            fig.add_trace(go.Scatter(
                x=buy_timestamps,
                y=buy_prices,
                mode='markers',
                name='매수 신호',
                marker=dict(color=self.colors['buy'], size=10, symbol='triangle-up')
            ))
        
        # 매도 신호 추가
        if sell_signals:
            sell_timestamps, sell_prices = zip(*sell_signals)
            fig.add_trace(go.Scatter(
                x=sell_timestamps,
                y=sell_prices,
                mode='markers',
                name='매도 신호',
                marker=dict(color=self.colors['sell'], size=10, symbol='triangle-down')
            ))
        
        # 레이아웃 설정
        layout = self.default_layout.copy()
        layout.update({
            'title': title,
            'xaxis_title': '날짜',
            'yaxis_title': '가격',
            'hovermode': 'closest'
        })
        
        fig.update_layout(layout)
        
        return fig
    
    
    
    # 모델별 수익률 비교 차트(백테스트) 
    def create_model_performance_chart(self, data: dict) -> go.Figure:
        """
        모델별 성능 비교용 막대 차트 (대체용)
        
        Args:
            data (dict): {model_id: (initial_value, final_value)} 형식

        Returns:
            go.Figure: Plotly Figure 객체
        """
        if not data:
            return go.Figure()

        model_ids = list(data.keys())
        returns = [
            round(((final - initial) / initial) * 100, 2)
            if initial > 0 else 0
            for (initial, final) in data.values()
        ]

        fig = go.Figure(data=[
            go.Bar(x=model_ids, y=returns, text=[f'{r}%' for r in returns], textposition='auto')
        ])
        
        layout = self.default_layout.copy()
        layout.update({
            'title': 'trading_stats 기반 모델 수익률 (%)',
            'xaxis_title': '모델 ID',
            'yaxis_title': '총 수익률 (%)',
            'margin': dict(l=40, r=40, t=40, b=80)
        })
        
        fig.update_layout(layout)
        return fig
    
    
    # 모델 별 총 거래 수
    def create_trade_count_chart(self, results: Dict[str, Any]) -> go.Figure:
        model_ids = []
        trade_counts = []

        for model_id, data in results.items():
            metrics = data.get("metrics", {})
            model_ids.append(model_id)
            trade_counts.append(metrics.get("total_trades", 0))

        fig = go.Figure(data=[
            go.Bar(
                x=model_ids,
                y=trade_counts,
                marker=dict(
                    color="#93b4d9", 
                    line=dict(color="#436d9d", width=1.2)  # 테두리: 차분한 진파랑
                ),
                hoverinfo="x+y",
                hoverlabel=dict(
                    bgcolor="#e0ecf9",     # hover 배경: 흐린 하늘
                    font_size=13,
                    font_color="#1e293b"   # hover 글씨: 진회색
                )
            )
        ])

        fig.update_layout(
            title="모델 별 총 거래 수",
            xaxis_title="모델 ID",
            yaxis_title="총 거래 수",
            template="plotly_white",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#111111", size=14),
            bargap=0.45,  # 얇게 유지
            xaxis=dict(
                tickangle=-45,
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#e5e7eb"
            )
        )

        return fig



    
    # 모델별 수익률 비교 차트(백테스트)
    def create_performance_comparison_chart(
        self,
        metrics: Dict[str, Dict[str, float]],
        metric_name: str,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        모델 성능 비교 차트 (수익률 높은 모델일수록 진한 파랑)
        """
        models = []
        values = []

        for model_id, model_metrics in metrics.items():
            if metric_name in model_metrics:
                models.append(model_id)
                values.append(model_metrics[metric_name])

        if not models:
            fig = go.Figure()
            fig.update_layout(
                title=f"데이터 없음: {metric_name}",
                xaxis_title="모델",
                yaxis_title=metric_name
            )
            return fig

        # 수익률 내림차순 정렬
        sorted_indices = np.argsort(values)[::-1]
        sorted_models = [models[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]

        # 수익률이 클수록 진한 파랑 (Blues colormap)
        cmap = plt.get_cmap('Blues')
        num_items = len(sorted_values)
        colors = [f"rgba{cmap(0.3 + 0.7 * (1 - i / max(num_items-1, 1)))[:3] + (1.0,)}" for i in range(num_items)]
        # 위에서부터 진한 → 아래로 연한 파랑이 되게 인덱스 순서 뒤집음

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=sorted_models,
            x=sorted_values,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.1f}%" for v in sorted_values],
            textposition="outside",
            hoverinfo="x+y"
        ))

        fig.update_layout(
            title=title if title else f"{metric_name} 모델 비교",
            xaxis_title=metric_name,
            yaxis_title="모델 ID",
            template="plotly_white",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#111111", size=14),
            margin=dict(l=100, r=40, t=60, b=40),
            xaxis=dict(showgrid=True, zeroline=False),
            yaxis=dict(autorange="reversed")
        )

        return fig
    
    # 모델별 성능 비교용 막대 차트 
    def build_model_performance_chart(data: dict) -> dict:
        """
        모델별 성능 비교용 막대 차트 
        
        Args:
            data (dict): {model_id: (initial_value, final_value)} 형식

        Returns:
            dict: Plotly JSON format { 'data': [...], 'layout': {...} }
        """
        if not data:
            return {'data': [], 'layout': {}}

        model_ids = list(data.keys())
        returns = [
                round(final, 2) if initial == 0 else round(((final - initial) / initial) * 100, 2)
                for (initial, final) in data.values()
            ]


        fig = go.Figure(data=[
            go.Bar(x=model_ids, y=returns, text=[f'{r}%' for r in returns], textposition='auto')
        ])
        fig.update_layout(
            title='trading_stats 기반 모델 수익률 (%)',
            xaxis_title='모델 ID',
            yaxis_title='총 수익률 (%)',
            yaxis=dict(range=[-1, 1]),  # Y축 범위 명시: 수동 설정
            bargap=0.4,
            margin=dict(l=40, r=40, t=40, b=80),
            height=350
        )

        return fig.to_dict() 
    
    # 매수/매도 비율 바차트
    def create_trade_buy_sell_chart_from_aggregate(
        self,
        side_counts: List[Dict[str, Any]],
        title: str = "매수 vs 매도 비율"
    ) -> go.Figure:
        side_dict = {row['side'].upper(): row['count'] for row in side_counts}

        buy_count = side_dict.get('BUY', 0)
        sell_count = side_dict.get('SELL', 0)
        total = buy_count + sell_count

        if total == 0:
            fig = go.Figure()
            fig.update_layout(
                title="데이터 없음: 매수/매도 비율",
                xaxis_title="거래 유형",
                yaxis_title="거래 수"
            )
            return fig

        fig = go.Figure(data=[
            go.Bar(
                x=["BUY", "SELL"],
                y=[buy_count, sell_count],
                marker=dict(
                    color=["#A3C2FE", "#66E9CA"],
                    line=dict(color=["#A3C2FE", "#66E9CA"], width=1.2)
                ),
                name="거래 수",
                width=0.3
            )
        ])

        fig.add_annotation(
            x=0.95, y=0.95,
            xref="paper", yref="paper",
            text=(
                f"총 거래: {total}건<br>"
                f"BUY: {buy_count}건 ({buy_count/total:.2%})<br>"
                f"SELL: {sell_count}건 ({sell_count/total:.2%})"
            ),
            showarrow=False,
            font=dict(size=12, color="#1f2937"),
            align="right",
            bgcolor="rgba(243, 244, 246, 0.9)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4
        )

        layout = self.default_layout.copy()
        layout.update({
            'title': title,
            'xaxis_title': "거래 유형",
            'yaxis_title': "거래 수",
            'bargap': 0.4,
            'plot_bgcolor': "#ffffff",
            'paper_bgcolor': "#ffffff",
            'font': dict(color="#1f2937", size=14)
        })

        fig.update_layout(**layout)
        return fig



    # 모델 성능 비교 레이더 차트
    def create_model_comparison_radar_chart(
        self, 
        metrics: Dict[str, Dict[str, float]],
        metric_names: List[str],
        model_ids: Optional[List[str]] = None,
        title: str = "모델 성능 비교"
    ) -> go.Figure:
        """
        모델 성능 비교 레이더 차트 생성
        
        Args:
            metrics: 모델별 성능 지표 딕셔너리
            metric_names: 비교할 지표 이름 목록
            model_ids: 비교할 모델 ID 목록 (옵션, 없으면 모든 모델 비교)
            title: 차트 제목
            
        Returns:
            Plotly Figure 객체
        """
        if model_ids is None:
            model_ids = list(metrics.keys())
        
        # 최대 5개 모델만 비교
        if len(model_ids) > 5:
            model_ids = model_ids[:5]
        
        # 레이더 차트 생성
        fig = go.Figure()
        
        # 각 모델별 데이터 추가
        for i, model_id in enumerate(model_ids):
            if model_id not in metrics:
                continue
                
            model_metrics = metrics[model_id]
            
            # 선택한 지표만 사용
            values = []
            for metric in metric_names:
                values.append(model_metrics.get(metric, 0))
            
            # 첫 값을 마지막에 다시 추가하여 레이더 차트 닫기
            values.append(values[0])
            labels = metric_names + [metric_names[0]]
            
            # 색상 선택
            color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name=model_id,
                line_color=color
            ))
        
        # 레이아웃 설정
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]  # 정규화된 값 사용
                )
            ),
            title=title,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def create_live_trading_dashboard(
        self, 
        trading_stats: Dict[str, Any],
        price_data: pd.DataFrame,
        trades: List[Dict[str, Any]]
    ) -> Dict[str, go.Figure]:
        """
        실시간 트레이딩 대시보드 차트 생성
        
        Args:
            trading_stats: 트레이딩 통계 정보
            price_data: 가격 데이터
            trades: 거래 내역
            
        Returns:
            차트 목록 딕셔너리
        """
        charts = {}
        
        # 포트폴리오 가치 차트
        if 'portfolio_values' in trading_stats and 'timestamps' in trading_stats:
            charts['portfolio'] = self.create_portfolio_value_chart(
                portfolio_values=trading_stats['portfolio_values'],
                timestamps=trading_stats['timestamps'],
                title="실시간 포트폴리오 가치"
            )
        
        # 수익률 차트
        if 'returns' in trading_stats and 'timestamps' in trading_stats:
            charts['returns'] = self.create_returns_chart(
                returns=trading_stats['returns'],
                timestamps=trading_stats['timestamps'],
                title="실시간 수익률 변화"
            )
        
        # 가격 및 거래 차트
        if not price_data.empty and 'date' in price_data.columns and 'close' in price_data.columns:
            # 매수/매도 신호 추출
            buy_signals = []
            sell_signals = []
            
            for trade in trades:
                if 'timestamp' in trade and 'price' in trade:
                    if trade.get('side') == 'buy':
                        buy_signals.append((trade['timestamp'], trade['price']))
                    elif trade.get('side') == 'sell':
                        sell_signals.append((trade['timestamp'], trade['price']))
            
            charts['price'] = self.create_trade_chart(
                prices=price_data['close'].tolist(),
                timestamps=price_data['date'].astype(str).tolist(),
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                title="가격 차트 및 거래 신호"
            )
        
        return charts