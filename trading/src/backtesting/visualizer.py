import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import json


class Visualizer:
    """
    백테스트 결과를 시각화하기 위한 클래스
    """
    
    def __init__(self, results: Dict[str, Any], save_dir: Optional[str] = None):
        """
        Visualizer 클래스 초기화
        
        Args:
            results: 백테스트 결과 딕셔너리
            save_dir: 시각화 결과를 저장할 디렉토리 경로 (옵션)
        """
        self.results = results
        self.save_dir = save_dir
        
        # 저장 디렉토리 생성
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        
        # 시각화 스타일 설정
        sns.set_style("whitegrid")
        
    def plot_equity_curve(self, save_path: Optional[str] = None) -> None:
        """
        자본금 곡선(Equity Curve) 시각화
        
        Args:
            save_path: 그래프를 저장할 파일 경로 (옵션)
        """
        plt.figure(figsize=(14, 7))
        
        # 포트폴리오 가치 플롯
        plt.plot(self.results["portfolio_values"], label="Portfolio Value", color="royalblue", linewidth=2)
        
        # 매수/매도 지점 표시
        for trade in self.results["trades"]:
            idx = self.results["timestamps"].index(trade["timestamp"])
            
            if trade["action"] > 0:  # 매수
                plt.scatter(idx, self.results["portfolio_values"][idx], color="green", marker="^", s=100)
            elif trade["action"] < 0:  # 매도
                plt.scatter(idx, self.results["portfolio_values"][idx], color="red", marker="v", s=100)
        
        # 성능 지표 텍스트
        metrics_text = f"""
        Initial Balance: ${self.results.get('initial_balance', 10000):.2f}
        Final Value: ${self.results["portfolio_values"][-1]:.2f}
        Total Return: {self.results["metrics"]["cumulative_return"]:.2%}
        Annual Return: {self.results["metrics"]["annual_return"]:.2%}
        Sharpe Ratio: {self.results["metrics"]["sharpe_ratio"]:.2f}
        Max Drawdown: {self.results["metrics"]["max_drawdown"]:.2%}
        """
        
        plt.text(0.02, 0.97, metrics_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.3))
        
        # 그래프 레이블 및 제목
        plt.title("Portfolio Equity Curve", fontsize=16)
        plt.xlabel("Trading Days", fontsize=12)
        plt.ylabel("Portfolio Value ($)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        # 저장 또는 표시
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    def plot_returns_comparison(self, benchmark_returns: Optional[np.ndarray] = None, save_path: Optional[str] = None) -> None:
        """
        전략 수익률과 벤치마크 수익률 비교 시각화
        
        Args:
            benchmark_returns: 벤치마크 수익률 배열 (옵션)
            save_path: 그래프를 저장할 파일 경로 (옵션)
        """
        plt.figure(figsize=(14, 7))
        
        # 전략 누적 수익률 계산
        strategy_returns = np.array(self.results["returns"])
        strategy_cum_returns = np.cumprod(1 + strategy_returns) - 1
        
        # 전략 누적 수익률 플롯
        plt.plot(strategy_cum_returns, label="Strategy", color="royalblue", linewidth=2)
        
        # 벤치마크 수익률이 있으면 함께 플롯
        if benchmark_returns is not None:
            benchmark_cum_returns = np.cumprod(1 + benchmark_returns) - 1
            plt.plot(benchmark_cum_returns, label="Benchmark", color="gray", linewidth=2, linestyle="--")
            
            # 알파, 베타 계산
            excess_returns = strategy_returns - benchmark_returns
            annual_alpha = excess_returns.mean() * 252
            beta = np.cov(strategy_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
            
            # 정보 비율(Information Ratio) 계산
            tracking_error = np.std(excess_returns) * np.sqrt(252)
            info_ratio = annual_alpha / tracking_error if tracking_error != 0 else 0
            
            # 비교 지표 텍스트
            comparison_text = f"""
            Alpha (annual): {annual_alpha:.2%}
            Beta: {beta:.2f}
            Information Ratio: {info_ratio:.2f}
            """
            
            plt.text(0.02, 0.7, comparison_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.3))
        
        # 그래프 레이블 및 제목
        plt.title("Cumulative Returns Comparison", fontsize=16)
        plt.xlabel("Trading Days", fontsize=12)
        plt.ylabel("Cumulative Return", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.tight_layout()
        
        # 저장 또는 표시
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
            
    def plot_monthly_returns_heatmap(self, save_path: Optional[str] = None) -> None:
        """
        월별 수익률 히트맵 시각화
        
        Args:
            save_path: 그래프를 저장할 파일 경로 (옵션)
        """
        # 타임스탬프와 수익률 데이터가 있는지 확인
        if not self.results.get("timestamps") or not self.results.get("returns"):
            print("월별 수익률 히트맵을 생성하기 위한 타임스탬프 또는 수익률 데이터가 없습니다.")
            return
            
        # 데이터프레임 생성
        try:
            df = pd.DataFrame({
                'date': pd.to_datetime(self.results["timestamps"]),
                'return': self.results["returns"]
            })
            
            # 월별 수익률 계산
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            monthly_returns = df.groupby(['year', 'month'])['return'].sum().unstack()
            
            plt.figure(figsize=(14, 8))
            
            # 히트맵 생성
            ax = sns.heatmap(monthly_returns, annot=True, fmt=".2%", cmap="RdYlGn", center=0,
                           linewidths=1, cbar_kws={'label': 'Monthly Return'})
            
            # 그래프 레이블 및 제목
            plt.title("Monthly Returns (%)", fontsize=16)
            plt.xlabel("")
            plt.ylabel("")
            
            # 월 이름으로 레이블 변경
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticklabels(month_names, rotation=0)
            
            plt.tight_layout()
            
            # 저장 또는 표시
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            else:
                plt.show()
        except Exception as e:
            print(f"월별 수익률 히트맵 생성 중 오류 발생: {e}")
            
    def plot_drawdown_periods(self, top_n: int = 5, save_path: Optional[str] = None) -> None:
        """
        주요 낙폭 기간 시각화
        
        Args:
            top_n: 시각화할 최대 낙폭 기간 수
            save_path: 그래프를 저장할 파일 경로 (옵션)
        """
        plt.figure(figsize=(14, 10))
        
        # 포트폴리오 가치 및 낙폭 계산
        portfolio_values = pd.Series(self.results["portfolio_values"])
        drawdown = (portfolio_values / portfolio_values.cummax() - 1) * 100
        
        # 전체 낙폭 플롯
        plt.subplot(2, 1, 1)
        plt.plot(portfolio_values, label="Portfolio Value", color="royalblue")
        plt.title("Portfolio Value Over Time", fontsize=14)
        plt.xlabel("Trading Days")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 낙폭 플롯
        plt.subplot(2, 1, 2)
        plt.fill_between(range(len(drawdown)), 0, drawdown, color="crimson", alpha=0.3)
        plt.plot(drawdown, color="crimson", label="Drawdown %")
        
        # 주요 낙폭 기간 찾기
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0
        current_drawdown = 0
        
        for i, dd in enumerate(drawdown):
            if not in_drawdown and dd < -1:  # 1% 이상 낙폭 시작
                in_drawdown = True
                start_idx = i
                current_drawdown = dd
            elif in_drawdown:
                if dd < current_drawdown:  # 낙폭 깊어짐
                    current_drawdown = dd
                elif dd >= -1:  # 낙폭 종료
                    in_drawdown = False
                    drawdown_periods.append({
                        'start': start_idx,
                        'end': i,
                        'max_drawdown': current_drawdown,
                        'duration': i - start_idx
                    })
        
        # 현재 진행 중인 낙폭 추가
        if in_drawdown:
            drawdown_periods.append({
                'start': start_idx,
                'end': len(drawdown) - 1,
                'max_drawdown': current_drawdown,
                'duration': len(drawdown) - 1 - start_idx
            })
        
        # 낙폭 크기 기준 정렬 후 상위 N개 표시
        drawdown_periods.sort(key=lambda x: x['max_drawdown'])
        top_drawdowns = drawdown_periods[:min(top_n, len(drawdown_periods))]
        
        # 주요 낙폭 기간 하이라이트
        colors = ['orange', 'green', 'purple', 'blue', 'brown']
        for i, dd_period in enumerate(top_drawdowns):
            color = colors[i % len(colors)]
            plt.axvspan(dd_period['start'], dd_period['end'], alpha=0.2, color=color, 
                        label=f"DD {i+1}: {dd_period['max_drawdown']:.2f}% ({dd_period['duration']} days)")
        
        plt.title("Drawdown Periods", fontsize=14)
        plt.xlabel("Trading Days")
        plt.ylabel("Drawdown (%)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right")
        plt.ylim(min(drawdown) * 1.1, 1)
        
        plt.tight_layout()
        
        # 저장 또는 표시
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
            
    def plot_trade_analysis(self, save_path: Optional[str] = None) -> None:
        """
        거래 분석 시각화
        
        Args:
            save_path: 그래프를 저장할 파일 경로 (옵션)
        """
        if not self.results.get("trades") or len(self.results["trades"]) == 0:
            print("거래 분석을 위한 거래 데이터가 없습니다.")
            return
            
        plt.figure(figsize=(14, 12))
        
        # 거래 데이터 추출
        trades = pd.DataFrame(self.results["trades"])
        
        # 1. 거래 타입 분포 (매수/매도)
        plt.subplot(2, 2, 1)
        trade_types = ['Buy' if t['action'] > 0 else 'Sell' if t['action'] < 0 else 'Hold' for t in self.results["trades"]]
        sns.countplot(x=trade_types)
        plt.title("Trade Type Distribution", fontsize=14)
        plt.xlabel("")
        plt.ylabel("Count")
        
        # 2. 거래 수익률 분포
        if 'trade_return' in trades.columns:
            plt.subplot(2, 2, 2)
            sns.histplot(trades['trade_return'], kde=True, color="royalblue")
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
            plt.title("Trade Returns Distribution", fontsize=14)
            plt.xlabel("Trade Return")
            plt.ylabel("Frequency")
        
        # 3. 포지션 보유 기간 분포
        if len(self.results["trades"]) > 1:
            try:
                # 포지션 보유 기간 계산
                holding_periods = []
                position = 0
                position_start = 0
                
                for i, trade in enumerate(self.results["trades"]):
                    idx = self.results["timestamps"].index(trade["timestamp"])
                    
                    if position == 0 and trade["action"] != 0:  # 새 포지션 시작
                        position = 1 if trade["action"] > 0 else -1
                        position_start = idx
                    elif position != 0 and (trade["action"] * position < 0 or trade["position"] == 0):  # 포지션 종료
                        holding_periods.append(idx - position_start)
                        position = 0
                
                plt.subplot(2, 2, 3)
                sns.histplot(holding_periods, kde=False, color="green")
                plt.title("Position Holding Period Distribution", fontsize=14)
                plt.xlabel("Holding Period (days)")
                plt.ylabel("Frequency")
            except Exception as e:
                print(f"포지션 보유 기간 계산 중 오류 발생: {e}")
        
        # 4. 수익/손실 거래 비율
        plt.subplot(2, 2, 4)
        win_rate = self.results["metrics"]["win_rate"]
        plt.pie([win_rate, 1-win_rate], labels=['Profitable', 'Loss'], colors=['green', 'red'], autopct='%1.1f%%')
        plt.title("Profit/Loss Trade Ratio", fontsize=14)
        
        plt.tight_layout()
        
        # 저장 또는 표시
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    def generate_performance_report(self, save_path: Optional[str] = None) -> None:
        """
        성능 보고서 생성 및 저장
        
        Args:
            save_path: 보고서를 저장할 파일 경로 (옵션)
        """
        report = {
            "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "portfolio_summary": {
                "initial_balance": self.results.get("initial_balance", 10000),
                "final_value": float(self.results["portfolio_values"][-1]),
                "total_return": float(self.results["metrics"]["cumulative_return"]),
                "annual_return": float(self.results["metrics"]["annual_return"]),
                "trading_days": len(self.results["portfolio_values"]),
                "start_date": self.results["timestamps"][0] if self.results.get("timestamps") else None,
                "end_date": self.results["timestamps"][-1] if self.results.get("timestamps") else None
            },
            "risk_metrics": {
                "sharpe_ratio": float(self.results["metrics"]["sharpe_ratio"]),
                "max_drawdown": float(self.results["metrics"]["max_drawdown"]),
                "annual_volatility": float(self.results["metrics"]["annual_volatility"])
            },
            "trading_summary": {
                "total_trades": len(self.results["trades"]),
                "win_rate": float(self.results["metrics"]["win_rate"]),
                "avg_trade_return": np.mean(self.results["returns"]) if len(self.results["returns"]) > 0 else 0,
                "max_trade_return": np.max(self.results["returns"]) if len(self.results["returns"]) > 0 else 0,
                "min_trade_return": np.min(self.results["returns"]) if len(self.results["returns"]) > 0 else 0
            }
        }
        
        # 보고서를 JSON 파일로 저장
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=4)
            print(f"성능 보고서가 {save_path}에 저장되었습니다.")
        
        return report
                
    def visualize_all(self, benchmark_returns: Optional[np.ndarray] = None) -> None:
        """
        모든 시각화 함수 실행 및 결과 저장
        
        Args:
            benchmark_returns: 벤치마크 수익률 배열 (옵션)
        """
        if not self.save_dir:
            print("결과를 저장할 디렉토리를 지정해주세요.")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 자본금 곡선
        equity_path = os.path.join(self.save_dir, f"equity_curve_{timestamp}.png")
        self.plot_equity_curve(equity_path)
        
        # 2. 벤치마크 비교
        if benchmark_returns is not None:
            comparison_path = os.path.join(self.save_dir, f"benchmark_comparison_{timestamp}.png")
            self.plot_returns_comparison(benchmark_returns, comparison_path)
        
        # 3. 월별 수익률 히트맵
        try:
            monthly_path = os.path.join(self.save_dir, f"monthly_returns_{timestamp}.png")
            self.plot_monthly_returns_heatmap(monthly_path)
        except:
            print("월별 수익률 히트맵 생성 실패")
        
        # 4. 낙폭 기간
        drawdown_path = os.path.join(self.save_dir, f"drawdown_periods_{timestamp}.png")
        self.plot_drawdown_periods(save_path=drawdown_path)
        
        # 5. 거래 분석
        trades_path = os.path.join(self.save_dir, f"trade_analysis_{timestamp}.png")
        self.plot_trade_analysis(save_path=trades_path)
        
        # 6. 성능 보고서
        report_path = os.path.join(self.save_dir, f"performance_report_{timestamp}.json")
        self.generate_performance_report(save_path=report_path)
        
        print(f"모든 시각화 결과가 {self.save_dir} 디렉토리에 저장되었습니다.") 