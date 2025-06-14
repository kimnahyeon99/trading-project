
$(document).ready(function() {
    // 페이지 로드 시 데이터 로드
    loadDashboardData();

    // 1분마다 자동 새로고침
    setInterval(loadDashboardData, 60000);
});

function loadDashboardData() {
    // 트레이딩 통계 로드
    $.getJSON('/api/trading-stats', function(data) {
        updateTradingStats(data);
    });

    // 차트 로드
    loadCharts();
}

function refreshDashboard() {
    // 새로고침 버튼 클릭 시 데이터 다시 로드
    $.getJSON('/api/trading-stats?refresh=true', function(data) {
        updateTradingStats(data);
    });

    loadCharts();
}

// 트레이딩 통계 
function updateTradingStats(data) {
    if (!data || $.isEmptyObject(data)) {
        $('#portfolio-value').text('데이터 없음');
        $('#portfolio-change').html('변화: <span>-</span>');
        $('#today-trades').html('거래 횟수: <span>-</span>');
        $('#total-return').text('데이터 없음');
        $('#positions-count').text('데이터 없음');
        $('#positions-value').html('가치: <span>-</span>');
        $('#recent-trades').html('<tr><td colspan="7" class="text-center">데이터가 없습니다.</td></tr>');
        return;
    }

    // 트레이딩 통계 데이터 추출(대시보드 화면에 표시 필요한 데이터)
    const stats = data.trading_stats;

    if (Array.isArray(stats) && stats.length > 0) {
        // 배열일 때: 가장 오래된 값과 최신값 비교
        const oldest = stats[0];
        const latest = stats[stats.length - 1];
    
        const oldestValue = oldest.portfolio_value || 0;
        const latestValue = latest.portfolio_value || 0;
        const portfolioChange = latestValue - oldestValue;
        const portfolioChangePercent = oldestValue > 0 ? (portfolioChange / oldestValue * 100) : 0;
    
        const pnl = latest.total_pnl || 0;
        const dailyPnl = latest.daily_pnl || 0;
        // initialBalance: 가장 처음 포트폴리오 가치
        const initialBalance = latestValue - pnl;
        // totalreturnPercent: 총 수익률
        const totalreturnPercent = initialBalance > 0
        ? (pnl / initialBalance) * 100
        : 0;
    
        $('#portfolio-value').text(`$${latestValue.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}`);
        $('#portfolio-change').html(`
            변화: <span class="${portfolioChange >= 0 ? 'text-success' : 'text-danger'}">
                ${portfolioChange >= 0 ? '+' : ''}$${portfolioChange.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                (${portfolioChangePercent.toFixed(2)}%)
            </span>
        `);
        $('#total-return').text(`${totalreturnPercent.toFixed(5)}%`);
        $('#daily-pnl').text(`$${dailyPnl.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}`);
    
        
    } else if (stats && typeof stats === 'object') {
        // 객체일 때: 그대로 처리
        const currentBalance = stats.portfolio_value || 0;
        const pnl = stats.total_pnl || 0;
        const dailyPnl = stats.daily_pnl || 0;
        const initialBalance = currentBalance - pnl;
        const pnlPercent = initialBalance > 0 ? (pnl / initialBalance * 100) : 0;
    
        $('#portfolio-value').text(`$${currentBalance.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}`);
        $('#portfolio-change').html(`변화: <span class="${pnl >= 0 ? 'text-success' : 'text-danger'}">${pnl >= 0 ? '+' : ''}$${pnl.toLocaleString('ko-KR', { maximumFractionDigits: 2 })} (${pnlPercent.toFixed(5)}%)</span>`);
        $('#total-return').text(`${pnlPercent.toFixed(5)}%`);
        $('#daily-pnl').text(`$${dailyPnl.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}`);
    }

    // 오늘 거래 횟수 계산 (trades 기반)
    if (data.trades && data.trades.length > 0) {
        const today = new Date().toISOString().split('T')[0];
        const todayTrades = data.trades.filter(t => t.timestamp && t.timestamp.startsWith(today));
        $('#daily-trades').html(`거래 횟수: <span>${todayTrades.length}</span>`);
    }

        // 기간 계산
    if (stats && stats.start_date && stats.end_date) {
        const startTime = new Date(stats.start_date);
        const endTime = new Date(stats.end_date);
        const diffTime = Math.abs(endTime - startTime);
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        $('#total-duration').html(`기간: <span>${diffDays}일</span>`);
    }


    // 포지션 데이터
    if (data.positions) {
        let positionsValue = 0;
        let positionsCount = 0;

        for (const symbol in data.positions) {
            const qty = parseFloat(data.positions[symbol].quantity || 0);
            const price = parseFloat(data.positions[symbol].current_price || 0);

            // 수량이 0보다 큰 포지션만 계산
            if (qty > 0) {
                positionsValue += qty * price;
                positionsCount += 1;
            }
        }

        $('#positions-count').text(positionsCount);
        $('#positions-value').html(`가치: <span>$${positionsValue.toLocaleString('ko-KR', {
            minimumFractionDigits: 6,
            maximumFractionDigits: 6
        })}</span>`);
    }


    // 최근 거래 내역
    if (data.trades && data.trades.length > 0) {
        const recentTrades = data.trades;
        let tradesHtml = '';
        recentTrades.forEach(trade => {
            const timestamp = trade.timestamp || '';
            const symbol = trade.symbol || '';
            const side = (trade.side || '').toUpperCase();
            const quantity = (trade.quantity || 0).toLocaleString('ko-KR', { maximumFractionDigits: 6 });
            const price = (trade.price || 0).toLocaleString('ko-KR', { maximumFractionDigits: 2 });
            const amount = ((trade.quantity || 0) * (trade.price || 0)).toLocaleString('ko-KR', { maximumFractionDigits: 2 });
            const status = trade.status || '';
            const sideClass = side === 'BUY' ? 'text-success' : 'text-danger';
            const statusClass = status === 'success' ? 'text-success' : 'text-danger';
            tradesHtml += `
                <tr>
                    <td>${timestamp}</td>
                    <td>${symbol}</td>
                    <td class="${sideClass}">${side}</td>
                    <td>${quantity}</td>
                    <td>$${price}</td>
                    <td>$${amount}</td>
                </tr>`;
        });
        $('#recent-trades').html(tradesHtml);
    } else {
        $('#recent-trades').html('<tr><td colspan="7" class="text-center">거래 내역이 없습니다.</td></tr>');
    }
}
       







// <차트 로드>
    
function loadCharts() {
    // 포트폴리오 가치 차트
    $.getJSON('/api/charts/portfolio', function(data) {
        if (data && !data.error) {
            const finalLayout = { ...data.layout };
            Plotly.newPlot('portfolio-chart', data.data, finalLayout);
        } else {
            $('#portfolio-chart').html('<div class="text-center py-5">포트폴리오 데이터가 없습니다.</div>');
        }
    });

    // 수익률 차트
    $.getJSON('/api/charts/returns', function(data) {
        if (data && !data.error) {
            const finalLayout = { ...data.layout };
            Plotly.newPlot('returns-chart', data.data, finalLayout);
        } else {
            $('#returns-chart').html('<div class="text-center py-5">수익률 데이터가 없습니다.</div>');
        }
    });

    // 낙폭 차트
    $.getJSON('/api/charts/drawdown', function(data) {
        if (data && !data.error) {
            const finalLayout = { ...data.layout };
            Plotly.newPlot('drawdown-chart', data.data, finalLayout);
        } else {
            $('#drawdown-chart').html('<div class="text-center py-5">낙폭 데이터가 없습니다.</div>');
        }
    });

    // 매수 vs 매도 비율 바차트
    $.getJSON('/api/charts/trade-buy-sell', function(data) {
        if (data && !data.error) {
            const finalLayout = { ...data.layout };
            Plotly.newPlot('trade-buy-sell-chart', data.data, finalLayout);
        } else {
            $('#trade-buy-sell-chart').html('<div class="text-center py-5">데이터 없음</div>');
        }
    });

}
