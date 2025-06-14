
        $(document).ready(function() {
            // 페이지 로드 시 데이터 로드
            loadTradingState();
            
            // 버튼 이벤트 리스너
            $('#refresh-btn').click(function() {
                loadTradingState(true);
            });
            
            $('#start-btn').click(function() {
                // TODO: 실제 API 연동 시 트레이딩 시작 API 호출
                alert('트레이딩이 시작되었습니다.');
                updateTradingStatus(true);
            });
            
            $('#pause-btn').click(function() {
                // TODO: 실제 API 연동 시 트레이딩 일시정지 API 호출
                alert('트레이딩이 일시정지되었습니다.');
                updateTradingStatus(false, 'paused');
            });
            
            $('#stop-btn').click(function() {
                // TODO: 실제 API 연동 시 트레이딩 중지 API 호출
                if (confirm('정말로 트레이딩을 중지하시겠습니까?')) {
                    alert('트레이딩이 중지되었습니다.');
                    updateTradingStatus(false, 'stopped');
                }
            });
            
            // 심볼 선택 이벤트
            $('#symbol-select').change(function() {
                const symbol = $(this).val();
                if (symbol) {
                    loadPriceChart(symbol);
                }
            });
            
            // 10초마다 자동 새로고침
            setInterval(loadTradingState, 10000);
        });
        
        function loadTradingState(forceRefresh = false) {
            const refresh = forceRefresh ? '?refresh=true' : '';
            
            $.getJSON(`/api/trading-stats${refresh}`, function(data) {
                updateTradingData(data);
                loadSymbols(data);
            });
        }
        function updateTradingData(data) {
            if (!data || $.isEmptyObject(data)) {
                updateTradingStatus(false, 'stopped');
                return;
            }
        
            updateTradingStatus(true);
            
            const stats = data.trading_stats;
        
            // 📌 stats가 배열일 때 (시간순 데이터 여러 개 있는 경우)
            if (Array.isArray(stats) && stats.length > 0) {
                const oldest = stats[0];
                const latest = stats[stats.length - 1];
        
                const oldestValue = oldest.portfolio_value || 0;
                const latestValue = latest.portfolio_value || 0;
                const portfolioChange = latestValue - oldestValue;
                const portfolioChangePercent = oldestValue > 0 ? (portfolioChange / oldestValue * 100) : 0;
        
                const pnl = latest.total_pnl || 0;
                const dailyPnl = latest.daily_pnl || 0;
                const initialBalance = latestValue - pnl;
                const totalreturnPercent = initialBalance > 0 ? (pnl / initialBalance * 100) : 0;
        
                $('#portfolio-value').text(`$${latestValue.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}`);
                $('#portfolio-change').html(`변화: <span class="${portfolioChange >= 0 ? 'text-success' : 'text-danger'}">
                    ${portfolioChange >= 0 ? '+' : ''}$${portfolioChange.toLocaleString('ko-KR', { maximumFractionDigits: 2 })} (${portfolioChangePercent.toFixed(2)}%)</span>`);
                $('#total-return').text(`${totalreturnPercent.toFixed(5)}%`);
                $('#daily-pnl').text(`$${dailyPnl.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}`);
        
            } else if (stats && typeof stats === 'object') {
                // 📌 stats가 단일 객체일 때
                const currentBalance = stats.portfolio_value || 0;
                const pnl = stats.total_pnl || 0;
                const dailyPnl = stats.daily_pnl || 0;
                const initialBalance = currentBalance - pnl;
                const pnlPercent = initialBalance > 0 ? (pnl / initialBalance * 100) : 0;
        
                $('#portfolio-value').text(`$${currentBalance.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}`);
                $('#portfolio-change').html(`변화: <span class="${pnl >= 0 ? 'text-success' : 'text-danger'}">
                    ${pnl >= 0 ? '+' : ''}$${pnl.toLocaleString('ko-KR', { maximumFractionDigits: 2 })} (${pnlPercent.toFixed(5)}%)</span>`);
                $('#total-return').text(`${pnlPercent.toFixed(5)}%`);
                $('#daily-pnl').text(`$${dailyPnl.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}`);
            }
        
            // 📌 오늘 거래 횟수 계산
            if (data.trades && data.trades.length > 0) {
                const today = new Date().toISOString().split('T')[0];
                const todayTrades = data.trades.filter(t => t.timestamp && t.timestamp.startsWith(today));
                $('#daily-trades').html(`거래 횟수: <span>${todayTrades.length}</span>`);
            }
        
            // 📌 기간 계산 (시작일-종료일 존재 시)
            if (stats && stats.start_date && stats.end_date) {
                const startTime = new Date(stats.start_date);
                const endTime = new Date(stats.end_date);
                const diffTime = Math.abs(endTime - startTime);
                const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
                $('#total-duration').html(`기간: <span>${diffDays}일</span>`);
            }
        
            // 📌 포지션 정보
            if (data.positions) {
                let positionsValue = 0;
                let positionsCount = 0;
                let activeSymbols = 0;

                for (const symbol in data.positions) {
                    const qty = parseFloat(data.positions[symbol].quantity || 0);
                    const price = parseFloat(data.positions[symbol].current_price || 0);

                    if (qty > 0) {
                        positionsValue += qty * price;
                        positionsCount += 1;
                        activeSymbols += 1;
                    }
                }

                $('#positions-count').text(positionsCount);
                $('#positions-value').html(`가치: <span>$${positionsValue.toLocaleString('ko-KR', {
                    minimumFractionDigits: 6,
                    maximumFractionDigits: 6
                })}</span>`);
                $('#active-symbols').text(activeSymbols);  // ⭐ 추가된 부분!
            }

        
            // 📌 계정 현금
            $.getJSON('/api/account', function(accountData) {
                const cash = parseFloat(accountData.cash || 0).toLocaleString('ko-KR', { maximumFractionDigits: 2 });
                $('#account-balance').text(`$${cash}`);
            });
        
            // 📌 최근 거래 내역
            updateTradingStats(data);
        
            // 📌 포지션 / 주문 테이블
            updatePositionsTable(data.positions || {});
            updateOrdersTable(data.open_orders || {});
        
            // 📌 포트폴리오 차트
            $.getJSON('/api/charts/portfolio', function(chartData) {
                if (chartData && !chartData.error) {
                    Plotly.newPlot('portfolio-chart', chartData.data, chartData.layout);
                }
            });
        }
        
        
        function updateTradingStatus(isRunning, status = '') {
            const statusBadge = $('#trading-status');
            
            if (isRunning) {
                statusBadge.removeClass('bg-warning bg-danger').addClass('bg-success');
                statusBadge.text('실행 중');
                $('.live-indicator').show();
            } else if (status === 'paused') {
                statusBadge.removeClass('bg-success bg-danger').addClass('bg-warning');
                statusBadge.text('일시정지');
                $('.live-indicator').hide();
            } else {
                statusBadge.removeClass('bg-success bg-warning').addClass('bg-danger');
                statusBadge.text('중지됨');
                $('.live-indicator').hide();
            }
            
            // 버튼 상태 업데이트
            $('#start-btn').prop('disabled', isRunning);
            $('#pause-btn').prop('disabled', !isRunning);
            $('#stop-btn').prop('disabled', !isRunning && status === 'stopped');
        }
        
        function updatePositionsTable(positions) {
            const positionsTable = $('#positions-table');
            
            if (!positions || Object.keys(positions).length === 0) {
                positionsTable.html('<tr><td colspan="8" class="text-center">보유 중인 포지션이 없습니다.</td></tr>');
                return;
            }
            
            let html = '';
            
            for (const symbol in positions) {
                const position = positions[symbol];
                const quantity = parseFloat(position.quantity || 0);
                const entryPrice = parseFloat(position.avg_entry_price || 0);
                const currentPrice = parseFloat(position.current_price || 0);
                const unrealizedPnl = parseFloat(position.unrealized_pnl || 0);
                
                // market value 계산
                const marketValue = quantity * currentPrice;
        
                const positionType = quantity > 0 ? 'Long' : (quantity < 0 ? 'Short' : 'None');
                const positionBadgeClass = quantity > 0 ? 'badge-long' : (quantity < 0 ? 'badge-short' : '');
                
                const pnlClass = unrealizedPnl > 0 ? 'text-success' : (unrealizedPnl < 0 ? 'text-danger' : '');
                
                html += `
                    <tr>
                        <td>${symbol}</td>
                        <td><span class="position-badge ${positionBadgeClass}">${positionType}</span></td>
                        <td>${Math.abs(quantity).toLocaleString('ko-KR', {maximumFractionDigits: 6})}</td>
                        <td>$${entryPrice.toLocaleString('ko-KR', {maximumFractionDigits: 2})}</td>
                        <td>$${currentPrice.toLocaleString('ko-KR', {maximumFractionDigits: 2})}</td>
                        <td>$${marketValue.toLocaleString('ko-KR', {maximumFractionDigits: 2})}</td>
                        <td class="${pnlClass}">$${unrealizedPnl.toLocaleString('ko-KR', {maximumFractionDigits: 2})}</td>
                        <td>
                            <button class="btn btn-sm btn-danger close-position" data-symbol="${symbol}">청산</button>
                        </td>
                    </tr>
                `;
            }
        
            positionsTable.html(html);
        }
        
            
            positionsTable.html(html);
            
            // 청산 버튼 이벤트 리스너
            $('.close-position').click(function() {
                const symbol = $(this).data('symbol');
                if (confirm(`${symbol} 포지션을 청산하시겠습니까?`)) {
                    // TODO: 실제 API 연동 시 포지션 청산 API 호출
                    alert(`${symbol} 포지션 청산이 요청되었습니다.`);
                }
            });
        
        
        function updateOrdersTable(orders) {
            const ordersTable = $('#orders-table');
            
            if (!orders || Object.keys(orders).length === 0) {
                ordersTable.html('<tr><td colspan="8" class="text-center">미체결 주문이 없습니다.</td></tr>');
                return;
            }
            
            let html = '';
            
            for (const orderId in orders) {
                const order = orders[orderId];
                const symbol = order.symbol || '';
                const side = (order.side || '').toUpperCase();
                const quantity = parseFloat(order.quantity || 0);
                const price = parseFloat(order.price || 0);
                const createTime = order.create_time || '';
                const status = order.status || '';
                
                const sideClass = side === 'BUY' ? 'text-success' : 'text-danger';
                
                html += `
                    <tr>
                        <td>${orderId}</td>
                        <td>${symbol}</td>
                        <td class="${sideClass}">${side}</td>
                        <td>${quantity.toLocaleString('ko-KR', {maximumFractionDigits: 6})}</td>
                        <td>$${price.toLocaleString('ko-KR', {maximumFractionDigits: 2})}</td>
                        <td>${createTime}</td>
                        <td>${status}</td>
                        <td>
                            <button class="btn btn-sm btn-warning cancel-order" data-order-id="${orderId}">취소</button>
                        </td>
                    </tr>
                `;
            }
            
            ordersTable.html(html);
            
            // 취소 버튼 이벤트 리스너
            $('.cancel-order').click(function() {
                const orderId = $(this).data('order-id');
                if (confirm(`주문 ${orderId}를 취소하시겠습니까?`)) {
                    // TODO: 실제 API 연동 시 주문 취소 API 호출
                    alert(`주문 ${orderId} 취소가 요청되었습니다.`);
                }
            });
        }
        
        function updateTradingStats(data) {

        const stats = data.trading_stats;
        
   
    
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
        
        function loadSymbols(data) {
            const symbolSelect = $('#symbol-select');
            const currentSymbol = symbolSelect.val();
            
            // 현재 포지션에서 심볼 추출
            const symbols = Object.keys(data.positions || {});
            
            // 이미 옵션이 설정되어 있고 현재 선택된 심볼이 있으면 유지
            if (symbolSelect.find('option').length > 1 && currentSymbol && symbols.includes(currentSymbol)) {
                return;
            }
            
            // 옵션 초기화
            symbolSelect.empty();
            symbolSelect.append('<option value="">심볼 선택</option>');
            
            // 심볼 옵션 추가
            for (const symbol of symbols) {
                symbolSelect.append(`<option value="${symbol}">${symbol}</option>`);
            }
            
            // 심볼이 있으면 첫 번째 심볼 선택
            if (symbols.length > 0) {
                symbolSelect.val(symbols[0]);
                loadPriceChart(symbols[0]);
            }
        }
        
        function loadPriceChart(symbol) {
            $.getJSON(`/api/market-data?symbol=${symbol}&limit=100`, function(marketData) {
                if (!marketData || marketData.length === 0) {
                    $('#price-chart').html('<div class="text-center py-5">해당 심볼의 시장 데이터가 없습니다.</div>');
                    return;
                }
                
                // 거래 내역에서 해당 심볼의 매수/매도 신호 추출
                $.getJSON('/api/trading-stats', function(tradingStats) {
                    if (!tradingStats || !tradingStats.trading_stats || !tradingStats.trading_stats.trades) {
                        return;
                    }
                    
                    const trades = tradingStats.trading_stats.trades.filter(t => t.symbol === symbol);
                    const buySignals = [];
                    const sellSignals = [];
                    
                    for (const trade of trades) {
                        if (trade.side === 'buy') {
                            buySignals.push([trade.timestamp, trade.price]);
                        } else if (trade.side === 'sell') {
                            sellSignals.push([trade.timestamp, trade.price]);
                        }
                    }
                    
                    // 시장 데이터에서 가격 및 날짜 추출
                    const dates = marketData.map(d => d.date || d.timestamp);
                    const prices = marketData.map(d => d.close);
                    
                    // 차트 생성
                    $.getJSON(`/api/charts/price?symbol=${symbol}&buy_signals=${JSON.stringify(buySignals)}&sell_signals=${JSON.stringify(sellSignals)}`, function(chartData) {
                        if (chartData && !chartData.error) {
                            Plotly.newPlot('price-chart', chartData.data, chartData.layout);
                        } else {
                            // 차트 API가 없는 경우 직접 생성
                            const trace = {
                                x: dates,
                                y: prices,
                                type: 'scatter',
                                mode: 'lines',
                                name: `${symbol} 가격`,
                                line: { color: '#1f77b4', width: 2 }
                            };
                            
                            const layout = {
                                title: `${symbol} 가격 및 거래 신호`,
                                xaxis: { title: '날짜' },
                                yaxis: { title: '가격' },
                                template: 'plotly_white'
                            };
                            
                            Plotly.newPlot('price-chart', [trace], layout);
                        }
                    });
                });
            });
        }
   