// 모델 목록 로드
function loadModels() {
  fetch('/api/backtest-results')
    .then(response => response.json())
    .then(data => {
      const modelSelect = document.getElementById('modelSelect');
      const modelIds = Object.keys(data);
      if (modelIds.length === 0) return;

      modelIds.forEach(modelId => {
        const option = document.createElement('option');
        option.value = modelId;
        option.textContent = modelId;
        modelSelect.appendChild(option);
      });

      // 자동 선택 제거됨
    });
}



// 특정 모델 결과 로드
function loadBacktestResult(modelId) {
  if (!modelId) return;
  fetch(`/api/backtest-results?model_id=${modelId}`)
    .then(response => response.json())
    .then(data => {
      if (!data || !data[modelId]) return;
      const result = data[modelId];
      displayMetrics(result.metrics);
      loadCharts(modelId);
      displayTrades(result.trades);
    });
}

// 성능 요약 표시
// function displayMetrics(metrics) {
//   const elements = {
//     totalReturn: (metrics.total_return).toFixed(2) + '%',
//     sharpeRatio: metrics.sharpe_ratio.toFixed(2),
//     maxDrawdown: (metrics.max_drawdown * 100).toFixed(2) + '%',
//     winRate: (metrics.win_rate * 100).toFixed(2) + '%'
//   };

//   for (const id in elements) {
//     const el = document.getElementById(id);
//     el.textContent = elements[id];

//     el.classList.add('value-text');        
//     el.classList.add('highlight-value');   

//     setTimeout(() => {
//       el.style.color = '#111111'; 
//     }, 700);
//   }
// }

function displayMetrics(metrics) {
  const elements = {
    totalReturn: (metrics.total_return).toFixed(2) + '%',
    sharpeRatio: metrics.sharpe_ratio.toFixed(2),
    maxDrawdown: (metrics.max_drawdown * 100).toFixed(2) + '%',
    winRate: (metrics.win_rate * 100).toFixed(2) + '%'
  };

  for (const id in elements) {
    const el = document.getElementById(id);
    el.textContent = elements[id];
    el.classList.remove('highlight-value');
    void el.offsetWidth;
    el.classList.add('highlight-value');
    setTimeout(() => {
      el.classList.remove('highlight-value');
    }, 1000); // 
  }
}





// 모델별 동적 차트 로딩 
function loadCharts(modelId) {

  fetch(`/api/charts/trade-distribution`)
  .then(res => res.json())
  .then(data => Plotly.newPlot('modelTradeCountChart', data.data, data.layout));

  fetch(`/api/charts/trade-distribution?model_id=${modelId}`)
    .then(res => res.json())
    .then(data => Plotly.newPlot('tradeDistributionChart', data.data, data.layout));

  fetch(`/api/charts/model-comparison`)
    .then(res => res.json())
    .then(data => Plotly.newPlot('modelComparisonChart', data.data, data.layout));
  
  fetch(`/api/charts/risk-return`)
    .then(res => res.json())
    .then(data => Plotly.newPlot('riskReturnChart', data.data, data.layout));
  
}

// 거래 내역 테이블 렌더링
function displayTrades(trades) {
  const tbody = document.querySelector('#tradesTable tbody');
  tbody.innerHTML = '';
  if (!trades || trades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="text-center">거래 내역이 없습니다.</td></tr>';
    return;
  }
  trades.forEach(backtest_trade => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${backtest_trade.timestamp}</td>
      <td>${backtest_trade.action > 0 ? '매수' : '매도'}</td>
      <td>${backtest_trade.price.toFixed(2)}</td>
      <td>${backtest_trade.shares}</td>
      <td>${backtest_trade.cost ? backtest_trade.cost.toFixed(2) : '-'}</td>
      <td>${backtest_trade.portfolio_value.toFixed(2)}</td>
    `;
    tbody.appendChild(row);
  });
}


// 성능 요약 초기화
function resetMetrics() {
  const ids = ['totalReturn', 'sharpeRatio', 'maxDrawdown', 'winRate'];
  ids.forEach(id => {
    const el = document.getElementById(id);
    el.textContent = '-';
    el.style.color = '#111111'; // 기본 색상
  });
}



// 페이지 로드 시 실행
document.addEventListener('DOMContentLoaded', function () {
  loadModels();

  // 기본 차트 4개 모두 로드
  fetch(`/api/charts/initial-vs-final`)
    .then(res => res.json())
    .then(data => Plotly.newPlot('initialVsFinalChart', data.data, data.layout));

  fetch(`/api/charts/risk-return`)
    .then(res => res.json())
    .then(data => Plotly.newPlot('riskReturnChart', data.data, data.layout));

  fetch(`/api/charts/model-comparison`)
    .then(res => res.json())
    .then(data => Plotly.newPlot('modelComparisonChart', data.data, data.layout));

  fetch(`/api/charts/trade-distribution`)
    .then(res => res.json())
    .then(data => Plotly.newPlot('modelTradeCountChart', data.data, data.layout));

  // 모델 선택 시에만 백테스트 결과 로딩
  document.getElementById('modelSelect').addEventListener('change', function () {
    const selectedValue = this.value;
    if (!selectedValue) {
      resetMetrics();  // 선택 해제 시 값 초기화
      return;
    }
    loadBacktestResult(selectedValue);  // 모델 선택 시엔 결과 불러오기
  });
});


