let widget;

function initChart() {
    const symbol = document.getElementById('symbolSelect').value;
    const interval = document.getElementById('intervalSelect').value;
    const theme = document.getElementById('themeSelect').value;

    if (widget) {
        widget.remove();
    }

    widget = new TradingView.widget({
        "width": "100%",
        "height": 460,
        "symbol": symbol,
        "interval": interval,
        "timezone": "Asia/Seoul",
        "theme": theme,
        "style": "1",
        "locale": "kr",
        "toolbar_bg": theme === 'dark' ? "#1e1e1e" : "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart",
        "studies": [
            "MASimple@tv-basicstudies",
            "RSI@tv-basicstudies"
        ],
        "show_popup_button": true,
        "popup_width": "1000",
        "popup_height": "650"
    });
}

function updateChart() {
    initChart();
}

function initMarketOverview() {
    new TradingView.MarketOverview({
        "colorTheme": "dark",
        "dateRange": "12M",
        "showChart": true,
        "locale": "kr",
        "width": "100%",
        "height": "400",
        "largeChartUrl": "",
        "isTransparent": false,
        "showSymbolLogo": true,
        "showFloatingTooltip": false,
        "plotLineColorGrowing": "rgba(41, 98, 255, 1)",
        "plotLineColorFalling": "rgba(41, 98, 255, 1)",
        "gridLineColor": "rgba(240, 243, 250, 0)",
        "scaleFontColor": "rgba(120, 123, 134, 1)",
        "belowLineFillColorGrowing": "rgba(41, 98, 255, 0.12)",
        "belowLineFillColorFalling": "rgba(41, 98, 255, 0.12)",
        "belowLineFillColorGrowingBottom": "rgba(41, 98, 255, 0)",
        "belowLineFillColorFallingBottom": "rgba(41, 98, 255, 0)",
        "symbolActiveColor": "rgba(41, 98, 255, 0.12)",
        "tabs": [
            {
                "title": "암호화폐",
                "symbols": [
                    {"s": "BINANCE:BTCUSDT", "d": "Bitcoin"},
                    {"s": "BINANCE:ETHUSDT", "d": "Ethereum"},
                    {"s": "BINANCE:ADAUSDT", "d": "Cardano"},
                    {"s": "BINANCE:SOLUSDT", "d": "Solana"}
                ]
            },
            {
                "title": "한국 주식",
                "symbols": [
                    {"s": "KRX:005930", "d": "삼성전자"},
                    {"s": "KRX:000660", "d": "SK하이닉스"},
                    {"s": "KRX:035420", "d": "NAVER"},
                    {"s": "KRX:035720", "d": "카카오"}
                ]
            },
            {
                "title": "미국 주식",
                "symbols": [
                    {"s": "NASDAQ:AAPL", "d": "Apple"},
                    {"s": "NASDAQ:TSLA", "d": "Tesla"},
                    {"s": "NASDAQ:NVDA", "d": "NVIDIA"},
                    {"s": "NASDAQ:MSFT", "d": "Microsoft"}
                ]
            }
        ],
        "container_id": "tradingview_widget_container"
    });
}

// 페이지 로드 시 초기화
window.addEventListener('load', function() {
    initChart();

    // 시장 개요 위젯 초기화
    setTimeout(() => {
        const marketContainer = document.querySelector('.tradingview-widget-container__widget');
        if (marketContainer) {
            new TradingView.MiniWidget({
                "symbol": "BINANCE:BTCUSDT",
                "width": "100%",
                "height": "400",
                "locale": "kr",
                "dateRange": "12M",
                "colorTheme": "dark",
                "trendLineColor": "rgba(41, 98, 255, 1)",
                "underLineColor": "rgba(41, 98, 255, 0.3)",
                "underLineBottomColor": "rgba(41, 98, 255, 0)",
                "isTransparent": false,
                "autosize": false,
                "container_id": marketContainer
            });
        }
    }, 1000);
});
