const TradingView = require('tradingview-ws');

async function main() {
  await TradingView.connect();
  
  const symbols = ['BINANCE:BTCUSDT']; 
  const interval = '1';
  
  // 배열과 다른 매개변수를 분리해서 전달
  TradingView.getCandles(symbols, interval, (data) => {
    console.log('Received candle:', data);
  });
}

main().catch(console.error);