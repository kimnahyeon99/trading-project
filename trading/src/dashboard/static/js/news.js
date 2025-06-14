
  function filterNews(ticker) {
    const activeTabId = document.querySelector('.nav-tabs .nav-link.active').id;
    let tabParam = (activeTabId === 'yahoo-tab') ? 'yahoo' : 'polygon';
    // window.location.href = `/news?name=${ticker}&source=${tabParam}`;
  

  document.querySelectorAll('.ticker-btn').forEach(btn => {
    btn.classList.remove('active');
  });

  // ✅ 클릭된 버튼만 active 클래스 추가
  const clickedBtn = Array.from(document.querySelectorAll('.ticker-btn'))
    .find(btn => btn.textContent === ticker);
  if (clickedBtn) {
    clickedBtn.classList.add('active');
  }

  // ✅ URL 이동 (새로고침)
  window.location.href = `/news?name=${ticker}&source=${tabParam}`;
}

  document.addEventListener("DOMContentLoaded", function () {
    const queryString = window.location.search;
    fetch('/api/news' + queryString)
      .then(res => res.json())
      .then(data => {
        renderNews(data.polygon || [], 'polygon');
        renderNews(data.yahoo || [], 'yahoo');
      })
      .catch(() => {
        document.getElementById('polygon-news').innerHTML = '<p>오류 발생</p>';
        document.getElementById('yahoo-news').innerHTML = '<p>오류 발생</p>';
      });


      function renderNews(newsList, type) {
        const container = document.getElementById(`${type}-news`);
        const loadMoreBtn = document.getElementById(`${type}-load-more`);
        const countContainerId = `${type}-count`;
        let currentIndex = 0;
        const perPage = 10;
      
        
        container.innerHTML = '';
      
        // 정렬: 최신 날짜 순 (내림차순)
        newsList.sort((a, b) => {
          const dateA = new Date(a.published_date_kst || a.date || 0);
          const dateB = new Date(b.published_date_kst || b.date || 0);
          return dateB - dateA;
        });
      
        // 총 개수 표시 
        let countEl = document.getElementById(countContainerId);
        if (!countEl) {
          countEl = document.createElement('div');
          countEl.id = countContainerId;
          countEl.className = 'news-count';
          container.parentElement.insertBefore(countEl, container);
        }
        countEl.innerHTML = `총 <span class="news-count-number">${newsList.length}</span>건`;

      
        function showNext() {
          const end = currentIndex + perPage;
          const pageItems = newsList.slice(currentIndex, end);
      
          if (pageItems.length === 0 && currentIndex === 0) {
            container.innerHTML = '<p>데이터가 없습니다.</p>';
            if (loadMoreBtn) loadMoreBtn.style.display = 'none';
            return;
          }
      
          pageItems.forEach(news => {
            const title = news.title || '(제목 없음)';
            const summary = news.summary || '';
            const sentiment = (news.sentiment || 'neutral').toLowerCase();
            const date = news.published_date_kst || news.date || '-';
            const url = news.url || '#';
      
            let sentimentClass = 'sentiment-neutral';
            if (sentiment === 'positive') sentimentClass = 'sentiment-positive';
            else if (sentiment === 'negative') sentimentClass = 'sentiment-negative';
      
            const div = document.createElement('div');
            div.className = 'news-card';
            div.innerHTML = `
              <h5><a href="${url}" target="_blank">${title}</a></h5>
              <p>${summary}</p>
              <div class="news-meta">
                <span class="sentiment-tag ${sentimentClass}">
                  ${sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}
                </span>
                <span>${date}</span>
                <a href="${url}" target="_blank"><i class="bi bi-link-45deg"></i> ${url}</a>
              </div>
            `;
            container.appendChild(div);
          });
      
          currentIndex += perPage;
          if (currentIndex >= newsList.length && loadMoreBtn) {
            loadMoreBtn.style.display = 'none';
          }
        }
      
        if (loadMoreBtn) {
          loadMoreBtn.style.display = 'block'; 
          loadMoreBtn.removeEventListener('click', showNext);
          loadMoreBtn.addEventListener('click', showNext);
          showNext();
        }
      }
    });
 