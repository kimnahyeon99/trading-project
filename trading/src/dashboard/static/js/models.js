$(document).ready(function () {
    // 페이지 로드 시 데이터 로드
    loadModelList();
    loadModelComparisonChart();
    loadActiveModel();

    // 새로고침 버튼 이벤트
    $('#refresh-btn').click(function () {
        loadModelList(true);
        loadModelComparisonChart(true);
        loadActiveModel(true);
    });

    // 모델 업로드 폼 제출 이벤트
    $('#upload-form').submit(function (e) {
        e.preventDefault();
        uploadModel();
    });

    // 모델 상세보기 버튼 (동적 요소 대응)
    $(document).on('click', '.view-model', function () {
        const modelId = $(this).data('model-id');
        viewModelDetails(modelId);
    });

    // 모델 삭제 버튼 (동적 요소 대응)
    $(document).on('click', '.delete-model', function () {
        const modelId = $(this).data('model-id');
        $('#delete-model-name').text(modelId);
        $('#confirm-delete-btn').data('model-id', modelId);
        const modal = new bootstrap.Modal(document.getElementById('delete-model-modal'));
        modal.show();
    });

    // 삭제 확인 버튼
    $('#confirm-delete-btn').click(function () {
        const modelId = $(this).data('model-id');
        deleteModel(modelId);
    });
});

function loadModelList(refresh = false) {
    const refreshParam = refresh ? '?refresh=true' : '';

    $.getJSON(`/api/models${refreshParam}`, function (data) {
        const modelList = $('#model-list');
        modelList.empty();

        if ($.isEmptyObject(data)) {
            modelList.html('<tr><td colspan="5" class="text-center">등록된 모델이 없습니다.</td></tr>');
            return;
        }

        for (const modelId in data) {
            const model = data[modelId];
            const createdTime = model.created_time;
            const modifiedTime = model.modified_time;

            modelList.append(`
                <tr>
                    <td>${modelId}</td>
                    <td>${createdTime}</td>
                    <td>${modifiedTime}</td>
                    <td>
                        <button class="btn btn-sm btn-info view-model" data-model-id="${modelId}">
                            <i class="bi bi-eye"></i>
                        </button>
                        <button class="btn btn-sm btn-danger delete-model" data-model-id="${modelId}">
                            <i class="bi bi-trash"></i>
                        </button>
                    </td>
                </tr>
            `);
        }
    });
}

function viewModelDetails(modelId) {
    $.getJSON(`/api/models?model_id=${modelId}`, function (data) {
        if ($.isEmptyObject(data) || !data[modelId]) {
            $('#model-detail-content').html('<div class="alert alert-danger">모델 정보를 찾을 수 없습니다.</div>');
            return;
        }

        const model = data[modelId];

        let detailHtml = `
            <h5>기본 정보</h5>
            <table class="table">
                <tr><th width="30%">모델 ID</th><td>${modelId}</td></tr>
                <tr><th>파일 경로</th><td>${model.file_path}</td></tr>
                <tr><th>생성 시간</th><td>${model.created_time}</td></tr>
                <tr><th>수정 시간</th><td>${model.modified_time}</td></tr>
            </table>
        `;

        $.getJSON(`/api/backtest-results?model_id=${modelId}`, function (backtest) {
            if (!$.isEmptyObject(backtest) && backtest[modelId]) {
                const results = backtest[modelId];
                const metrics = results.metrics || {};

                detailHtml += `
                    <h5 class="mt-4">백테스트 결과</h5>
                    <table class="table">
                        <tr><th width="30%">백테스트 날짜</th><td>${results.backtest_date || '-'}</td></tr>
                        <tr><th>테스트 기간</th><td>${results.start_date || '-'} ~ ${results.end_date || '-'}</td></tr>
                        <tr><th>초기 잔고</th><td>$${results.initial_balance?.toLocaleString() || '-'}</td></tr>
                        <tr><th>최종 잔고</th><td>$${results.final_balance?.toLocaleString() || '-'}</td></tr>
                        <tr><th>총 수익률</th><td>${metrics.total_return ? (metrics.total_return * 100).toFixed(2) + '%' : '-'}</td></tr>
                        <tr><th>샤프 비율</th><td>${metrics.sharpe_ratio?.toFixed(2) || '-'}</td></tr>
                        <tr><th>최대 낙폭</th><td>${metrics.max_drawdown ? (metrics.max_drawdown * 100).toFixed(2) + '%' : '-'}</td></tr>
                        <tr><th>승률</th><td>${metrics.win_rate ? (metrics.win_rate * 100).toFixed(2) + '%' : '-'}</td></tr>
                        <tr><th>총 거래 수</th><td>${metrics.total_trades || '-'}</td></tr>
                    </table>
                `;
            } 

            $('#model-detail-content').html(detailHtml);
            const modal = new bootstrap.Modal(document.getElementById('model-detail-modal'));
            modal.show();
        });
    });
}

function loadModelComparisonChart(refresh = false) {
    const refreshParam = refresh ? '?refresh=true' : '';

    $.getJSON(`/api/charts/model-performance-alt${refreshParam}`, function (data) {
        if (data && !data.error) {
            Plotly.newPlot('model-performance-alt-chart', data.data, data.layout);
        } else {
            $('#model-performance-alt-chart').html('<div class="text-center py-5">모델 비교 데이터가 없습니다.</div>');
        }
    });
}

function loadActiveModel(refresh = false) {
    const refreshParam = refresh ? '?refresh=true' : '';

    $.getJSON(`/api/active-model${refreshParam}`, function (data) {
        const activeModelInfo = $('#active-model-info');

        if (!data || $.isEmptyObject(data) || !data.model_id) {
            activeModelInfo.html('<p class="mb-0">활성화된 모델이 없습니다.</p>');
            return;
        }

        activeModelInfo.html(`
            <div>
                <h6>${data.model_id}</h6>
                <p class="mb-1">활성화 시간: ${data.activated_time || '-'}</p>
                <p class="mb-0">
                    <span class="badge bg-success">활성</span>
                </p>
            </div>
        `);
    });
}

function uploadModel() {
    const modelId = $('#model-id').val();
    const modelFile = $('#model-file')[0].files[0];
    const description = $('#model-description').val();

    if (!modelId) {
        alert('모델 ID를 입력하세요.');
        return;
    }

    if (!modelFile) {
        alert('모델 파일을 선택하세요.');
        return;
    }

    const formData = new FormData();
    formData.append('model_id', modelId);
    formData.append('model_file', modelFile);
    formData.append('description', description);

    $.ajax({
        url: '/models',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            alert('모델 업로드 성공!');
            $('#upload-form')[0].reset();
            loadModelList(true);
        },
        error: function (xhr, status, error) {
            alert('모델 업로드 실패: ' + xhr.responseText);
        }
    });
}

       

