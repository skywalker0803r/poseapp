const uploadForm = document.getElementById('upload-form');
const videoUpload = document.getElementById('video-upload');
const videoStream = document.getElementById('video-stream');
const jointSelect = document.getElementById('joint-select');
const statusText = document.getElementById('status-text');
const loadingSpinner = document.getElementById('loading-spinner');

const valX = document.getElementById('val-x');
const valY = document.getElementById('val-y');
const valZ = document.getElementById('val-z');

const ctx = document.getElementById('joint-chart').getContext('2d');

let socket = null;
let chart = null;

const maxDataPoints = 100;
const dataX = [];
const dataY = [];
const dataZ = [];

uploadForm.addEventListener('submit', async e => {
  e.preventDefault();
  if (videoUpload.files.length === 0) {
    alert("請選擇影片檔");
    return;
  }

  loadingSpinner.style.display = 'inline-block';
  statusText.textContent = "影片上傳中...";

  const formData = new FormData();
  formData.append('video', videoUpload.files[0]);

  try {
    const res = await fetch('/upload', {
      method: 'POST',
      body: formData
    });
    const result = await res.json();
    if (result.success) {
      statusText.textContent = "上傳成功，開始分析...";
      videoStream.src = '/video_feed';

      if (socket) {
        socket.disconnect();
      }
      initSocket();
    } else {
      statusText.textContent = "上傳失敗";
      alert("影片上傳失敗");
    }
  } catch(err) {
    statusText.textContent = "錯誤發生";
    alert("影片上傳錯誤：" + err);
  } finally {
    loadingSpinner.style.display = 'none';
  }
});

jointSelect.addEventListener('change', () => {
  if (socket) {
    socket.emit('select_joint', { joint: jointSelect.value });
    resetChartData();
    updateChartTitle();
    resetCoordinateValues();
  }
});

function initSocket() {
  socket = io();

  socket.on('connect', () => {
    console.log('Socket connected');
    socket.emit('select_joint', { joint: jointSelect.value });
    resetChartData();
    updateChartTitle();
    statusText.textContent = "分析中...";
  });

  socket.on('joint_data', data => {
    if (data.x !== null && data.y !== null && data.z !== null) {
      pushData(dataX, data.x);
      pushData(dataY, data.y);
      pushData(dataZ, data.z);
      updateChart();

      valX.textContent = `X: ${data.x.toFixed(3)}`;
      valY.textContent = `Y: ${data.y.toFixed(3)}`;
      valZ.textContent = `Z: ${data.z.toFixed(3)}`;
    } else {
      valX.textContent = `X: --`;
      valY.textContent = `Y: --`;
      valZ.textContent = `Z: --`;
    }
  });

  socket.on('pose_feedback', (data) => {
    if (data.error) {
        document.getElementById('pose-status').innerText = "分析錯誤: " + data.error;
    } else {
        document.getElementById('pose-status').innerText =
            `肘部角度: ${data.elbow_angle.toFixed(2)}° → ${data.status}`;
    }
});

// 相似度分析
const ctx = document.getElementById('similarityChart').getContext('2d');
const similarityChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      {
        label: '起手相似度',
        data: [],
        borderColor: 'rgba(75, 192, 192, 1)',
        fill: false
      },
      {
        label: '結束相似度',
        data: [],
        borderColor: 'rgba(255, 99, 132, 1)',
        fill: false
      }
    ]
  },
  options: {
    responsive: true,
    animation: false,
    scales: {
      y: {
        min: 0,
        max: 1,
        ticks: {
          stepSize: 0.1
        }
      }
    }
  }
});

let frameCount = 0;
let maxStartSimilarity = 0;
let maxEndSimilarity = 0;
socket.on('standard_action_comparison', (data) => {
    if (data.error) {
        document.getElementById('start_similarity').innerText = "分析錯誤: " + data.error;
        document.getElementById('end_similarity').innerText = "分析錯誤: " + data.error;
    } else {
        // 顯示即時數據
        document.getElementById('start_similarity').innerText = "起手姿勢相似度: " + data.similarity_to_start;
        document.getElementById('end_similarity').innerText = "結束姿勢相似度: " + data.similarity_to_end;

        // 更新並顯示最大峰值
        if (data.similarity_to_start > maxStartSimilarity) {
            maxStartSimilarity = data.similarity_to_start;
            document.getElementById('max_start_similarity').innerText = "起手姿勢最高相似度: " + maxStartSimilarity;
        }
        if (data.similarity_to_end > maxEndSimilarity) {
            maxEndSimilarity = data.similarity_to_end;
            document.getElementById('max_end_similarity').innerText = "結束姿勢最高相似度: " + maxEndSimilarity;
        }
        // 更新圖表
        frameCount += 1;
        similarityChart.data.labels.push(frameCount);
        similarityChart.data.datasets[0].data.push(data.similarity_to_start);
        similarityChart.data.datasets[1].data.push(data.similarity_to_end);
        similarityChart.update();
    }
});

  socket.on('disconnect', () => {
    console.log('Socket disconnected');
    statusText.textContent = "連線中斷";
  });
}

function pushData(arr, val) {
  arr.push(val);
  if (arr.length > maxDataPoints) {
    arr.shift();
  }
}

function resetChartData() {
  dataX.length = 0;
  dataY.length = 0;
  dataZ.length = 0;
}

function resetCoordinateValues() {
  valX.textContent = "X: --";
  valY.textContent = "Y: --";
  valZ.textContent = "Z: --";
}

function updateChartTitle() {
  const idx = jointSelect.value;
  const name = jointSelect.options[jointSelect.selectedIndex].text;
  document.getElementById('chart-title').textContent = `關節 ${idx}: ${name} 位置變化 (x/y/z)`;
}

function updateChart() {
  if (!chart) {
    chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: dataX.map((_, i) => i),
        datasets: [
          {
            label: 'X',
            borderColor: 'red',
            backgroundColor: 'rgba(255,0,0,0.2)',
            data: dataX,
            fill: true,
            tension: 0.2,
            pointRadius: 0,
          },
          {
            label: 'Y',
            borderColor: 'green',
            backgroundColor: 'rgba(0,128,0,0.2)',
            data: dataY,
            fill: true,
            tension: 0.2,
            pointRadius: 0,
          },
          {
            label: 'Z',
            borderColor: 'blue',
            backgroundColor: 'rgba(0,0,255,0.2)',
            data: dataZ,
            fill: true,
            tension: 0.2,
            pointRadius: 0,
          },
        ]
      },
      options: {
        animation: false,
        responsive: true,
        scales: {
          x: { display: false },
          y: { beginAtZero: true }
        },
        plugins: {
          legend: { position: 'top' }
        }
      }
    });
  } else {
    chart.data.labels = dataX.map((_, i) => i);
    chart.data.datasets[0].data = dataX;
    chart.data.datasets[1].data = dataY;
    chart.data.datasets[2].data = dataZ;
    chart.update('none');
  }
}