import React from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function WineChart({ data }) {
  if (!data) return null;

  // Trích xuất dữ liệu thực sự (nếu API bọc trong data.data)
  const rawData = data.data ? data.data : data;

  // Danh sách loại trừ tất cả các trường không phải thông số hóa học
  const exclude = ['id', 'status', 'quality_score', 'quality_label', 'created_at', 'user_id', 'input_data', 'message', 'success', 'data', 'type'];
  
  const labels = Object.keys(rawData).filter(key => !exclude.includes(key.toLowerCase()));
  const values = labels.map(key => rawData[key]);

  const chartData = {
    labels: labels.map(l => l.replace(/_/g, ' ').toUpperCase()),
    datasets: [{
      label: "index",
      data: values,
      backgroundColor: 'rgba(56, 178, 172, 0.8)',
      borderRadius: 5,
    }]
  };

  return (
    <div className="chart-wrapper">
      <h3 style={{ fontSize: '15px', color: '#64748b', marginBottom: '15px' }}> Chemical composition symbol</h3>
      <div style={{ height: "280px" }}>
        <Bar data={chartData} options={{ responsive: true, maintainAspectRatio: false }} />
      </div>
    </div>
  );
}

export default WineChart;