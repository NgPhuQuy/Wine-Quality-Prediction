import React, { useRef } from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function WineChart({ data }) {
  const chartRef = useRef();

  if (!data) return null;

  const labels = Object.keys(data);
  const values = Object.values(data);

  // 🎯 options
  const options = {
    plugins: {
      legend: { display: false }
    },
    scales: {
      x: {
        ticks: { color: "#94a3b8", font: { size: 10 } },
        grid: { display: false }
      },
      y: {
        ticks: { color: "#94a3b8" },
        grid: { color: "rgba(148,163,184,0.1)" }
      }
    }
  };

  // 🎯 data chuẩn (KHÔNG phải function)
  const chartData = {
    labels: labels,
    datasets: [
      {
        label: "Wine Features",
        data: values,
        backgroundColor: (context) => {
          const chart = context.chart;
          const { ctx, chartArea } = chart;

          if (!chartArea) return "#38bdf8"; // tránh crash lần đầu

          const gradient = ctx.createLinearGradient(
            0,
            chartArea.top,
            0,
            chartArea.bottom
          );
          gradient.addColorStop(0, "#38bdf8");
          gradient.addColorStop(1, "#6366f1");

          return gradient;
        },
        borderRadius: 8
      }
    ]
  };

  return (
    <div className="chart-card" style={{ marginTop: "30px" }}>
      <h3>Feature Visualization</h3>
      <Bar
        ref={chartRef}
        data={chartData}
        options={options}
        height={140}
    />
    </div>
  );
}

export default WineChart;