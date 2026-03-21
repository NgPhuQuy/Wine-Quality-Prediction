import React from "react";

function ResultCard({ result }) {
  // 1. Kiểm tra dữ liệu đầu vào
  if (result === null || result === undefined) return null;

  // 2. Chuyển đổi và tính toán logic
  const score = Number(result);
  const percent = (score / 10) * 100;

  // 3. Hàm lấy màu sắc theo điểm số
  const getColor = () => {
    if (score >= 7) return "#10b981"; // Xanh ngọc (Excellent)
    if (score >= 5) return "#f59e0b"; // Vàng kim (Average)
    return "#ef4444"; // Đỏ (Poor)
  };

  const color = getColor();

  return (
    <div className="result-card luxury" style={{ textAlign: "center", padding: "20px" }}>
      <p style={{ color: "#94a3b8", fontSize: "12px", fontWeight: "bold", letterSpacing: "2px", marginBottom: "20px" }}>
        ANALYSIS RESULTS
      </p>

      {/* 🎯 Biểu đồ Gauge bằng CSS Conic-Gradient */}
      <div className="gauge-container" style={{
        position: "relative",
        width: "120px",
        height: "120px",
        margin: "0 auto 20px"
      }}>
        <div 
          className="gauge-fill"
          style={{
            width: "100%",
            height: "100%",
            borderRadius: "50%",
            background: `conic-gradient(${color} ${percent}%, #1e293b ${percent}%)`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            transition: "background 0.5s ease"
          }}
        >
          {/* Vòng tròn nhỏ bên trong để tạo hiệu ứng rỗng */}
          <div className="gauge-inner" style={{
            width: "85%",
            height: "85%",
            backgroundColor: "#ffffff", // Hoặc màu nền card của bạn
            borderRadius: "50%",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "inset 0 2px 4px rgba(0,0,0,0.1)"
          }}>
            <span style={{ fontSize: "32px", fontWeight: "bold", color: "#1e293b" }}>{score}</span>
            <span style={{ fontSize: "10px", color: "#94a3b8" }}>/ 10</span>
          </div>
        </div>
      </div>

      {/* 🏆 Trạng thái văn bản */}
      <h2 style={{ color: color, fontSize: "24px", margin: "10px 0" }}>
        {score >= 7 ? "Excellent 🍷" : score >= 5 ? "Average 🧪" : "Poor ⚠️"}
      </h2>

      {/* 📝 Mô tả chi tiết */}
      <p style={{ fontSize: "14px", color: "#475569", lineHeight: "1.6", maxWidth: "250px", margin: "0 auto" }}>
        {score >= 7
          ? "A refined wine with balanced chemical composition and premium quality."
          : score >= 5
          ? "Moderate quality with standard indicators, suitable for general consumption."
          : "Below standard quality, the brewing process or alcohol content needs revision."}
      </p>
    </div>
  );
}

export default ResultCard;