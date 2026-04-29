import React from "react";

function ResultCard({ result }) {
  // Nếu không có result hoặc result.data thì không hiện
  if (!result || !result.data) return null;

  // Lấy dữ liệu từ bên trong object 'data' theo đúng log Console của bạn
  const innerData = result.data;
  
  // Lấy raw_score (xác suất) để tính %
  const proba = parseFloat(innerData.raw_score || 0);
  const percent = Math.round(proba * 100);

  // Lấy quality_score (0 hoặc 1) để biết rượu tốt hay không
  const isGood = innerData.quality_score === 1;
  const color = isGood ? "#10b981" : "#94a3b8";

  return (
    <div className="result-card luxury" style={{ textAlign: "center", padding: "20px", background: "white", borderRadius: "20px" }}>
      <p style={{ color: "#94a3b8", fontSize: "11px", fontWeight: "bold", letterSpacing: "1px" }}>ANALYSIS RESULT</p>
      
      <div style={{
        width: "120px", height: "120px", margin: "20px auto", borderRadius: "50%",
        background: `conic-gradient(${color} ${percent}%, #f1f5f9 ${percent}%)`,
        display: "flex", alignItems: "center", justifyContent: "center",
        transition: "all 0.6s ease-in-out"
      }}>
        <div style={{ width: "85%", height: "85%", background: "#fff", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", fontWeight: "800", fontSize: "24px", color: "#1e293b" }}>
          {percent}%
        </div>
      </div>

      <h2 style={{ color: color, marginBottom: "5px" }}>
        {isGood ? "Good Quality Wine" : "Normal Quality Wine"}
      </h2>
      
      <p style={{ fontSize: "13px", color: "#64748b" }}>
        {percent === 0 
          ? "Backend returned 0. Please increase Alcohol/Sulphates." 
          : `Model confidence: ${percent}%`}
      </p>
    </div>
  );
}

export default ResultCard;