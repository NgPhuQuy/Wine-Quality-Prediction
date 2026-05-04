import React from "react";

const HistoryList = ({ historyData }) => {
  if (!historyData || historyData.length === 0) return null;

  return (
    <div className="elegant-history-wrapper" style={{ marginTop: '40px' }}>
      <h2 style={{ marginBottom: '20px', fontSize: '18px', color: '#1e293b' }}>Prediction history</h2>
      <div className="elegant-grid">
        {historyData.map((item, index) => {
          
          // --- LOGIC ĐỌC DỮ LIỆU TỪ LOG CONSOLE CỦA BẠN ---
          
          // 1. Lấy object chứa kết quả (Ưu tiên lấy trong data.data nếu có)
          const resultObj = item.data ? item.data : item;

          // 2. Lấy raw_score (xác suất) để tính điểm hệ 10
          // Mình thêm kiểm tra các tên biến có thể có từ Backend
          const raw = parseFloat(resultObj.raw_score || resultObj.proba || 0);
          
          // 3. Tính điểm hệ 10 (35% -> 3.5)
          const score = (raw * 10).toFixed(1); 
          const percent = Math.round(raw * 100);
          
          // 4. Phân loại (Dựa trên quality_score hoặc điểm >= 7)
          const isGood = resultObj.quality_score === 1 || parseFloat(score) >= 7;
          const status = isGood ? "GOOD" : "NORMAL";

          return (
            <div key={index} className="elegant-card">
              <div className="card-header">
                <span className="batch-tag">Sample #{historyData.length - index}</span>
                <div className={`status-pill ${status.toLowerCase()}`}>{status}</div>
              </div>

              <div className="card-main">
                <div className="accuracy-small">Reliability: {percent}%</div>
                {/* Hiển thị con số 3.5 hoặc 8.2... */}
                <h2 className="score-display">{score}</h2>
                <p className="score-label">ĐIỂM HỆ 10</p>
              </div>

              <div className="card-metrics-grid">
                <div className="m-item">
                  <span>ALC</span>
                  <p>{resultObj.alcohol || "—"}%</p>
                </div>
                <div className="m-item">
                  <span>PH</span>
                  <p>{resultObj.ph || resultObj.pH || "—"}</p>
                </div>
                <div className="m-item">
                  <span>SUGAR</span>
                  <p>{resultObj.residual_sugar || "—"}</p>
                </div>
                <div className="m-item">
                  <span>ACID</span>
                  <p>{resultObj.volatile_acidity || "—"}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default HistoryList;