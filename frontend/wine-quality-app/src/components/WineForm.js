import React, { useState } from "react";
import { predictWine } from "../api/mockApi";

const ranges = {
  fixed_acidity: [4, 16],
  volatile_acidity: [0.1, 1.5],
  citric_acid: [0, 1],
  residual_sugar: [0.5, 65],
  chlorides: [0.01, 0.2],
  free_sulfur_dioxide: [1, 300],
  total_sulfur_dioxide: [6, 450],
  density: [0.987, 1.005],
  pH: [2.8, 4],
  sulphates: [0.2, 2],
  alcohol: [8, 15]
};

function WineForm({ setResult, setLoading, setChartData }) {
  const [wineType, setWineType] = useState("red");

  // Hàm khởi tạo giá trị mặc định bằng 0
  const getDefaultForm = () => {
    const obj = {};
    Object.keys(ranges).forEach((key) => {
      obj[key] = 0; 
    });
    return obj;
  };

  const [form, setForm] = useState(getDefaultForm());

  const handleChange = (key, value) => {
    const num = parseFloat(value);
    setForm((prev) => ({
      ...prev,
      [key]: isNaN(num) ? 0 : Number(num.toFixed(3)),
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setChartData(form);

    try {
      const response = await predictWine(form);
      if (response && response.quality !== undefined) {
        setResult(response.quality);
      } else {
        throw new Error("Invalid response");
      }
    } catch (error) {
      console.error(error);
      alert("API server connection error!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="wine-form-container">
      {/* Chọn loại rượu - Nút bấm thiết kế tinh tế hơn */}
      <div style={{ marginBottom: "25px", display: "flex", gap: "12px", justifyContent: "center" }}>
        <button
          type="button"
          onClick={() => setWineType("red")}
          style={{
            flex: 1,
            padding: "12px",
            borderRadius: "12px",
            background: wineType === "red" ? "linear-gradient(135deg, #ef4444, #991b1b)" : "#f1f5f9",
            color: wineType === "red" ? "white" : "#64748b",
            border: "none",
            fontWeight: "600",
            cursor: "pointer",
            transition: "all 0.3s ease"
          }}
        >
          🍷 Red Wine
        </button>

        <button
          type="button"
          onClick={() => setWineType("white")}
          style={{
            flex: 1,
            padding: "12px",
            borderRadius: "12px",
            background: wineType === "white" ? "linear-gradient(135deg, #38bdf8, #075985)" : "#f1f5f9",
            color: wineType === "white" ? "white" : "#64748b",
            border: "none",
            fontWeight: "600",
            cursor: "pointer",
            transition: "all 0.3s ease"
          }}
        >
          🥂 White Wine
        </button>
      </div>

      {/* Form Slider */}
      <form onSubmit={handleSubmit}>
        <div className="grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          {Object.keys(form).map((key) => {
            const min = ranges[key][0];
            const max = ranges[key][1];
            const value = form[key];
            const percent = ((value - min) / (max - min)) * 100;
            const color = wineType === "red" ? "#ef4444" : "#38bdf8";

            return (
              <div className="input-group" key={key} style={{ marginBottom: '10px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', color: '#475569' }}>
                  {key.replaceAll("_", " ").toUpperCase()}: <b style={{ color: color }}>{value}</b>
                </label>

                <input
                  type="range"
                  min={min}
                  max={max}
                  step="0.001"
                  value={value}
                  onChange={(e) => handleChange(key, e.target.value)}
                  style={{
                    width: '100%',
                    cursor: 'pointer',
                    accentColor: color,
                    background: `linear-gradient(90deg, ${color} ${percent}%, #e2e8f0 ${percent}%)`
                  }}
                />

                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: "11px", color: "#94a3b8", marginTop: '4px' }}>
                  <span>{min}</span>
                  <span>{max}</span>
                </div>
              </div>
            );
          })}
        </div>

        <button
          type="submit"
          className="btn-predict"
          style={{
            marginTop: "30px",
            width: "100%",
            padding: "15px",
            borderRadius: "16px",
            border: "none",
            background: "linear-gradient(135deg, #1e293b, #334155)",
            color: "white",
            fontSize: "16px",
            fontWeight: "bold",
            letterSpacing: "1px",
            cursor: "pointer",
            boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1)"
          }}
        >
          ANALYZE QUALITY
        </button>
      </form>
    </div>
  );
}

export default WineForm;