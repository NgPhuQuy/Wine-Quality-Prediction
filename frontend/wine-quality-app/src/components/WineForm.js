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

  const getDefaultForm = () => {
    const obj = {};
    Object.keys(ranges).forEach((key) => { obj[key] = 0; });
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
      } else { throw new Error("Invalid response"); }
    } catch (error) {
      console.error(error);
      alert("API server connection error!");
    } finally { setLoading(false); }
  };

  // Định nghĩa style cho nút + và -
  const btnStyle = {
    width: "30px",
    height: "30px",
    borderRadius: "50%",
    border: "1px solid #e2e8f0",
    background: "white",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: "bold",
    fontSize: "18px",
    color: "#475569",
    transition: "all 0.2s"
  };

  return (
    <div className="wine-form-container">
      <div style={{ marginBottom: "25px", display: "flex", gap: "12px" }}>
        <button type="button" onClick={() => setWineType("red")} style={{ flex: 1, padding: "12px", borderRadius: "12px", background: wineType === "red" ? "#ef4444" : "#f1f5f9", color: wineType === "red" ? "white" : "#64748b", border: "none", fontWeight: "600", cursor: "pointer" }}>
          🍷 Red Wine
        </button>
        <button type="button" onClick={() => setWineType("white")} style={{ flex: 1, padding: "12px", borderRadius: "12px", background: wineType === "white" ? "#38bdf8" : "#f1f5f9", color: wineType === "white" ? "white" : "#64748b", border: "none", fontWeight: "600", cursor: "pointer" }}>
          🥂 White Wine
        </button>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
          {Object.keys(form).map((key) => {
            const min = ranges[key][0];
            const max = ranges[key][1];
            const value = form[key];
            const percent = ((value - min) / (max - min)) * 100;
            const color = wineType === "red" ? "#ef4444" : "#38bdf8";

            return (
              <div className="input-group" key={key} style={{ marginBottom: "15px" }}>
                <label style={{ display: "block", marginBottom: "8px", fontSize: "13px", color: "#475569" }}>
                  {key.replaceAll("_", " ").toUpperCase()}: <b style={{ color: color }}>{value}</b>
                </label>

                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  {/* Nút - */}
                  <button
                    type="button"
                    onClick={() => handleChange(key, Math.max(min, value - 0.1))} // Tăng bước nhảy lên 0.1 cho dễ bấm
                    style={btnStyle}
                    onMouseOver={(e) => e.target.style.background = "#f8fafc"}
                    onMouseOut={(e) => e.target.style.background = "white"}
                  > - </button>

<input
  type="range"
  min={min}
  max={max}
  step="0.001"
  value={value}
  onChange={(e) => handleChange(key, e.target.value)}
  style={{
    flex: 1,
    background: `linear-gradient(
      90deg,
      ${color} 0%,
      ${color} ${percent}%,
      #e2e8f0 ${percent}%,
      #e2e8f0 100%
    )`
  }}
/>  

                  {/* Nút + */}
                  <button
                    type="button"
                    onClick={() => handleChange(key, Math.min(max, value + 0.1))}
                    style={btnStyle}
                    onMouseOver={(e) => e.target.style.background = "#f8fafc"}
                    onMouseOut={(e) => e.target.style.background = "white"}
                  > + </button>
                </div>
              </div>
            );
          })}
        </div>

        <button type="submit" style={{ marginTop: "30px", width: "100%", padding: "15px", borderRadius: "16px", border: "none", background: "linear-gradient(135deg, #1e293b, #334155)", color: "white", fontSize: "16px", fontWeight: "bold", cursor: "pointer" }}>
          ANALYZE QUALITY
        </button>
      </form>
    </div>
  );
}

export default WineForm;