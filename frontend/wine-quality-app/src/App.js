import React, { useState } from "react";
import WineForm from "./components/WineForm";
import ResultCard from "./components/ResultCard";
import ModelInfo from "./components/ModelInfo";
import WineChart from "./components/WineChart";
import "./styles/App.css";

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [chartData, setChartData] = useState(null);

  return (
    <div className="App-Dashboard">
      {}
      <aside className="sidebar">
        <h1 className="main-title">Wine <span>Quality</span> Predict</h1>
        <ModelInfo />
      </aside>

      {}
      <main className="main-content">
        <div className="form-card">
          <h2 style={{ marginBottom: "20px", fontSize: "18px" }}>Wine sample parameters</h2>
          <WineForm
            setResult={setResult}
            setLoading={setLoading}
            setChartData={setChartData}
          />
        </div>

        {loading && <div className="spinner"></div>}

        {!loading && result && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1.5fr", gap: "24px" }}>
            <ResultCard result={result} />
            <div className="chart-card">
              <WineChart data={chartData} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;