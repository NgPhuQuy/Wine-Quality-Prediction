import React, { useState, useEffect } from "react";
import WineForm from "./components/WineForm";
import ResultCard from "./components/ResultCard";
import ModelInfo from "./components/ModelInfo";
import WineChart from "./components/WineChart";
import HistoryList from "./components/HistoryList";
import { logoutUser } from "./api/Api";
import "./styles/App.css";

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [chartData, setChartData] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  const [username, setUsername] = useState("");
  const [predictionHistory, setPredictionHistory] = useState([]);

  useEffect(() => {
    const storedUsername = localStorage.getItem("username");
    if (storedUsername) setUsername(storedUsername);
  }, []);

  // LOGIC XỬ LÝ DỮ LIỆU: Đã được fix để nhận đúng số thực
  const handleNewPrediction = (apiResponse, inputData) => {
    console.log("App nhận dữ liệu:", apiResponse);
    
    // Lưu toàn bộ response (bao gồm quality_score và raw_score) vào state
    setResult(apiResponse);
    setChartData(inputData);
    
    const historyEntry = {
      ...apiResponse,
      input_data: inputData,
    };
    setPredictionHistory(prev => [historyEntry, ...prev]);
  };

  const handleLogout = async () => {
    if (window.confirm("Are you sure you want to log out?")) {
      await logoutUser();
      window.location.href = "/"; 
    }
  };

  return (
    <div className="App-Dashboard">
      <aside className="sidebar">
        <h1 className="main-title">Wine <span>Quality</span> Predict</h1>
        <ModelInfo />
      </aside>

      <main className="main-content">
        <header className="app-header">
          <div className="user-badge">
            <div className="user-icon-circle">
              <img src="https://cdn-icons-png.flaticon.com/512/1144/1144760.png" alt="user" />
            </div>
            <span className="username-text">{username || "User"}</span>
          </div>

          <div className="header-actions">
            <button className="btn-modern btn-history" onClick={() => setShowHistory(!showHistory)}>
              {showHistory ? "⬅ Go back to" : " Predicted Pattern History"}
            </button>
            <button className="btn-modern btn-logout" onClick={handleLogout}> LOG OUT</button>
          </div>
        </header>

        {showHistory ? (
          <HistoryList historyData={predictionHistory} />
        ) : (
          <>
            <div className="form-card">
              <h2 className="section-title">Wine sample parameters</h2>
              {/*  Truyền đúng hàm handleNewPrediction vào đây */}
              <WineForm 
                setResult={handleNewPrediction} 
                setLoading={setLoading} 
                setChartData={setChartData} 
              />
            </div>

            {loading && <div className="spinner"></div>}

            {!loading && result && (
              <div className="result-chart-layout">
                <ResultCard result={result} />
                <div className="chart-card">
                  <WineChart data={chartData} />
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

export default App;