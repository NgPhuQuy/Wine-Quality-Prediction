import React from "react";

function ResultCard({ result }) {
  if (result === null || result === undefined) return null;

  // đảm bảo là number
  const score = Number(result);

  // logic đánh giá
  const isGood = score >= 7;

  const getLabel = () => {
    if (score >= 7) return "Good 🍷";
    if (score >= 5) return "Average 🧪";
    return "Bad ⚠️";
  };

  return (
    <div className="result-card" style={{ textAlign: "center" }}>
      <p style={{ color: "#64748b", fontSize: "12px", fontWeight: "bold", marginBottom: "10px" }}>
        ANALYSIS RESULTS
      </p>

      <div style={{ fontSize: "40px", marginBottom: "10px" }}>
        {isGood ? "🍷" : "🧪"}
      </div>

      <h2 className={isGood ? "status-good" : "status-bad"}>
        {getLabel()} ({score})
      </h2>

      <p style={{ fontSize: "13px", marginTop: "15px", color: "#475569" }}>
        {isGood
          ? "The wine is of good quality, with chemical indicators within ideal limits."
          : "The wine quality is unsatisfactory; the brewing process or alcohol content needs to be re-examined."}
      </p>
    </div>
  );
}

export default ResultCard;