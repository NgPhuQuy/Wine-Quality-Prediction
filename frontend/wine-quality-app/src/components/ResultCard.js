import React from "react";

function ResultCard({ result }) {
  if (!result) return null;

  const isGood = result.includes("Good");

  return (
    <div className="result-card" style={{ textAlign: "center" }}>
      <p style={{ color: "#64748b", fontSize: "12px", fontWeight: "bold", marginBottom: "10px" }}>
        ANALYSIS RESULTS
      </p>
      <div style={{ fontSize: "40px", marginBottom: "10px" }}>
        {isGood ? "🍷" : "🧪"}
      </div>
      <h2 className={isGood ? "status-good" : "status-bad"}>
        {result}
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