import React, { useEffect, useState } from "react";
import { getModelInfo } from "../api/Api";

const featureDescriptions = {
  fixed_acidity:        "Tartaric acid concentration",
  volatile_acidity:     "Acetic acid (vinegar taste)",
  citric_acid:          "Freshness & flavor enhancer",
  residual_sugar:       "Remaining sugar after fermentation",
  chlorides:            "Salt content",
  free_sulfur_dioxide:  "Prevents oxidation",
  total_sulfur_dioxide: "Preservation level",
  density:              "Mass per volume",
  pH:                   "Acidity level",
  sulphates:            "Antimicrobial agent",
  alcohol:              "Alcohol percentage"
};

function ModelInfo() {
  const [data,  setData]  = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    // getModelInfo() retourne maintenant { success, data, message }
    getModelInfo().then((res) => {
      if (res.success) {
        setData(res.data); // res.data = { status, model_info, metadata }
      } else {
        setError(res.message || "Không tải được thông tin model.");
      }
    });
  }, []);

  // Đang tải
  if (!data && !error) {
    return <p style={{ color: "#94a3b8" }}>Loading model info...</p>;
  }

  // Lỗi
  if (error) {
    return <p style={{ color: "#f87171", fontSize: "13px" }}>{error}</p>;
  }

  // Lấy dữ liệu
  const { model_info, metadata } = data;

  return (
    <div className="model-info">
      <h3>Model Overview</h3>

      <p><b>Model:</b> {model_info.type}</p>

      <p>
        <b>Accuracy:</b>{" "}
        {metadata.metrics?.accuracy
          ? (metadata.metrics.accuracy * 100).toFixed(1) + "%"
          : "N/A"}
      </p>

      <p><b>Last Trained:</b> {metadata.train_date}</p>

      <hr style={{ margin: "15px 0", opacity: 0.2 }} />

      <h4>Dataset Features ({model_info.n_features_in} features)</h4>

      <div style={{ fontSize: "12px", lineHeight: "1.6" }}>
        {Object.entries(featureDescriptions).map(([key, val]) => (
          <p key={key}>
            <b style={{ textTransform: "capitalize" }}>{key.replaceAll("_", " ")}</b>: {val}
          </p>
        ))}
      </div>

      <hr style={{ margin: "15px 0", opacity: 0.2 }} />

      <p style={{ fontSize: "11px", color: "#94a3b8", fontStyle: "italic" }}>
        Status: {data.status} | Features used: {metadata.features?.join(", ")}
      </p>
    </div>
  );
}

export default ModelInfo;