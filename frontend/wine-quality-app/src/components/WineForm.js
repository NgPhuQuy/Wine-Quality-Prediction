import React, { useState } from "react";
import { predictWine } from "../api/mockApi";

function WineForm({ setResult, setLoading, setChartData }) {
  const [form, setForm] = useState({
    fixed_acidity: "", volatile_acidity: "", citric_acid: "",
    residual_sugar: "", chlorides: "", free_sulfur_dioxide: "",
    total_sulfur_dioxide: "", density: "", pH: "", sulphates: "", alcohol: ""
  });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();


    if (Object.values(form).some(value => value === "")) {
      alert("Please fill in all the information!");
      return;
    }

    setLoading(true);
    setChartData(form);
    try {
      const response = await predictWine(form);
      setResult(response.prediction);
    } catch (error) {
      alert("API server connection error!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="grid">
        {Object.keys(form).map((key) => (
          <div className="input-group" key={key}>
            <label>{key.replaceAll("_", " ")}</label>
            <input
              type="number"
              step="any"
              name={key}
              value={form[key]}
              onChange={handleChange}
              placeholder="0.00"
            />
          </div>
        ))}
      </div>
      <button type="submit" className="btn-predict">QUALITY PREDICT</button>
    </form>
  );
}

export default WineForm;