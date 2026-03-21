import React, { useState } from "react";
import { predictWine } from "../api/mockApi";

function WineForm({ setResult, setLoading, setChartData }) {
  const [form, setForm] = useState({
    fixed_acidity: "",
    volatile_acidity: "",
    citric_acid: "",
    residual_sugar: "",
    chlorides: "",
    free_sulfur_dioxide: "",
    total_sulfur_dioxide: "",
    density: "",
    pH: "",
    sulphates: "",
    alcohol: ""
  });

  const handleChange = (e) => {
    const { name, value } = e.target;

    // chỉ cho nhập số hợp lệ
    if (value === "" || !isNaN(value)) {
      setForm((prev) => ({
        ...prev,
        [name]: value
      }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // validate
    if (Object.values(form).some((value) => value === "")) {
      alert("Please fill in all the information!");
      return;
    }

    // convert sang number
    const formattedData = Object.fromEntries(
      Object.entries(form).map(([key, value]) => [key, Number(value)])
    );

    setLoading(true);
    setChartData(formattedData);

    try {
      const response = await predictWine(formattedData);

      // check response hợp lệ
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
              required
            />
          </div>
        ))}
      </div>

      <button type="submit" className="btn-predict">
        QUALITY PREDICT
      </button>
    </form>
  );
}

export default WineForm;