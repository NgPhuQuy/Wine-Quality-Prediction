export const predictWine = async (formData) => {
  try {
    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    });

    if (!res.ok) throw new Error("API error");

    const data = await res.json();
    
    // BE trả về "status": "success", ta nên kiểm tra nó
    if (data.status === "success") {
      return data; 
    }
    throw new Error("Invalid response");
  } catch (error) {
    console.error(error);
    return null;
  }
};


export const getModelInfo = async () => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        model: "RandomForestClassifier",
        accuracy: "0.87",
        features: 11,
        description: "Predicts wine quality based on physicochemical properties."
      });
    }, 500);
  });
};
