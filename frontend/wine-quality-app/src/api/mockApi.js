export const predictWine = async (formData) => {
  try {
    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });

    // nếu API lỗi -> nhảy xuống catch
    if (!res.ok) {
      throw new Error("API error");
    }

    const data = await res.json();

    // đảm bảo đúng format
    if (data.quality !== undefined) {
      return data;
    }

    // nếu backend trả sai format → fallback
    throw new Error("Invalid response");

  } catch (error) {
    alert("API lỗi");

    let quality = 0;

    return {
      quality: quality,
      source: "mock"
    };
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
