// export const predictWine = async (data) => {
//   console.log("Data :", data)
//export const predictWine = async (data) => {
  //console.log("Data :", data)

//   return new Promise((resolve) => {
//     setTimeout(() => {

//       const alcohol = parseFloat(data.alcohol)

//       let prediction = "Bad ❌"

//       if (alcohol > 10) {
//         prediction = "Good 🍷"
//       }

//       resolve({
//         prediction: prediction
//       })

// src/api/mockApi.js

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
    console.warn("⚠️ API lỗi → dùng mock local");

    // ===== 🤖 FAKE AI (fallback) =====
    let score = 0;

    score += formData.alcohol * 0.3;
    score += (1 - Math.abs(formData.pH - 3.3)) * 2;
    score += formData.sulphates * 1.5;
    score -= formData.residual_sugar * 0.05;

    let quality = Math.min(9, Math.max(3, Math.round(score)));

    return {
      quality: quality,
      source: "mock"
    };
  }
};


// info model (giữ nguyên nhưng đẹp hơn)
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
