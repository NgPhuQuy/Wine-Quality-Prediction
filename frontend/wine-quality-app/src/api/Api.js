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
  try {
    const res = await fetch("http://127.0.0.1:8000/model-info");

    if (!res.ok) throw new Error("API error fetching model info");

    const data = await res.json();

    if (data.status === "success") {
      return data; 
    }
    
    throw new Error(data.message || "Invalid response");
  } catch (error) {
    console.error("Lỗi khi lấy thông tin mô hình:", error);
    return null;
  }
};