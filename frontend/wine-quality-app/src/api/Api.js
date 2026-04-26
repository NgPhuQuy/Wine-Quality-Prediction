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
// Thêm hàm Đăng ký
export const registerUser = async (userData) => {
  try {
    const res = await fetch("http://127.0.0.1:8000/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(userData),
    });
    return await res.json();
  } catch (error) {
    console.error("Lỗi đăng ký:", error);
    return null;
  }
};

// Thêm hàm Đăng nhập
export const loginUser = async (loginData) => {
  try {
    const res = await fetch("http://127.0.0.1:8000/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(loginData),
    });
    return await res.json();
  } catch (error) {
    console.error("Lỗi đăng nhập:", error);
    return null;
  }
};