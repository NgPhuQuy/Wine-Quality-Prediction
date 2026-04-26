export const predictWine = async (formData) => {
  try {
    // 1. Lấy user_id từ "túi" localStorage ra
    const userId = localStorage.getItem("user_id");

    // 2. Kiểm tra nếu chưa đăng nhập thì bắt đi đăng nhập ngay
    if (!userId) {
      alert("Lỗi!");
      return null;
    }

    // 3. Gom 11 thông số từ Form + user_id vào một gói duy nhất
    const dataToSend = {
      ...formData,
      user_id: parseInt(userId) // Ép kiểu về số (Integer) cho đúng yêu cầu của An/Đạt
    };

    // 4. Gửi lên Backend (Lưu ý dấu / ở cuối URL để tránh lỗi Redirect)
    const res = await fetch("http://127.0.0.1:8000/predict/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(dataToSend),
    });

    // 5. Nếu lỗi 422/500, lấy chi tiết lỗi để debug cho dễ
    if (!res.ok) {
      const errorDetail = await res.json();
      console.error("Lỗi chi tiết từ Backend:", errorDetail);
      throw new Error(errorDetail.detail || "API error");
    }

    const data = await res.json();
    
    if (data.status === "success") {
      return data; 
    }
    throw new Error("Invalid response");
  } catch (error) {
    console.error("Lỗi tại predictWine:", error);
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