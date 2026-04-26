export const predictWine = async (formData) => {
  try {
    const userId = localStorage.getItem("user_id");
    if (!userId) {
      alert("Bạn chưa đăng nhập!");
      return null;
    }

    const dataToSend = {
      ...formData,
      user_id: parseInt(userId)
    };

    const res = await fetch("http://127.0.0.1:8000/predict/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(dataToSend),
    });

    if (!res.ok) {
      const errorDetail = await res.json();
      console.error("Lỗi chi tiết từ Backend:", errorDetail);
      throw new Error(errorDetail.detail || "API error");
    }

    const data = await res.json();
    // FIX: Backend trả về WineResponse (có id, quality_score, created_at),
    // không có field "status". Thêm status thủ công để WineForm xử lý được.
    return { ...data, status: "success" };

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
    if (data.status === "success") return data;
    throw new Error(data.message || "Invalid response");
  } catch (error) {
    console.error("Lỗi khi lấy thông tin mô hình:", error);
    return null;
  }
};

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
