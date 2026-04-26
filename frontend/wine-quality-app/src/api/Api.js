// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────

const BASE_URL = "http://127.0.0.1:8000";

/** Lire user_id depuis localStorage (utilisé par plusieurs fonctions) */
export const getStoredUserId = () => localStorage.getItem("user_id");

/** Sauvegarder la session après login */
const saveSession = (data) => {
  localStorage.setItem("user_id",  String(data.user_id));
  localStorage.setItem("username", data.username);
};

/** Effacer la session lors du logout */
export const clearSession = () => {
  localStorage.removeItem("user_id");
  localStorage.removeItem("username");
};

// ─────────────────────────────────────────────
//  Auth
// ─────────────────────────────────────────────

export const registerUser = async (userData) => {
  try {
    const res = await fetch(`${BASE_URL}/auth/register`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(userData),
    });

    const data = await res.json();
    console.log("=== API DATA ===", data);
    console.log("=== RES STATUS ===", res.status);
    console.log("=== DATA STATUS ===", data?.status);
    console.log("=== DEBUG ===");
    console.log("HTTP status:", res.status);
    console.log("Data nhận được:", data);
    console.log("quality_score:", data?.quality_score);
    console.log("status:", data?.status);
    console.log("=============");
    if (!res.ok) {
      // Le backend renvoie { detail: "..." } en cas d'erreur FastAPI
      throw new Error(data.detail || "Erreur lors de l'inscription");
    }

    return { success: true, data };
  } catch (error) {
    console.error("Lỗi đăng ký:", error);
    return { success: false, message: error.message };
  }
};

export const loginUser = async (loginData) => {
  try {
    const res = await fetch(`${BASE_URL}/auth/login`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(loginData),
    });

    const data = await res.json();

    if (!res.ok) {
      // 401 → "Sai tài khoản hoặc mật khẩu..."
      throw new Error(data.detail || "Sai tài khoản hoặc mật khẩu");
    }

    // ✅ FIX : sauvegarder la session pour survivre au rechargement
    saveSession(data);

    return { success: true, data };
  } catch (error) {
    console.error("Lỗi đăng nhập:", error);
    return { success: false, message: error.message };
  }
};

export const logoutUser = async () => {
  try {
    await fetch(`${BASE_URL}/auth/logout`, { method: "POST" });
  } catch (_) {
    // Même si le réseau échoue, on efface la session locale
  } finally {
    // ✅ FIX : toujours nettoyer le localStorage au logout
    clearSession();
  }
};

// ─────────────────────────────────────────────
//  Prédiction
// ─────────────────────────────────────────────

export const predictWine = async (formData) => {
  try {
    const userId = getStoredUserId();

    if (!userId) {
      // Session expirée → renvoyer une erreur propre, pas juste alert()
      return { success: false, message: "Session expirée, veuillez vous reconnecter." };
    }

    const dataToSend = {
      ...formData,
      user_id: parseInt(userId, 10),
    };

    const res = await fetch(`${BASE_URL}/predict/`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(dataToSend),
    });

    const data = await res.json();

    if (!res.ok) {
      console.error("Lỗi chi tiết từ Backend:", data);
      throw new Error(data.detail || "Erreur API predict");
    }

   if (!data || data.status === "error") {
  throw new Error(data?.message || "Réponse invalide du serveur");
}

    return { success: true, data };
  } catch (error) {
    console.error("Lỗi tại predictWine:", error);
    return { success: false, message: error.message };
  }
};

// ─────────────────────────────────────────────
//  Model Info
// ─────────────────────────────────────────────

export const getModelInfo = async () => {
  try {
    // ✅ FIX : URL alignée avec le backend (/model-info ou /model/metadata — à choisir)
    const res = await fetch(`${BASE_URL}/model-info`);

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Erreur API model-info");
    }

    if (data.status !== "success") {
      throw new Error(data.message || "Phản hồi máy chủ không hợp lệ");
    }

    return { success: true, data };
  } catch (error) {
    console.error("Lỗi khi lấy thông tin mô hình:", error);
    // ✅ FIX : retourner un objet structuré plutôt que null
    return { success: false, message: error.message };
  }
};