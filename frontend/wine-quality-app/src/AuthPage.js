import React, { useState } from "react";
import { registerUser, loginUser } from "./api/Api";

function AuthPage({ onLogin }) {
  const [isRegister, setIsRegister] = useState(true);
  const [error, setError] = useState("");

  const [form, setForm] = useState({
    email: "",
    phone: "",
    username: "",
    password: ""
  });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setError(""); // xóa lỗi khi user gõ lại
  };

  const handleSubmit = async () => {
    setError("");

    if (isRegister) {
      // ── ĐĂNG KÝ ──────────────────────────────
      const result = await registerUser(form);
      if (result.success) {
        alert("Đăng ký thành công!");
        setIsRegister(false);
      } else {
        setError(result.message || "Lỗi đăng ký!");
      }

    } else {
      // ── ĐĂNG NHẬP ────────────────────────────
      const result = await loginUser({
        username: form.username,
        password: form.password
      });

      if (result.success) {
        onLogin(result.data); // báo App.js là đã đăng nhập thành công
      } else {
        setError(result.message || "Sai tài khoản hoặc mật khẩu!");
      }
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">

        <div className="auth-header">
          <h1 className="logo-text">
            <span className="logo-light">Wine</span>
            <span className="logo-accent">AI</span>
          </h1>
          <p className="intro">
            Sign in to access intelligent wine quality analysis powered by machine learning
          </p>
        </div>

        {/* Thông báo lỗi */}
        {error && (
          <p style={{ color: "var(--color-text-danger)", fontSize: "13px", margin: "0 0 10px" }}>
            {error}
          </p>
        )}

        {isRegister && (
          <>
            <input
              name="email"
              placeholder="Email address"
              onChange={handleChange}
            />
            <input
              name="phone"
              placeholder="Phone number"
              onChange={handleChange}
            />
          </>
        )}

        <input
          name="username"
          placeholder="Username"
          onChange={handleChange}
        />
        <input
          name="password"
          type="password"
          placeholder="Password"
          onChange={handleChange}
        />

        <button onClick={handleSubmit}>
          {isRegister ? "Create Account" : "Sign In"}
        </button>

        <p className="switch">
          {isRegister ? "Already have an account?" : "Don't have an account?"}
          <span onClick={() => { setIsRegister(!isRegister); setError(""); }}>
            {isRegister ? " Sign in" : " Register"}
          </span>
        </p>

      </div>
    </div>
  );
}

export default AuthPage;