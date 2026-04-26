import React, { useState } from "react";
import { registerUser, loginUser } from "./api/Api";
function AuthPage({ onLogin }) {
  const [isRegister, setIsRegister] = useState(true);

  const [form, setForm] = useState({
    email: "",
    phone: "",
    username: "",
    password: ""
  });

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value
    });
  };

 const handleSubmit = async () => {
  if (isRegister) {
    const data = await registerUser(form);
    if (data && !data.detail) {
      alert("Đăng ký thành công!");
      setIsRegister(false);
    } else {
      alert("Lỗi đăng ký!");
    }
  } else {
    // ĐĂNG NHẬP
    const data = await loginUser({
      username: form.username,
      password: form.password
    });

    
    if (data && data.message === "Đăng nhập thành công") {
      alert(data.username + "đăng nhập thành công!");
      

      localStorage.setItem("user_logged", data.username); 
      
      onLogin(); // Cho vào trang Dashboard ngay và luôn
    } else {
      alert("Sai tài khoản hoặc mật khẩu ");
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
          <span onClick={() => setIsRegister(!isRegister)}>
            {isRegister ? " Sign in" : " Register"}
          </span>
        </p>

      </div>
    </div>
  );
}

export default AuthPage;