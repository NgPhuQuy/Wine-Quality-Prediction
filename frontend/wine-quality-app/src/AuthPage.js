import React, { useState } from "react";

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

  const handleSubmit = () => {
    if (isRegister) {
      localStorage.setItem("user", JSON.stringify(form));
      alert("Register success!");
      setIsRegister(false);
    } else {
      const saved = JSON.parse(localStorage.getItem("user"));

      if (
        saved &&
        form.username === saved.username &&
        form.password === saved.password
      ) {
        onLogin();
      } else {
        alert("Wrong username or password!");
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