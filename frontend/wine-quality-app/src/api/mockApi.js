export const predictWine = async (data) => {
  console.log("Data gửi lên:", data)

  return new Promise((resolve) => {
    setTimeout(() => {

      const alcohol = parseFloat(data.alcohol)

      let prediction = "Bad ❌"

      if (alcohol > 10) {
        prediction = "Good 🍷"
      }

      resolve({
        prediction: prediction
      })

    }, 1000)
  })
}

export const getModelInfo = async () => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        model: "Random Forest",
        accuracy: "87%",
        description: "Model dự đoán chất lượng rượu vang dựa trên các đặc trưng hóa học"
      })
    }, 500)
  })
}