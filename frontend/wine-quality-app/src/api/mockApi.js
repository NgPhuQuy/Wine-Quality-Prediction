// export const predictWine = async (data) => {
//   console.log("Data :", data)

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

//     }, 1000)
//   })
// }
export const predictWine = async (formData) => {
  const res = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(formData),
  });

  const data = await res.json();
  return data;
};

export const getModelInfo = async () => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        model: "Random Forest",
        accuracy: "87%",
        description: "This model predicts wine quality based on chemical characteristics."
      })
    }, 500)
  })
}