import React,{useState} from "react"
import { predictWine } from "../api/mockApi"

function WineForm({setResult,setLoading,setChartData}){

const [form,setForm] = useState({

fixed_acidity:"",
volatile_acidity:"",
citric_acid:"",
residual_sugar:"",
chlorides:"",
free_sulfur_dioxide:"",
total_sulfur_dioxide:"",
density:"",
pH:"",
sulphates:"",
alcohol:""

})

const handleChange=(e)=>{

setForm({
...form,
[e.target.name]:e.target.value
})

}

const handleSubmit=async(e)=>{

e.preventDefault()

for(const key in form){
if(form[key]===""){
alert("Please fill all fields")
return
}
}

setLoading(true)

setChartData(form)

try{

const result = await predictWine(form)

setResult(result.prediction)

}catch{

alert("Prediction error")

}

setLoading(false)

}
return (
  <form onSubmit={handleSubmit} className="form">

    <div className="grid">

      {Object.keys(form).map((key) => (
        <div className="input-group" key={key}>
          <label>{key.replaceAll("_", " ")}</label>
          <input
            type="number"
            name={key}
            value={form[key]}
            onChange={handleChange}
            placeholder="Enter value..."
          />
        </div>
      ))}

    </div>

    <button type="submit" className="btn">
       Predict Now
    </button>

  </form>
)
}

export default WineForm