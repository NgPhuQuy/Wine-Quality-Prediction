import React,{useEffect,useState} from "react"
import {getModelInfo} from "../api/mockApi"

function ModelInfo(){

const [model,setModel] = useState(null)

useEffect(()=>{

getModelInfo().then(data=>setModel(data))

},[])

if(!model) return null

return(

<div className="model-info">

<h3>Model Information</h3>

<p>Model: {model.model}</p>
<p>Accuracy: {model.accuracy}</p>
<p>Features: {model.features}</p>

</div>

)

}

export default ModelInfo