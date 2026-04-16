import React,{useEffect,useState} from "react"
import {getModelInfo} from "../api/Api"

const featureDescriptions = {
  fixed_acidity: "Tartaric acid concentration",
  volatile_acidity: "Acetic acid (vinegar taste)",
  citric_acid: "Freshness & flavor enhancer",
  residual_sugar: "Remaining sugar after fermentation",
  chlorides: "Salt content",
  free_sulfur_dioxide: "Prevents oxidation",
  total_sulfur_dioxide: "Preservation level",
  density: "Mass per volume",
  pH: "Acidity level",
  sulphates: "Antimicrobial agent",
  alcohol: "Alcohol percentage"
}

function ModelInfo(){

const [model,setModel] = useState(null)

useEffect(()=>{
  getModelInfo().then(data=>setModel(data))
},[])

if(!model) return null

return(

<div className="model-info">

<h3> Model Overview</h3>

<p><b>Model:</b> {model.model}</p>
<p><b>Accuracy:</b> {(model.accuracy * 100).toFixed(0)}%</p>
<p><b>Features:</b> {model.features}</p>

<hr style={{margin:"15px 0", opacity:0.2}}/>

<h4> Dataset Features</h4>

<div style={{fontSize:"12px", lineHeight:"1.6"}}>
  {Object.entries(featureDescriptions).map(([key,val])=>(
    <p key={key}>
      <b>{key.replaceAll("_"," ")}</b>: {val}
    </p>
  ))}
</div>

<hr style={{margin:"15px 0", opacity:0.2}}/>

<p style={{fontSize:"12px", color:"#94a3b8"}}>
  This model predicts wine quality based on physicochemical properties.
</p>

</div>

)

}

export default ModelInfo