import React from "react"

function ResultCard({result}){

if(!result) return null

return(

<div className="result">

<h2>Prediction Result</h2>

<h1>Wine Quality: {result}</h1>

</div>

)

}

export default ResultCard