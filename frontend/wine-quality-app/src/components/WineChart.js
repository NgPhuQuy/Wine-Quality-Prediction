import React from "react"
import { Bar } from "react-chartjs-2"
import {
Chart as ChartJS,
CategoryScale,
LinearScale,
BarElement,
Title,
Tooltip,
Legend
} from "chart.js"

ChartJS.register(
CategoryScale,
LinearScale,
BarElement,
Title,
Tooltip,
Legend
)

function WineChart({data}){

if(!data) return null

const labels = Object.keys(data)

const chartData = {
labels: labels,
datasets: [
{
label: "Wine Features",
data: Object.values(data),
backgroundColor:"rgba(75,192,192,0.6)"
}
]
}

return(

<div style={{marginTop:"30px"}}>
<h3>Feature Visualization</h3>
<Bar data={chartData}/>
</div>

)

}

export default WineChart