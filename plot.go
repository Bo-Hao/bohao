package bohao

import (
	"os"
	"strconv"

	"github.com/go-echarts/go-echarts/charts"
)

// DrawXYScatterPlot : 
func DrawXYScatterPlot(data [][]float64, savePlace string) {
	scatter := charts.NewScatter()
	
	if len(data) > 2 && len(data[0]) > 2 && len(data)%2 == 0{
		NSets := len(data)/2
		for i := 0; i < NSets; i ++{
			x := data[2*i]
			y := data[2*i + 1]
			scatter.AddYAxis("Set " + strconv.Itoa(i + 1), TransposeFloat([][]float64{x, y}))
		}
	}else if len(data[0]) > 2 && len(data)%2 == 0{
		x := data[0]
		y := data[1]
		scatter.AddYAxis("Set 1", TransposeFloat([][]float64{x, y}))
	}else if len(data[0]) == 2 && len(data) > 2{
		scatter.AddYAxis("Set 1", data)
	}

	scatter.PageTitle = "Bo Hao Scatter"
	scatter.SetGlobalOptions(charts.TitleOpts{Title: "Scatter"}, charts.YAxisOpts{Scale: true})

	h, _ := os.Create(savePlace)
	scatter.Render(h)
}

// DrawLinePlot : 
func DrawLinePlot(data [][]float64, savePlace string) {
	Line := charts.NewLine()
	
	if len(data) > 2 && len(data[0]) > 2 && len(data)%2 == 0{
		NSets := len(data)/2
		for i := 0; i < NSets; i ++{
			x := data[2*i]
			y := data[2*i + 1]
			Line.AddYAxis("Set " + strconv.Itoa(i + 1), TransposeFloat([][]float64{x, y}))
		}
	}else if len(data[0]) > 2 && len(data)%2 == 0{
		x := data[0]
		y := data[1]
		Line.AddYAxis("Set 1", TransposeFloat([][]float64{x, y}))
	}else if len(data[0]) == 2 && len(data) > 2{
		Line.AddYAxis("Set 1", data)
	}

	Line.PageTitle = "Bo Hao plot"
	Line.SetGlobalOptions(charts.TitleOpts{Title: "Plot"}, charts.YAxisOpts{Scale: true})

	h, _ := os.Create(savePlace)
	Line.Render(h)
}
