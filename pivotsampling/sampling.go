package pivotsampling

import (
	"bohao"
	"fmt"
	"math"
	"math/rand"
)

func Version() {
	fmt.Println("1.0 beta")
}

func dist(pt1, pt2 []float64) float64 {
	sum := 0.
	for i := 0; i < len(pt1); i++ {
		sum += math.Pow(pt1[i]-pt2[i], 2)
	}
	return math.Sqrt(sum)
}

func clipMatrix(positionMatrix [][][]float64, index int) [][][]float64 {
	positionMatrix = append(positionMatrix[:index], positionMatrix[index+1:]...)
	for i := 0; i < len(positionMatrix); i ++{
		positionMatrix[i] = append(positionMatrix[i][:index], positionMatrix[i][index+1:]...)
	}
	return positionMatrix
}

func searchSmallest(sli [][]float64) int {
	infValue := sli[0][0]
	if infValue == -1{
		infValue = sli[1][0]
	} 
	var smallestIndex []int
	for i := 0; i < len(sli); i ++{
		if sli[i][0] < 0{
			continue
		}else if sli[i][0] < infValue {
			infValue = sli[i][0]
			smallestIndex = []int{i}
		}else if sli[i][0] == infValue {
			smallestIndex = append(smallestIndex, i)
		}
	}

	if len(smallestIndex) == 1 {
		fmt.Println(len(smallestIndex), len(sli))
		return smallestIndex[0]
	}else if len(sli) == 1 {
		fmt.Println(len(smallestIndex), len(sli))
		return -1 
	}else {
		fmt.Println(len(smallestIndex), len(sli))
		return smallestIndex[rand.Intn(len(smallestIndex))]
	}
}


func LocalPivotSampling(position [][]float64, incluProb []float64) []int {
	NUnit := len(position)
	if len(incluProb) == 1{
		prob := incluProb[0]
		var incluProb []float64
		for i := 0; i < NUnit; i ++{
			incluProb = append(incluProb, prob)
		}
	}

	var positionMatrix [][][]float64
	for i := 0; i < NUnit; i ++{
		tmp := make([][]float64, NUnit)
		positionMatrix = append(positionMatrix, tmp)
	}

	for i := 0; i < NUnit; i++ {
		for j := i; j < NUnit; j++ {
			if i == j {
				positionMatrix[i][i] = []float64{-1., float64(i), float64(j)}
			} else {
				distance := dist(position[i], position[j])
				positionMatrix[i][j] = []float64{distance, float64(i), float64(j)}
				positionMatrix[j][i] = []float64{distance, float64(j), float64(i)}
			}
		}
	}


	for len(positionMatrix) > 1 {
		sampleIndex := rand.Intn(len(positionMatrix))
		choose := searchSmallest(positionMatrix[sampleIndex])
		if choose == -1{
			break
		}

		pair := positionMatrix[sampleIndex][choose]
		idx1 := int(pair[1])
		idx2 := int(pair[2])

		prob1 := incluProb[idx1]
		prob2 := incluProb[idx2]
		totalProb := prob1 + prob2

		// updating
		if totalProb < 1 {
			if rand.Float64() < prob2/totalProb {
				prob1 = 0
				prob2 = totalProb
			} else {
				prob1 = totalProb
				prob2 = 0
			}
		} else if totalProb >= 1 {
			if rand.Float64() < (1-prob2)/(2-totalProb) {
				prob1 = 1
				prob2 = totalProb - 1
			} else {
				prob1 = totalProb - 1
				prob2 = 1
			}
		}

		incluProb[idx1] = prob1 
		incluProb[idx2] = prob2

		var case1, case2 bool 
		if prob1 == 0 || prob1 == 1{
			case1 = true
		}
		if prob2 == 0 || prob2 == 1{
			case2 = true
		}

		if case1 && case2 {
			if choose > sampleIndex{
				positionMatrix = clipMatrix(positionMatrix, choose)
				positionMatrix = clipMatrix(positionMatrix, sampleIndex)
			}else {
				positionMatrix = clipMatrix(positionMatrix, sampleIndex)
				positionMatrix = clipMatrix(positionMatrix, choose)
			}
		}else if case1 {
			positionMatrix = clipMatrix(positionMatrix, sampleIndex)
		}else if case2 {
			positionMatrix = clipMatrix(positionMatrix, choose)
		}
	}


	

	return []int{1, 2, 3}
}

func Do() {
	var data, sample [][]float64

	for i := 0; i < 625; i++ {
		data = append(data, []float64{float64(i), float64(20. / 625.)})
	}

	for len(data) > 1 {
		sampleIndex := rand.Intn(len(data))

		var choose int
		if sampleIndex+1 > len(data)-1 && sampleIndex-1 < 0 {
			break
		} else if sampleIndex-1 < 0 {
			choose = sampleIndex + 1
		} else if sampleIndex+1 > len(data)-1 {
			choose = sampleIndex - 1
		} else {
			point := data[sampleIndex][0]
			point1 := data[sampleIndex-1][0]
			point2 := data[sampleIndex+1][0]
			if math.Abs(point1-point) > math.Abs(point2-point) {
				choose = sampleIndex + 1
			} else if math.Abs(point1-point) < math.Abs(point2-point) {
				choose = sampleIndex - 1
			} else {
				if rand.Float64() > 0.5 {
					choose = sampleIndex - 1
				} else {
					choose = sampleIndex + 1
				}
			}
		}

		// updating
		uniti := data[sampleIndex]
		unitj := data[choose]

		totalProb := uniti[1] + unitj[1]
		if totalProb < 1 {
			if rand.Float64() < unitj[1]/totalProb {
				uniti[1] = 0
				unitj[1] = totalProb
			} else {
				uniti[1] = totalProb
				unitj[1] = 0
			}
		} else if uniti[1]+unitj[1] >= 1 {
			if rand.Float64() < (1-unitj[1])/(2-totalProb) {
				uniti[1] = 1
				unitj[1] = totalProb - 1
			} else {
				uniti[1] = totalProb - 1
				unitj[1] = 1
			}
		}

		// check done
		data[sampleIndex] = uniti
		data[choose] = unitj

		if uniti[1] == 1 {
			sample = append(sample, uniti)
		}
		if unitj[1] == 1 {
			sample = append(sample, unitj)
		}

		if data[sampleIndex][1] == 0 || data[sampleIndex][1] == 1 {
			data = append(data[:sampleIndex], data[sampleIndex+1:]...)
		} else if data[choose][1] == 0 || data[choose][1] == 1 {
			data = append(data[:choose], data[choose+1:]...)
		}
	}
	fmt.Println(sample)
	fmt.Println(len(sample))
	fmt.Println("done")
	bohao.DrawXYScatterPlot(sample, "/Users/pengbohao/fushan/sequentialWork/all_combination/plot.html")
}
