package pivotalsampling

import (
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
	for i := 0; i < len(positionMatrix); i++ {
		positionMatrix[i] = append(positionMatrix[i][:index], positionMatrix[i][index+1:]...)
	}
	return positionMatrix
}

func searchSmallest(sli [][]float64) int {
	infValue := sli[0][0]
	if infValue == -1 {
		infValue = sli[1][0]
	}
	var smallestIndex []int
	for i := 0; i < len(sli); i++ {
		if sli[i][0] < 0 {
			continue
		} else if sli[i][0] < infValue {
			infValue = sli[i][0]
			smallestIndex = []int{i}
		} else if sli[i][0] == infValue {
			smallestIndex = append(smallestIndex, i)
		}
	}

	if len(smallestIndex) == 1 {
		return smallestIndex[0]
	} else if len(sli) == 1 {
		return -1
	} else {
		return smallestIndex[rand.Intn(len(smallestIndex))]
	}
}

func standardizedDist(position [][]float64) [][]float64 {
	pT := make([][]float64, len(position[0]))
	for i := 0; i < len(position); i++ {
		for j := 0; j < len(position[i]); j++ {
			pT[j] = append(pT[j], position[i][j])
		}
	}
	var maxList, minList []float64
	for i := 0; i < len(pT); i++ {
		max := pT[0][0]
		min := pT[0][0]
		for j := 0; j < len(pT[i]); j++ {
			if pT[i][j] > max {
				max = pT[i][j]
			} else if pT[i][j] < min {
				min = pT[i][j]
			}
		}
		maxList = append(maxList, max)
		minList = append(minList, min)
	}

	for i := 0; i < len(pT); i++ {
		mid := (maxList[i] + minList[i]) / 2.
		ran := maxList[i] - minList[i]
		for j := 0; j < len(pT[i]); j++ {
			pT[i][j] = (pT[i][j] - mid) / (ran / 2)
		}
	}

	standPosition := make([][]float64, len(pT[0]))
	for i := 0; i < len(pT); i++ {
		for j := 0; j < len(pT[i]); j++ {
			standPosition[j] = append(standPosition[j], pT[i][j])
		}
	}
	return standPosition
}

func LocalPivotalSampling(position [][]float64, incluProb []float64) []int {
	NUnit := len(position)
	if len(incluProb) == 1 {
		prob := incluProb[0]
		var incluProb []float64
		for i := 0; i < NUnit; i++ {
			incluProb = append(incluProb, prob)
		}
	}

	position = standardizedDist(position)

	var positionMatrix [][][]float64
	for i := 0; i < NUnit; i++ {
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
		if choose == -1 {
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
		if prob1 == 0 || prob1 == 1 {
			case1 = true
		}
		if prob2 == 0 || prob2 == 1 {
			case2 = true
		}

		if case1 && case2 {
			if choose > sampleIndex {
				positionMatrix = clipMatrix(positionMatrix, choose)
				positionMatrix = clipMatrix(positionMatrix, sampleIndex)
			} else {
				positionMatrix = clipMatrix(positionMatrix, sampleIndex)
				positionMatrix = clipMatrix(positionMatrix, choose)
			}
		} else if case1 {
			positionMatrix = clipMatrix(positionMatrix, sampleIndex)
		} else if case2 {
			positionMatrix = clipMatrix(positionMatrix, choose)
		}
	}

	var sample []int
	for i := 0; i < len(incluProb); i++ {
		if incluProb[i] == 1 {
			sample = append(sample, i)
		}
	}

	return sample
}
