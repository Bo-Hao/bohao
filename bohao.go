package bohao

import (
	"fmt"
	"math"
	"sort"
)

var RangeColor = []string{
	"#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8",
	"#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026",
}

func hold() {
	fmt.Println("version- 1.1.0")
}

func AddOne(data [][]float64) (withOne [][]float64) {
	for i := 0; i < len(data); i++ {
		withOne = append(withOne, append(data[i], 1.0))
	}
	return
}

func IsIn_int(num int, sli []int) bool {
	respond := false
	for i := 0; i < len(sli); i++ {
		if num == sli[i] {
			respond = true
		}
	}
	return respond
}

func IsIn_float(num float64, sli []float64) bool {
	respond := false
	for i := 0; i < len(sli); i++ {
		if num == sli[i] {
			respond = true
		}
	}
	return respond
}

func IsIn_str(num string, sli []string) bool {
	respond := false
	for i := 0; i < len(sli); i++ {
		if num == sli[i] {
			respond = true
		}
	}
	return respond
}

func Transpose_int(mat [][]int) [][]int {
	row := len(mat)
	col := len(mat[0])

	t := make([][]int, col)
	for j := 0; j < col; j++ {
		for i := 0; i < row; i++ {
			t[j] = append(t[j], mat[i][j])
		}
	}
	return t
}

func Transpose_float(mat [][]float64) [][]float64 {

	row := len(mat)
	col := len(mat[0])

	t := make([][]float64, col)
	for j := 0; j < col; j++ {
		for i := 0; i < row; i++ {
			t[j] = append(t[j], mat[i][j])
		}
	}
	return t

}

func Transpose_str(mat [][]string) [][]string {
	row := len(mat)
	col := len(mat[0])

	t := make([][]string, col)
	for j := 0; j < col; j++ {
		for i := 0; i < row; i++ {
			t[j] = append(t[j], mat[i][j])
		}
	}
	return t
}

func Sum_float(sli []float64) (result float64) {

	for i := 0; i < len(sli); i++ {
		result += sli[i]
	}
	return
}

func Sum_int(sli []int) (result int) {

	for i := 0; i < len(sli); i++ {
		result += sli[i]
	}
	return
}

func RemoveDuplicateElement(addrs []string) []string {
	result := make([]string, 0, len(addrs))
	temp := map[string]struct{}{}
	for _, item := range addrs {
		if _, ok := temp[item]; !ok {
			temp[item] = struct{}{}
			result = append(result, item)
		}
	}
	return result
}

func RemoveDuplicateElement_float64(addrs []float64) []float64 {
	result := make([]float64, 0, len(addrs))
	temp := map[float64]struct{}{}
	for _, item := range addrs {
		if _, ok := temp[item]; !ok {
			temp[item] = struct{}{}
			result = append(result, item)
		}
	}
	return result
}

func RemoveDuplicateElement_int(addrs []int) []int {
	result := make([]int, 0, len(addrs))
	temp := map[int]struct{}{}
	for _, item := range addrs {
		if _, ok := temp[item]; !ok {
			temp[item] = struct{}{}
			result = append(result, item)
		}
	}
	return result
}

func EncodeYLabel(raw_y []string) ([]int, map[string]int) {
	label := RemoveDuplicateElement(raw_y)
	sort.Strings(label)

	label_map := make(map[string]int)
	for i := 0; i < len(label); i++ {
		label_map[label[i]] = i
	}
	new_y := make([]int, len(raw_y))
	for i := 0; i < len(raw_y); i++ {
		new_y[i] = label_map[raw_y[i]]
	}
	return new_y, label_map
}

func ReverseMap_strToint(m map[string]int) (reversed_map map[int]string) {
	for key, value := range m {
		reversed_map[value] = key
	}
	return
}

func SumProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("Wrong length for sum product")
	}

	sum := 0.
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func MaxSlice_int(v []int) (m int) {
	if len(v) > 0 {
		m = v[0]
	}
	for i := 1; i < len(v); i++ {
		if v[i] > m {
			m = v[i]
		}
	}
	return
}

func MinSlice_int(v []int) (m int) {
	if len(v) > 0 {
		m = v[0]
	}
	for i := 1; i < len(v); i++ {
		if v[i] < m {
			m = v[i]
		}
	}
	return
}

func MaxSlice_float64(v []float64) (m float64) {
	if len(v) > 0 {
		m = v[0]
	}
	for i := 1; i < len(v); i++ {
		if v[i] > m {
			m = v[i]
		}
	}
	return
}

func MinSlice_float64(v []float64) (m float64) {
	if len(v) > 0 {
		m = v[0]
	}
	for i := 1; i < len(v); i++ {
		if v[i] < m {
			m = v[i]
		}
	}
	return
}

func HistogramData_int(sli []int) [][]int {
	maximum := MaxSlice_int(sli)
	minimum := MinSlice_int(sli)
	n := maximum - minimum + 1
	count := make([]int, n)
	for i := 0; i < len(sli); i++ {
		count[sli[i]-minimum] += 1
	}
	result := make([][]int, n)
	for i := 0; i < n; i++ {
		result[i] = []int{i + minimum, count[i]}
	}
	return result
}

func Norm(x []float64, y []float64) (norm float64) {
	length := int(math.Min(float64(len(x)), float64(len(y))))
	for i := 0; i < length; i++ {
		norm += math.Pow(x[i]-y[i], 2)
	}
	norm = math.Sqrt(norm)
	return
}

func GlobalMoransI(data [][]float64) float64 {
	var weightMatrix [][]float64 // record weight
	var xbar float64 = 0.0
	var N float64 = float64(len(data))
	var coordLength int = len(data[0]) - 1

	for i := 0; i < len(data); i++ {
		xbar += data[i][2]
		tmp := []float64{}
		for j := 0; j < len(data); j++ {
			norm := Norm(data[i][:coordLength-1], data[j][:coordLength-1])
			if norm == 0 {
				tmp = append(tmp, 0)
			} else {
				tmp = append(tmp, 1.0/norm)
			}
		}
		weightMatrix = append(weightMatrix, tmp)
	}
	xbar = xbar / N

	var upperPart float64
	var weightSum, squareError float64
	for i := 0; i < len(data); i++ {
		squareError += math.Pow(data[i][coordLength]-xbar, 2)
		for j := 0; j < len(data); j++ {
			upperPart += weightMatrix[i][j] * (data[i][coordLength] - xbar) * (data[j][coordLength] - xbar)
			weightSum += weightMatrix[i][j]
		}
	}
	//fmt.Println(upperPart, weightSum, squareError)
	I := N * upperPart / weightSum / squareError
	return I
}

func C(x, y int) int {
	c := 1
	for i := 0; i < y; i++ {
		c *= x
		x -= 1
	}
	for i := 0; i < y; i++ {
		c /= i + 1
	}
	return c
}

func Std(sli []float64) (result float64) {
	square := 0.
	sum := 0.
	n := len(sli)
	for i := 0; i < n; i++ {
		sum += float64(sli[i])
		square += math.Pow(float64(sli[i]), 2)
	}
	result = math.Sqrt(math.Abs(square-math.Pow(sum, 2)/float64(n)) / float64(n-1))
	return
}
func Std_int(sli []int) (result float64) {
	square := 0.
	sum := 0.
	n := len(sli)
	for i := 0; i < n; i++ {
		sum += float64(sli[i])
		square += math.Pow(float64(sli[i]), 2)
	}

	result = math.Sqrt(math.Abs(square-math.Pow(sum, 2)/float64(n)) / float64(n-1))
	return
}

func Mean(sli []float64) (mean float64) {
	mean = Sum_float(sli) / float64(len(sli))
	return
}

func Normalized(rawData [][]float64, NormalSize float64) ([][]float64, []float64, []float64) {
	/* rawData_T := Transpose_float(rawData)
	if NormalSize == 0. {
		NormalSize = 1.
	}
	normData := make([][]float64, len(rawData))
	for i := 0; i < len(rawData); i++ {
		for j := 0; j < len(rawData[i]); j++ {
			normData[i] = append(normData[i], 0.0)
		}
	}
	var max_list, min_list []float64
	for i := 0; i < len(rawData_T); i++ {
		max_list = append(max_list, MaxSlice_float64(rawData_T[i]))
		min_list = append(min_list, MinSlice_float64(rawData_T[i]))
	}

	for i := 0; i < len(rawData); i++ {
		for j := 0; j < len(rawData[i]); j++ {
			if max_list[j]-min_list[j] == 0.0 {
				normData[i][j] = 0
			} else {
				normData[i][j] = 2*NormalSize*(rawData[i][j]-min_list[j])/(max_list[j]-min_list[j]) - NormalSize
			}
		}
	} */
	if NormalSize == 0. {
		NormalSize = 1.
	}

	rawData_T := Transpose_float(rawData)
	normData := make([][]float64, len(rawData))
	for i := 0; i < len(rawData); i++ {
		for j := 0; j < len(rawData[i]); j++ {
			normData[i] = append(normData[i], 0.0)
		}
	}
	var max_list, min_list []float64
	for i := 0; i < len(rawData_T); i++ {
		max_list = append(max_list, Mean(rawData_T[i]))
		min_list = append(min_list, Std(rawData_T[i]))
	}

	for i := 0; i < len(rawData); i++ {
		for j := 0; j < len(rawData[i]); j++ {
			if math.Abs(min_list[j]) <= 0.00000001 {
				normData[i][j] = (rawData[i][j] - max_list[j])
			} else {
				normData[i][j] = (rawData[i][j] - max_list[j]) / (min_list[j])
			}
		}
	}
	return normData, max_list, min_list
}

func Normalize_adjust(rawData [][]float64, max_list, min_list []float64) [][]float64 {
	normData := make([][]float64, len(rawData))
	for i := 0; i < len(rawData); i++ {
		for j := 0; j < len(rawData[i]); j++ {
			normData[i] = append(normData[i], 0.0)
		}
	}

	for i := 0; i < len(rawData); i++ {
		for j := 0; j < len(rawData[i]); j++ {
			if math.Abs(min_list[j]) <= 0.00000001 {
				normData[i][j] = (rawData[i][j] - max_list[j])
			} else {
				normData[i][j] = (rawData[i][j] - max_list[j]) / (min_list[j])
			}
		}
	}
	return normData
}

func Generalize(normData [][]float64, max_list, min_list []float64, NormalSize float64) [][]float64 {
	/* if NormalSize == 0. {
		NormalSize = 1.
	}
	NormalSize = NormalSize * 2
	rawData := make([][]float64, len(normData))
	for i := 0; i < len(normData); i++ {
		for j := 0; j < len(normData[i]); j++ {
			rawData[i] = append(rawData[i], 0.0)
		}
	}
	for i := 0; i < len(normData); i++ {
		for j := 0; j < len(normData[i]); j++ {
			rawData[i][j] = (normData[i][j]+NormalSize)*(max_list[j]-min_list[j])/NormalSize/2 + min_list[j]
		}
	} */
	if NormalSize == 0. {
		NormalSize = 1.
	}
	rawData := make([][]float64, len(normData))
	for i := 0; i < len(normData); i++ {
		for j := 0; j < len(normData[i]); j++ {
			rawData[i] = append(rawData[i], 0.0)
		}
	}
	for i := 0; i < len(normData); i++ {
		for j := 0; j < len(normData[i]); j++ {
			rawData[i][j] = normData[i][j]*min_list[j] + max_list[j]
		}
	}

	return rawData
}

func Where_float(sli []float64, sub float64) (idx []int) {
	for i := 0; i < len(sli); i++ {
		if sli[i] == sub {
			idx = append(idx, i)
		}
	}
	return
}

func Where_int(sli []int, sub int) (idx []int) {
	for i := 0; i < len(sli); i++ {
		if sli[i] == sub {
			idx = append(idx, i)
		}
	}
	return
}

func IntersectionSlice_int(sli1 []int, sli2 []int) (idx [][]int) {
	for i := 0; i < len(sli1); i++ {
		intersect := Where_int(sli2, sli1[i])
		if len(intersect) != 0 {
			for j := 0; j < len(intersect); j++ {
				idx = append(idx, []int{sli1[i], sli2[intersect[j]]})
			}
		}
	}
	return
}

func Quantiles(list []float64) (q_list []float64) {
	length := len(list) - 1
	sort.Float64s(list)
	for q := 0.; q <= 4.; q += 1. {
		indx := float64(length) / 4. * float64(q)

		if indx == math.Ceil(indx) {
			q_list = append(q_list, list[int(indx)])
		} else {
			q_list = append(q_list, (list[int(indx)-1]+list[int(indx)])/2)
		}
	}
	return
}

func Percentile(Xs []float64, pctile float64) float64 {
	sort.Float64s(Xs)
	
	N := float64(len(Xs))
	//n := pctile * (N + 1) // R6
	n := 1/3.0 + pctile*(N+1/3.0) // R8
	kf, frac := math.Modf(n)
	k := int(kf)
	if k <= 0 {
		return Xs[0]
	} else if k >= len(Xs) {
		return Xs[len(Xs)-1]
	}
	return Xs[k-1] + frac*(Xs[k]-Xs[k-1])
}
