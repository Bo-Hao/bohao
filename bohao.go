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

// Version : simply hold fmt package active. 
func Version() {
	fmt.Println("version- 1.1.0")
}

// AddOne : add one to every elt. 
func AddOne(data [][]float64) (withOne [][]float64) {
	for i := 0; i < len(data); i++ {
		withOne = append(withOne, append(data[i], 1.0))
	}
	return
}

//IsInInt : check if elt in the sli. 
func IsInInt(num int, sli []int) bool {
	respond := false
	for i := 0; i < len(sli); i++ {
		if num == sli[i] {
			respond = true
		}
	}
	return respond
}

// IsInFloat : check if elt in the sli.
func IsInFloat(num float64, sli []float64) bool {
	respond := false
	for i := 0; i < len(sli); i++ {
		if num == sli[i] {
			respond = true
		}
	}
	return respond
}

// IsInStr : check if elt in the sli.
func IsInStr(num string, sli []string) bool {
	respond := false
	for i := 0; i < len(sli); i++ {
		if num == sli[i] {
			respond = true
		}
	}
	return respond
}

// TransposeInt : transpose for int form. 
func TransposeInt(mat [][]int) [][]int {
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

// TransposeFloat : transpose for float64 form. 
func TransposeFloat(mat [][]float64) [][]float64 {
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

// TransposeStr : transpose for string form.
func TransposeStr(mat [][]string) [][]string {
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

// SumFloat : perfomr sum.
func SumFloat(sli []float64) (result float64) {

	for i := 0; i < len(sli); i++ {
		result += sli[i]
	}
	return
}

// SumInt : perfomr sum.
func SumInt(sli []int) (result int) {

	for i := 0; i < len(sli); i++ {
		result += sli[i]
	}
	return
}

// RemoveDuplicateElement : remove the replications. 
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

// RemoveDuplicateElementFloat : remove the replications. 
func RemoveDuplicateElementFloat(addrs []float64) []float64 {
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

// RemoveDuplicateElementInt : remove the replications. 
func RemoveDuplicateElementInt(addrs []int) []int {
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

// EncodeYLabel : Encode Y label. 
func EncodeYLabel(rawY []string) ([]int, map[string]int) {
	label := RemoveDuplicateElement(rawY)
	sort.Strings(label)

	labelMap := make(map[string]int)
	for i := 0; i < len(label); i++ {
		labelMap[label[i]] = i
	}
	newY := make([]int, len(rawY))
	for i := 0; i < len(rawY); i++ {
		newY[i] = labelMap[rawY[i]]
	}
	return newY, labelMap
}

// ReverseMapStrToInt : reverse the map.
func ReverseMapStrToInt(m map[string]int) (reversedMap map[int]string) {
	for key, value := range m {
		reversedMap[value] = key
	}
	return
}

// SumProduct : perform sum product. 
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

// MaxSliceInt : find maximum.
func MaxSliceInt(v []int) (m int) {
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

// MinSliceInt : find minimum.
func MinSliceInt(v []int) (m int) {
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

// MaxSliceFloat : find maximum.
func MaxSliceFloat(v []float64) (m float64) {
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

// MinSliceFloat : find minimum.
func MinSliceFloat(v []float64) (m float64) {
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

// HistogramDataInt : rerange the data.
func HistogramDataInt(sli []int) [][]int {
	maximum := MaxSliceInt(sli)
	minimum := MinSliceInt(sli)
	n := maximum - minimum + 1
	count := make([]int, n)
	for i := 0; i < len(sli); i++ {
		count[sli[i]-minimum] ++
	}
	result := make([][]int, n)
	for i := 0; i < n; i++ {
		result[i] = []int{i + minimum, count[i]}
	}
	return result
}

// Norm : perform the 2 norm. 
func Norm(x []float64, y []float64) (norm float64) {
	length := int(math.Min(float64(len(x)), float64(len(y))))
	for i := 0; i < length; i++ {
		norm += math.Pow(x[i]-y[i], 2)
	}
	norm = math.Sqrt(norm)
	return
}

// GlobalMoransI : compute the Moran's I. 
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

// C : perform combination.
func C(x, y int) int {
	c := 1
	for i := 0; i < y; i++ {
		c *= x
		x--
	}
	for i := 0; i < y; i++ {
		c /= i + 1
	}
	return c
}

// SquareSum : perform square sum. 
func SquareSum(sli []float64) (result float64) {
	for i := 0; i < len(sli); i++ {
		result += math.Pow(sli[i], 2)
	}
	return
}

// Variance : calculate variance. 
func Variance(sli []float64) (result float64) {
	square := 0.
	sum := 0.
	n := len(sli)
	for i := 0; i < n; i++ {
		sum += float64(sli[i])
		square += math.Pow(float64(sli[i]), 2)
	}
	result = math.Abs(square-math.Pow(sum, 2)/float64(n)) / float64(n-1)
	return
}

// Std : compute standard deviation. 
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

// StdInt : compute standard deviation for int slice. 
func StdInt(sli []int) (result float64) {
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

// Mean : mean it. 
func Mean(sli []float64) (mean float64) {
	mean = SumFloat(sli) / float64(len(sli))
	return
}


// Normalized : normalize by mu and sd. 
func Normalized(rawData [][]float64) ([][]float64, []float64, []float64) {
	rawDataT := TransposeFloat(rawData)
	normData := make([][]float64, len(rawData))
	for i := 0; i < len(rawData); i++ {
		for j := 0; j < len(rawData[i]); j++ {
			normData[i] = append(normData[i], 0.0)
		}
	}

	meanList := make([]float64, 0, len(rawDataT) + 5)
	stdList := make([]float64, 0, len(rawDataT) + 5)
	for i := 0; i < len(rawDataT); i++ {
		meanList = append(meanList, Mean(rawDataT[i]))
		stdList = append(stdList, Std(rawDataT[i]))
	}

	for i := 0; i < len(rawData); i++ {
		for j := 0; j < len(rawData[i]); j++ {
			if math.Abs(stdList[j]) <= 0.00000001 {
				normData[i][j] = (rawData[i][j] - meanList[j])
			} else {
				normData[i][j] = (rawData[i][j] - meanList[j]) / (stdList[j])
			}
		}
	}
	return normData, meanList, stdList
}

// NormalizeAdjust : normalize by computed mu and sd. 
func NormalizeAdjust(rawData [][]float64, meanList, stdList []float64) [][]float64 {
	normData := make([][]float64, len(rawData))
	for i := 0; i < len(rawData); i++ {
		for j := 0; j < len(rawData[i]); j++ {
			normData[i] = append(normData[i], 0.0)
		}
	}
	
	for i := 0; i < len(rawData); i++ {
		for j := 0; j < len(rawData[i]); j++ {
			if math.Abs(stdList[j]) <= 0.00000001 {
				normData[i][j] = (rawData[i][j] - meanList[j])
			} else {
				normData[i][j] = (rawData[i][j] - meanList[j]) / (stdList[j])
			}
		}
	}
	return normData
}

// Generalize : generalize it by input mu and sd. 
func Generalize(normData [][]float64, meanList, stdList []float64) [][]float64 {
	rawData := make([][]float64, len(normData))
	for i := 0; i < len(normData); i++ {
		for j := 0; j < len(normData[i]); j++ {
			rawData[i] = append(rawData[i], 0.0)
		}
	}
	for i := 0; i < len(normData); i++ {
		for j := 0; j < len(normData[i]); j++ {
			rawData[i][j] = normData[i][j]*stdList[j] + meanList[j]
		}
	}

	return rawData
}

// WhereFloat : find where is input elt sub. 
func WhereFloat(sli []float64, sub float64) (idx []int) {
	for i := 0; i < len(sli); i++ {
		if sli[i] == sub {
			idx = append(idx, i)
		}
	}
	return
}

// WhereInt : find where is input elt sub. 
func WhereInt(sli []int, sub int) (idx []int) {
	for i := 0; i < len(sli); i++ {
		if sli[i] == sub {
			idx = append(idx, i)
		}
	}
	return
}

// IntersectionSliceInt : derive the intersection. 
func IntersectionSliceInt(sli1 []int, sli2 []int) (idx [][]int) {
	for i := 0; i < len(sli1); i++ {
		intersect := WhereInt(sli2, sli1[i])
		if len(intersect) != 0 {
			for j := 0; j < len(intersect); j++ {
				idx = append(idx, []int{sli1[i], sli2[intersect[j]]})
			}
		}
	}
	return
}

// Quantiles : calculate quantiles. 
func Quantiles(list []float64) (qList []float64) {
	qList = make([]float64, 0, 10)
	length := len(list) - 1
	sort.Float64s(list)
	for q := 0.; q <= 4.; q += 1. {
		indx := float64(length) / 4. * float64(q)

		if indx == math.Ceil(indx) {
			qList = append(qList, list[int(indx)])
		} else {
			qList = append(qList, (list[int(indx)-1]+list[int(indx)])/2)
		}
	}
	return
}

// Percentile : compute percentile.
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

// Factorial : perform factorial
func Factorial(n int) int{
	res := 1
	if n == 0{
		return 1
	}else {
		for i := 1; i <= n; i ++{
			res *= i 
		}
		return res 
	}
	
}