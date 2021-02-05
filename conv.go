package bohao

import (
	"log"
	"strconv"
)

// Str2Float : conv string to float64.
func Str2Float(s string) float64 {
	var result float64
	var err error
	if result, err = strconv.ParseFloat(s, 64); err != nil {
		log.Fatal(err)
	}
	return result
}

// Int2StrSli : convert []int to []string.
func Int2StrSli(sli []int) (result []string) {
	for i := 0; i < len(sli); i++ {
		result = append(result, strconv.Itoa(sli[i]))
	}
	return
}

// Str2IntSli : convert []string to []int.
func Str2IntSli(sli []string) (result []int) {
	for i := 0; i < len(sli); i++ {
		v, err := strconv.Atoi(sli[i])
		if err != nil {
			panic(err)
		}
		result = append(result, v)
	}
	return
}

// Str2FloatSli : convert []string to []float64.
func Str2FloatSli(sli []string) (result []float64) {
	for i := 0; i < len(sli); i++ {
		f, err := strconv.ParseFloat(sli[i], 64)
		if err != nil {
			log.Fatal(err)
		}
		result = append(result, f)
	}
	return
}

// ConvSliceFromStr2Float : convert [][]string to [][]float64.
func ConvSliceFromStr2Float(s [][]string) [][]float64 {
	var result [][]float64
	var tmp []float64
	for i := 0; i < len(s); i++ {
		tmp = []float64{}
		for j := 0; j < len(s[i]); j++ {
			tmp = append(tmp, Str2Float(s[i][j]))
		}
		result = append(result, tmp)
	}
	return result
}

// ConvSliceFromStr2int : convert [][]string to [][]int
func ConvSliceFromStr2int(s [][]string) [][]int {
	var result [][]int
	var tmp []int
	for i := 0; i < len(s); i++ {
		tmp = []int{}
		for j := 0; j < len(s[i]); j++ {
			elt, err := strconv.Atoi(s[i][j])
			if err != nil {
				panic(err)
			}
			tmp = append(tmp, elt)
		}
		result = append(result, tmp)
	}
	return result
}

// ConvSliceFromFloat2Str : convert [][]float64 to [][]string.
func ConvSliceFromFloat2Str(s [][]float64) (result [][]string) {
	for i := 0; i < len(s); i++ {
		var tmp []string
		for j := 0; j < len(s[i]); j++ {
			v := strconv.FormatFloat(s[i][j], 'f', 10, 64)
			tmp = append(tmp, v)
		}
		result = append(result, tmp)
	}
	return
}

// ExtractData : extract data L from the data.
func ExtractData(data [][]float64, L []int) (result [][]float64) {
	size := len(L)
	for i := 0; i < len(data); i++ {
		tmp := []float64{}
		for j := 0; j < size; j++ {
			tmp = append(tmp, data[i][L[j]])
		}
		result = append(result, tmp)
	}
	return
}
