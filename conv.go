package bohao

import (
	"log"
	"strconv"
)

func StrToFloat(s string) float64 {
	var result float64
	var err error
	if result, err = strconv.ParseFloat(s, 64); err != nil {
		log.Fatal(err)
	}
	return result
}

func IntToStr_sli(sli []int) (result []string){
	for i := 0; i < len(sli); i++{
		result = append(result , strconv.Itoa(sli[i]))
	}
	return
}

func StrToInt_sli(sli []string) (result []int){
	for i := 0; i < len(sli); i++{
		v, _ := strconv.Atoi(sli[i])
		result = append(result , v)
	}
	return
}

func StrToFloat_sli(sli []string) (result []float64) {
	for i := 0; i < len(sli); i ++{
		f, err := strconv.ParseFloat(sli[i], 64)
		if err != nil {
			log.Fatal(err)
		}
		result = append(result, f)
	}
	return 
}

func ConvSliceFromStr2Float(s [][]string) [][]float64 {
	var result [][]float64
	var tmp []float64
	for i := 0; i < len(s); i++ {
		tmp = []float64{}
		for j := 0; j < len(s[i]); j++ {
			tmp = append(tmp, StrToFloat(s[i][j]))
		}
		result = append(result, tmp)
	}
	return result
}

func ConvSliceFromStr2int(s [][]string) [][]int {
	var result [][]int
	var tmp []int
	for i := 0; i < len(s); i++ {
		tmp = []int{}
		for j := 0; j < len(s[i]); j++ {
			elt, _ := strconv.Atoi(s[i][j])
			tmp = append(tmp, elt)
		}
		result = append(result, tmp)
	}
	return result
}

func ConvSliceFromFloatToStr(s [][]float64) (result [][]string){
	for i := 0; i < len(s); i ++{
		var tmp []string
		for j := 0; j < len(s[i]); j ++{
			v := strconv.FormatFloat(s[i][j], 'f', 10, 64)
			tmp = append(tmp, v)
		}
		result = append(result, tmp)
	}
	return
}

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
