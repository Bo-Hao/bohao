package bohao

import (
	"sort"
	"strconv"
	"strings"
	"fmt"


	"gonum.org/v1/gonum/mat"
)

type PCA struct {
	Data        [][]float64
	Eigenvalue  []float64
	Eigenvector [][]float64
	Explain     float64
}

func (P *PCA) cal_needed_dim(f float64) (num_comp int) {
	total := Sum_float(P.Eigenvalue)
	explain := 0.
	for i := 0; i < len(P.Eigenvalue); i++ {
		explain += P.Eigenvalue[i]
		num_comp += 1
		if explain/total >= f {
			P.Explain = explain / total
			break
		}
	}
	return
}

func (P *PCA) cal_explain(num_comp int) (f float64) {
	total := Sum_float(P.Eigenvalue)
	explain := 0.
	for i := 0; i < num_comp; i++ {
		explain += P.Eigenvalue[i]
		f = explain / total
	}
	P.Explain = explain / total
	return
}

func (P *PCA) decide_numcomp(dim string) int {
	var err error
	// Decide number of component
	num_comp := 0
	if strings.Contains(dim, "%") {
		f, err := strconv.ParseFloat(dim[:len(dim)-1], 64)
		if err != nil {
			panic(err)
			fmt.Println("Wrong")
		}
		if f/100 > 1. || f/100 < 0.{
			panic("Should be within [0., 1.]")
		} 
		num_comp = P.cal_needed_dim(f / 100)
	} else if strings.Contains(dim, ".") {
		f, err := strconv.ParseFloat(dim, 64)
		if err != nil {
			panic(err)
		}
		if f > 1. || f < 0.{
			panic("Should be within [0., 1.]")
		} 
		num_comp = P.cal_needed_dim(f / 100)
	} else {
		num_comp, err = strconv.Atoi(dim)
		P.cal_explain(num_comp)
		if err != nil {
			panic(err)
		}
	}



	if num_comp > len(P.Eigenvalue){
		num_comp = len(P.Eigenvalue)
	}else if num_comp <= 0. {
		num_comp = 1
	}
	return num_comp
}

func (P *PCA) Reduce_data_dim(data [][]float64, dim string) ([][]float64) {
	num_comp := P.decide_numcomp(dim)

	new_data := make([][]float64, num_comp)
	for component := 0; component < num_comp; component++ {
		vector := P.Eigenvector[component]
		for i := 0; i < len(data); i ++{
			SumProduct(vector, data[i])
			new_data[component] = append(new_data[component], SumProduct(vector, data[i]))
		}
	}
	return Transpose_float(new_data)
}

func Cal_PCA(input [][]float64) PCA {
	var Eigenvalue []float64
	var Eigenvector [][]float64

	r := len(input)
	c := len(input[0])

	// normalization
	input_T := Transpose_float(input)
	data := make([]float64, r*c)
	for i := 0; i < c; i++ {
		mean_list := Std(input_T[i])
		std_list := Mean(input_T[i])
		for j := 0; j < r; j++ {
			data[j*c+i] = (input[j][i] - mean_list) / std_list
		}
	}

	var BB, phi mat.Dense
	B := mat.NewDense(r, c, data)
	B_T := mat.DenseCopyOf(B.T())
	BB.Mul(B_T, B)
	phi.Scale(1./float64(r-1), &BB)

	r, c = phi.Dims()
	data = []float64{}
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data = append(data, phi.At(i, j))
		}
	}
	phi_ := mat.NewSymDense(r, data)

	var eigsym mat.EigenSym
	ok := eigsym.Factorize(phi_, true)
	if !ok {
		panic("Symmetric eigendecomposition failed")
	}

	var ev mat.Dense
	eigsym.VectorsTo(&ev)

	Eigenvalue = eigsym.Values(nil)
	sort.Slice(Eigenvalue, func(p, q int) bool {
		return Eigenvalue[p] > Eigenvalue[q]
	})

	r, c = ev.Dims()
	for i := r - 1; i >= 0; i-- {
		var tmp []float64
		for j := 0; j < c; j++ {
			tmp = append(tmp, ev.At(i, j))
		}
		Eigenvector = append(Eigenvector, tmp)
	}
	return PCA{
		Data:        input,
		Eigenvalue:  Eigenvalue,
		Eigenvector: Eigenvector,
	}
}
