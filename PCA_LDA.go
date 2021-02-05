package bohao

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// PCA : 
type PCA struct {
	Data        [][]float64
	Eigenvalue  []float64
	Eigenvector [][]float64
	Explain     float64
}

func (P *PCA) calNeededDim(f float64) (numComp int) {
	total := SumFloat(P.Eigenvalue)
	explain := 0.
	for i := 0; i < len(P.Eigenvalue); i++ {
		explain += P.Eigenvalue[i]
		numComp ++
		if explain/total >= f {
			P.Explain = explain / total
			break
		}
	}
	return
}

func (P *PCA) calExplain(numComp int) (f float64) {
	total := SumFloat(P.Eigenvalue)
	explain := 0.
	for i := 0; i < numComp; i++ {
		explain += P.Eigenvalue[i]
		f = explain / total
	}
	P.Explain = explain / total
	return
}

func (P *PCA) decideNumComp(dim string) int {
	var err error
	// Decide number of component
	numComp := 0
	if strings.Contains(dim, "%") {
		f, err := strconv.ParseFloat(dim[:len(dim)-1], 64)
		if err != nil {
			panic(err)
			fmt.Println("Wrong")
		}
		if f/100 > 1. || f/100 < 0. {
			panic("Should be within [0., 1.]")
		}
		numComp = P.calNeededDim(f / 100)
	} else if strings.Contains(dim, ".") {
		f, err := strconv.ParseFloat(dim, 64)
		if err != nil {
			panic(err)
		}
		if f > 1. || f < 0. {
			panic("Should be within [0., 1.]")
		}
		numComp = P.calNeededDim(f / 100)
	} else {
		numComp, err = strconv.Atoi(dim)
		P.calExplain(numComp)
		if err != nil {
			panic(err)
		}
	}

	if numComp > len(P.Eigenvalue) {
		numComp = len(P.Eigenvalue)
	} else if numComp <= 0. {
		numComp = 1
	}
	return numComp
}

// ReduceDataDim : 
func (P *PCA) ReduceDataDim(data [][]float64, dim string) [][]float64 {
	numComp := P.decideNumComp(dim)

	newData := make([][]float64, numComp)
	for component := 0; component < numComp; component++ {
		vector := P.Eigenvector[component]
		for i := 0; i < len(data); i++ {
			SumProduct(vector, data[i])
			newData[component] = append(newData[component], SumProduct(vector, data[i]))
		}
	}
	return TransposeFloat(newData)
}

// CalPCA : 
func CalPCA(input [][]float64) PCA {
	var Eigenvalue []float64
	var Eigenvector [][]float64

	r := len(input)
	c := len(input[0])

	// normalization
	inputT := TransposeFloat(input)
	data := make([]float64, r*c)
	for i := 0; i < c; i++ {
		meanList := Std(inputT[i])
		stdList := Mean(inputT[i])
		for j := 0; j < r; j++ {
			if stdList == 0 {
				data[j*c+i] = (input[j][i] - meanList)
			} else {
				data[j*c+i] = (input[j][i] - meanList) / stdList
			}

		}
	}

	var BB, phi mat.Dense
	B := mat.NewDense(r, c, data)
	BT := mat.DenseCopyOf(B.T())
	BB.Mul(BT, B)
	phi.Scale(1./float64(r-1), &BB)

	r, c = phi.Dims()
	data = []float64{}
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data = append(data, phi.At(i, j))
		}
	}
	phi1 := mat.NewSymDense(r, data)
	var eigsym mat.EigenSym
	ok := eigsym.Factorize(phi1, true)
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
