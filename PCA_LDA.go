package bohao 

import(
	"sort"
	"gonum.org/v1/gonum/mat"
)

func PCA(input [][]float64) (eigenvalue []float64, eigenvector [][]float64){
	r := len(input)
	c := len(input[0])

	// normalization
	input_T := Transpose_float(input)
	data := make([]float64, r*c)
	for i := 0; i < c; i ++{
		mean_list := Std(input_T[i])
		std_list := Mean(input_T[i])
		for j := 0; j < r; j ++{
			data[j * c + i] =  (input[j][i] - mean_list)/std_list
		}
	}

	var BB, phi mat.Dense
	B := mat.NewDense(r, c, data)
	B_T := mat.DenseCopyOf(B.T())	
	BB.Mul(B_T, B)
	phi.Scale(1./float64(r - 1), &BB)

	r, c = phi.Dims()
	data = []float64{}
	for i := 0; i < r; i++ {
		for j := 0; j < c; j ++{
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

	eigenvalue = eigsym.Values(nil)
	sort.Slice(eigenvalue, func(p, q int) bool {
		return eigenvalue[p] > eigenvalue[q]
	})

	
	r, c = ev.Dims()
	for i := r - 1; i >= 0; i -- {
		var tmp []float64
		for j := 0; j < c; j ++{
			tmp = append(tmp, ev.At(i, j))
		}
		eigenvector = append(eigenvector, tmp)
	}
	return 
}