package bohao

import (
	//"fmt"

	"github.com/sajari/regression"
	//matrix "github.com/skelterjohn/go.matrix"
)

/* func LinearFit(A [][]float64, B []float64) []float64 {
	mat_A := matrix.MakeDenseMatrixStacked(A)
	mat_AT := matrix.MakeDenseCopy(mat_A).Transpose()

	mat_B := matrix.MakeDenseMatrix(B, len(B), 1)

	mat_ATA := matrix.ParallelProduct(mat_AT, mat_A)
	mat_ATAinv := matrix.Inverse(mat_ATA)
	mat_ATAinvAT := matrix.ParallelProduct(mat_ATAinv, mat_AT)
	mat_x := matrix.ParallelProduct(mat_ATAinvAT, mat_B)

	return mat_x.Array()
}
*/

type RegressionResult struct {
	r                 *regression.Regression
	Coeff             []float64
	R2                float64
	R2_adj            float64
	Formula           string
	VariancePredicted float64
	VarianceObserved  float64
}

func Simple_Regression(x, y []float64) *RegressionResult {
	// Check!
	if len(x) != len(y) {
		panic("The number of varibales and responses are not match!")
	}

	r := new(regression.Regression)
	for i := 0; i < len(x); i++ {
		r.Train(regression.DataPoint(y[i], []float64{x[i]}))
	}
	r.Run()

	R := new(RegressionResult)
	R.r = r
	R.Formula = r.Formula
	R.R2 = r.R2
	R.R2_adj = 1. - ((1. - r.R2) * float64(len(x)-1) / float64(len(x)-1.-1.))
	R.Coeff = r.GetCoeffs()
	R.VariancePredicted = r.VariancePredicted
	R.VarianceObserved = r.Varianceobserved

	return R
}

func (R RegressionResult) Simple_Predict(x []float64) (y []float64) {
	for i := 0; i < len(x); i++ {
		pre, err := R.r.Predict([]float64{x[i]})
		if err != nil {
			panic(err)
		}
		y = append(y, pre)
	}
	return
}

func Multi_Regression(x [][]float64, y []float64) *RegressionResult {
	// Check!
	if len(x) != len(y) {
		panic("The number of varibales and responses are not match!")
	}

	r := new(regression.Regression)
	for i := 0; i < len(x); i++ {
		r.Train(regression.DataPoint(y[i], x[i]))
	}
	r.Run()

	R := new(RegressionResult)
	R.r = r
	R.Formula = r.Formula
	R.R2 = r.R2
	R.R2_adj = 1. - ((1. - r.R2) * float64(len(x)-1) / float64(len(x)-len(x[0])-1))
	R.Coeff = r.GetCoeffs()
	R.VariancePredicted = r.VariancePredicted
	R.VarianceObserved = r.Varianceobserved

	return R
}

func (R RegressionResult) Predict(x [][]float64) (y []float64) {
	for i := 0; i < len(x); i++ {
		pre, err := R.r.Predict(x[i])
		if err != nil {
			panic(err)
		}
		y = append(y, pre)
	}
	return
}
