package bohao

import (
	//"fmt"

	matrix "github.com/skelterjohn/go.matrix"
)

func LinearFit(A [][]float64, B []float64) []float64 {
	mat_A := matrix.MakeDenseMatrixStacked(A)
	mat_AT := matrix.MakeDenseCopy(mat_A).Transpose()

	mat_B := matrix.MakeDenseMatrix(B, len(B), 1)

	mat_ATA := matrix.ParallelProduct(mat_AT, mat_A)
	mat_ATAinv := matrix.Inverse(mat_ATA)
	mat_ATAinvAT := matrix.ParallelProduct(mat_ATAinv, mat_AT)
	mat_x := matrix.ParallelProduct(mat_ATAinvAT, mat_B)

	return mat_x.Array()
}
