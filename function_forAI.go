package bohao

import (
	"fmt"
	"log"
	"math"

	"gorgonia.org/gorgonia"
)

var err error

// for batch traininig
type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }

// ToOneDimSlice : Data should be reshape by tensor module.
func ToOneDimSlice(sli [][]float64) []float64 {
	result := make([]float64, 0, len(sli)+5)
	for i := 0; i < len(sli); i++ {
		result = append(result, sli[i]...)
	}
	return result
}

// ActivationFunc : Define activation type
type ActivationFunc func(a *gorgonia.Node) (*gorgonia.Node, error)

// Linear : Do nothing activation function.
func Linear(a *gorgonia.Node) (*gorgonia.Node, error) {
	return a, nil
}

// Softmax : perform Softmax node.
func Softmax(a *gorgonia.Node) (*gorgonia.Node, error) {
	r, err := gorgonia.SoftMax(a, 1)
	if err != nil {
		log.Fatal(err)
	}

	return r, nil
}

// LossFunc : Define loss function type
type LossFunc func(Pred, y *gorgonia.Node) *gorgonia.Node

// MSError : Simple RMS error loss function
func MSError(Pred, y *gorgonia.Node) *gorgonia.Node {

	s, err := gorgonia.Sub(Pred, y)
	if err != nil {
		panic(err)
	}
	losses, err := gorgonia.Square(s)
	if err != nil {
		panic(err)
	}
	cost, err := gorgonia.Mean(losses)
	if err != nil {
		panic(err)
	}

	return cost
}

// RatioLoss : maybe a wrong way for loss function.
func RatioLoss(Pred, y *gorgonia.Node) *gorgonia.Node {
	one := gorgonia.NewScalar(Pred.Graph(), gorgonia.Float64, gorgonia.WithName("1"), gorgonia.WithValue(float64(1.0)))

	ratio := gorgonia.Must(gorgonia.Div(Pred, y))
	minus := gorgonia.Must(gorgonia.Sub(one, ratio))
	loss := gorgonia.Must(gorgonia.Square(minus))
	cost := gorgonia.Must(gorgonia.Mean(loss))

	return cost
}

// CrossEntropy : the loss function with respect to the negative Shannon entropy.
func CrossEntropy(Pred, y *gorgonia.Node) *gorgonia.Node {

	losses := gorgonia.Must(gorgonia.HadamardProd(gorgonia.Must(gorgonia.Log(Pred)), y))
	cost := gorgonia.Must(gorgonia.Mean(losses))
	cost = gorgonia.Must(gorgonia.Neg(cost))

	return cost
}

// PseudoHuberLossDelta1 : perform the approximating Huber loss with delta equal to 1.
func PseudoHuberLossDelta1(Pred, y *gorgonia.Node) *gorgonia.Node {

	one := gorgonia.NewScalar(Pred.Graph(), gorgonia.Float64, gorgonia.WithName("1"), gorgonia.WithValue(float64(1.0)))

	loss := gorgonia.Must(gorgonia.Sub(Pred, y))

	loss = gorgonia.Must(gorgonia.Square(loss))

	loss = gorgonia.Must(gorgonia.Add(one, loss))

	loss = gorgonia.Must(gorgonia.Sqrt(loss))

	loss = gorgonia.Must(gorgonia.Sub(loss, one))

	cost := gorgonia.Must(gorgonia.Mean(loss))

	return cost
}

// HuberLoss : A loss function which combine MSE and MAE with joint point at 1.
func HuberLoss(Pred, y *gorgonia.Node) *gorgonia.Node {
	var loss, LtNode, GteNode *gorgonia.Node
	delta := gorgonia.NewScalar(Pred.Graph(), gorgonia.Float64, gorgonia.WithName("delta"), gorgonia.WithValue(float64(1.)))
	pointFive := gorgonia.NewScalar(Pred.Graph(), gorgonia.Float64, gorgonia.WithName("0.5"), gorgonia.WithValue(float64(0.5)))
	zero := gorgonia.NewScalar(Pred.Graph(), gorgonia.Float64, gorgonia.WithName("0."), gorgonia.WithValue(float64(0.)))
	deltaPointFive := gorgonia.Must(gorgonia.Mul(delta, pointFive))

	loss, err := gorgonia.Sub(Pred, y)
	if err != nil {
		fmt.Println("wrong")
	}
	lossABS := gorgonia.Must(gorgonia.Abs(loss))

	GteNode = gorgonia.Must(gorgonia.Lt(lossABS, delta, true))
	LtNode = gorgonia.Must(gorgonia.Eq(GteNode, zero, true))

	lossG := gorgonia.Must(gorgonia.HadamardProd(lossABS, GteNode))
	lossG = gorgonia.Must(gorgonia.Square(lossG))
	lossG = gorgonia.Must(gorgonia.Mul(pointFive, lossG))

	lossL := gorgonia.Must(gorgonia.HadamardProd(lossABS, LtNode))
	maskPoint5 := gorgonia.Must(gorgonia.Mul(deltaPointFive, LtNode))

	lossL = gorgonia.Must(gorgonia.Sub(lossL, maskPoint5))
	lossL = gorgonia.Must(gorgonia.Mul(delta, lossL))

	loss = gorgonia.Must(gorgonia.Add(lossG, lossL))
	//cost := gorgonia.Must(gorgonia.Sum(loss))
	cost := gorgonia.Must(gorgonia.Mean(loss))

	return cost
}

// CalBatch : compute the needed number of batch.
func CalBatch(samplesize, batchsize int) (BatchSize, batches int) {
	BatchSize = batchsize
	if BatchSize > samplesize {
		BatchSize = samplesize
	} else if BatchSize <= 1 {
		BatchSize = 2
	}
	batches = int(math.Ceil(float64(samplesize) / float64(BatchSize)))
	return
}

// Value2Float : transfer the value to float64 in slice.
func Value2Float(PredVal gorgonia.Value, outputshape int) (result [][]float64) {
	oneDimSlice := PredVal.Data().([]float64)
	outputShape := outputshape
	tmp := make([]float64, 0, outputShape+1)
	result = make([][]float64, 0, int(float64(len(oneDimSlice))/float64(outputShape))+1)

	for i := 0; i < len(oneDimSlice); i++ {
		tmp = append(tmp, oneDimSlice[i])
		if len(tmp) == outputShape {
			result = append(result, tmp)
			tmp = make([]float64, 0, outputShape+1)
		}
	}
	return
}
