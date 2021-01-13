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

// Data should be reshape by tensor module.
func ToOneDimSlice(sli [][]float64) (result []float64) {
	for i := 0; i < len(sli); i++ {
		result = append(result, sli[i]...)
	}
	return
}

// Define activation type
type ActivationFunc func(a *gorgonia.Node) (*gorgonia.Node, error)

// Do nothing activation function.
func Linear(a *gorgonia.Node) (*gorgonia.Node, error) {
	return a, nil
}

func Softmax(a *gorgonia.Node) (*gorgonia.Node, error) {
	r, err := gorgonia.SoftMax(a, 1)
	if err != nil {
		log.Fatal(err)
	}

	return r, nil
}

// Define loss function type
type LossFunc func(Pred, y *gorgonia.Node) *gorgonia.Node

// Simple RMS error loss function
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

func RatioLoss(Pred, y *gorgonia.Node) *gorgonia.Node {
	one := gorgonia.NewScalar(Pred.Graph(), gorgonia.Float64, gorgonia.WithName("1"), gorgonia.WithValue(float64(1.0)))

	ratio := gorgonia.Must(gorgonia.Div(Pred, y))
	minus := gorgonia.Must(gorgonia.Sub(one, ratio))
	loss := gorgonia.Must(gorgonia.Square(minus))
	cost := gorgonia.Must(gorgonia.Mean(loss))
	
	return cost
}

func CrossEntropy(Pred, y *gorgonia.Node) *gorgonia.Node {

	losses := gorgonia.Must(gorgonia.HadamardProd(gorgonia.Must(gorgonia.Log(Pred)), y))
	cost := gorgonia.Must(gorgonia.Mean(losses))
	cost = gorgonia.Must(gorgonia.Neg(cost))

	return cost
}

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

func HuberLoss(Pred, y *gorgonia.Node) *gorgonia.Node {
	var loss, LtNode, GteNode *gorgonia.Node
	delta := gorgonia.NewScalar(Pred.Graph(), gorgonia.Float64, gorgonia.WithName("delta"), gorgonia.WithValue(float64(1.5)))
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
	cost := gorgonia.Must(gorgonia.Mean(loss))

	return cost
}

func Cal_Batch(samplesize, batchsize int) (BatchSize, batches int) {
	BatchSize = batchsize
	if BatchSize > samplesize {
		BatchSize = samplesize
	} else if BatchSize <= 1 {
		BatchSize = 2
	}
	batches = int(math.Ceil(float64(samplesize) / float64(BatchSize)))
	return
}

func Value2Float(PredVal gorgonia.Value, outputshape int) (result [][]float64) {
	oneDimSlice := PredVal.Data().([]float64)
	outputShape := outputshape
	var tmp []float64

	for i := 0; i < len(oneDimSlice); i++ {
		tmp = append(tmp, oneDimSlice[i])
		if len(tmp) == outputShape {
			result = append(result, tmp)
			tmp = []float64{}
		}
	}
	return
}
