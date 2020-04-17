package bohao 

import(
	//"fmt"
	"log"
	//"strconv"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
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
		for j := 0; j < len(sli[i]); j++ {
			result = append(result, sli[i][j])
		}
	}
	return
}

// Define activation type 
type ActivationFunc func(a *gorgonia.Node) (*gorgonia.Node, error)

// Do nothing activation function.
func Linear(a *gorgonia.Node) (*gorgonia.Node, error){
	return a, nil
}

// A activation release at 2019. Claim that it is outperform ReLU and tanh.
func Mish(a *gorgonia.Node) (*gorgonia.Node, error) {
	exp_a, err := gorgonia.Exp(a)
	if err != nil {
		log.Printf("Can't take the exponential")
	}
	var aback []float64
	for i := 0; i < a.Shape()[0] * a.Shape()[1]; i ++{
		aback = append(aback, 1) 
	}
	constant_one := gorgonia.NewConstant(tensor.New(tensor.WithBacking(aback), tensor.WithShape(a.Shape()[0], a.Shape()[1])))
	ln, err := gorgonia.Log(gorgonia.Must(gorgonia.Add(constant_one, exp_a)))
	//ln, err := gorgonia.Log(exp_a)
	if err != nil {
		log.Printf("Can't take the logarothm")
	} 
	tanh, err := gorgonia.Tanh(ln)
	if err != nil {
		log.Printf("Can't take the tan hyperbolic")
	}
	mish, err := gorgonia.HadamardProd(a, tanh)
	if err != nil {
		log.Printf("Can't take the multiple")
	}
	return mish, nil
} 

// Define loss function type 
type LossFunc func(Pred, y *gorgonia.Node) *gorgonia.Node

// Simple RMS error loss function 
func RMSError(Pred, y *gorgonia.Node) *gorgonia.Node {
	losses := gorgonia.Must(gorgonia.Square(gorgonia.Must(gorgonia.Sub(Pred, y))))
	cost := gorgonia.Must(gorgonia.Mean(losses))
	return cost
}