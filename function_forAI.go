package bohao

import (
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
		for j := 0; j < len(sli[i]); j++ {
			result = append(result, sli[i][j])
		}
	}
	return
}

// Define activation type
type ActivationFunc func(a *gorgonia.Node) (*gorgonia.Node, error)

// Do nothing activation function.
func Linear(a *gorgonia.Node) (*gorgonia.Node, error) {
	return a, nil
}

// Define loss function type
type LossFunc func(Pred, y *gorgonia.Node) *gorgonia.Node

// Simple RMS error loss function
func RMSError(Pred, y *gorgonia.Node) *gorgonia.Node {
	losses := gorgonia.Must(gorgonia.Square(gorgonia.Must(gorgonia.Sub(Pred, y))))
	cost := gorgonia.Must(gorgonia.Mean(losses))
	return cost
}

func RatioLoss(Pred, y *gorgonia.Node)*gorgonia.Node {
	
	losses := gorgonia.Must(gorgonia.Square(gorgonia.Must(gorgonia.Div(Pred, y))))
	l := gorgonia.Must(gorgonia.Log(losses))
	cost := gorgonia.Must(gorgonia.Mean(l))
	return cost
}

// remenber the old day, bach training is applied.
/* // Since different optimizer are not the same type. We should rewrite code.
if para.Solver == "RMSProp" {
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(para.Lr))

	// Start epoches training
	for epoch := 1; epoch < para.Epoches+1; epoch++ {
		// Start batches...
		for b := 0; b < batches; b++ {
			// Handling the
			start := b * batchSize
			end := start + batchSize
			over := 0
			if start >= sampleSize {
				break
			}
			if end > sampleSize {
				over = end - sampleSize
				end = sampleSize
			}

			//slice data Note: xT and xVal are same type but different size. So does yT and yVal.
			xVal, err := xT.Slice(sli{start, end})
			if err != nil {
				log.Fatal(err, "Can't slice the X data")
			}
			yVal, err := yT.Slice(sli{start, end})
			if err != nil {
				log.Fatal(err, "Can't slice the Y data")
			}

			// Define input output node
			X := gorgonia.NewMatrix(m.G, tensor.Float64, gorgonia.WithShape(batchSize-over, inputShape), gorgonia.WithName("X"))
			y := gorgonia.NewMatrix(m.G, tensor.Float64, gorgonia.WithShape(batchSize-over, outputShape), gorgonia.WithName("y"))

			// forward pass
			err = m.Forward(X)
			if err != nil {
				log.Fatal(err, "Forward pass fail")
			}

			// Define the loss function.
			cost := para.Lossfunc(m.Pred, y)

			// Record cost change
			gorgonia.Read(cost, &costVal)

			// Update the gradient.
			if _, err = gorgonia.Grad(cost, m.Learnables()...); err != nil {
				log.Fatal("Unable to udate gradient")
			}

			// Define the tape machine to record the gradient change for the nodes which should be optimized or activated.
			vm := gorgonia.NewTapeMachine(m.G, gorgonia.BindDualValues(m.Learnables()...))


			// Dump it
			gorgonia.Let(X, xVal)
			gorgonia.Let(y, yVal)

			// Optimizing...
			vm.Reset()
			vm.RunAll()
			solver.Step(gorgonia.NodesToValueGrads(m.Learnables()))
			vm.Reset()
		}

		// Print cost
		if epoch%50 == 0 {
			fmt.Println("Iteration: ", epoch, "  Cost: ", costVal)
		}
		// Stock it.
		S.LossRecord = append(S.LossRecord, []float64{float64(epoch), costVal.Data().(float64)})
	}
	m.FitStock.LossRecord = S.LossRecord

} else if para.Solver == "Adam" {
	solver := gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(para.Lr))

	// Start epoches training
	for epoch := 0; epoch < para.Epoches; epoch++ {
		// Start batches...
		for b := 0; b < batches; b++ {
			// Handling the
			start := b * batchSize
			end := start + batchSize
			over := 0
			if start >= sampleSize {
				break
			}
			if end > sampleSize {
				over = end - sampleSize
				end = sampleSize
			}

			//slice data Note: xT and xVal are same type but different size. So does yT and yVal.
			xVal, err := xT.Slice(sli{start, end})
			if err != nil {
				log.Fatal(err, "Can't slice the X data")
			}
			yVal, err := yT.Slice(sli{start, end})
			if err != nil {
				log.Fatal(err, "Can't slice the Y data")
			}

			// Define input output node
			X := gorgonia.NewMatrix(m.G, tensor.Float64, gorgonia.WithShape(batchSize-over, inputShape), gorgonia.WithName("X"))
			y := gorgonia.NewMatrix(m.G, tensor.Float64, gorgonia.WithShape(batchSize-over, outputShape), gorgonia.WithName("y"))

			// forward pass
			err = m.Forward(X)
			if err != nil {
				log.Fatal(err, "Forward pass fail")
			}

			// Define the loss function.
			cost := para.Lossfunc(m.Pred, y)

			// Record cost change
			gorgonia.Read(cost, &costVal)

			// Update the gradient.
			if _, err = gorgonia.Grad(cost, m.Learnables()...); err != nil {
				log.Fatal("Unable to udate gradient")
			}

			// Define the tape machine to record the gradient change for the nodes which should be optimized or activated.
			vm := gorgonia.NewTapeMachine(m.G, gorgonia.BindDualValues(m.Learnables()...))

			// Dump it
			gorgonia.Let(X, xVal)
			gorgonia.Let(y, yVal)

			// Optimizing...
			vm.Reset()
			vm.RunAll()
			solver.Step(gorgonia.NodesToValueGrads(m.Learnables()))
			vm.Reset()
		}

		// Print cost
		if epoch%50 == 0 {
			fmt.Println("Iteration: ", epoch, "  Cost: ", costVal)
		}
		// Stock it.
		S.LossRecord = append(S.LossRecord, []float64{float64(epoch), costVal.Data().(float64)})
	}
	m.FitStock.LossRecord = S.LossRecord
}

// The training process is complete.
log.Printf("training finish!") */
