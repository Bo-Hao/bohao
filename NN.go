package bohao

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"strconv"
	"time"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// A struct that stock the fitting data such as mean and standard error of the data and also save the loss when optimizing.
type Stock struct {
	LossRecord [][]float64
	max_list_x []float64
	min_list_x []float64
	max_list_y []float64
	min_list_y []float64
}

// Use as the input of newNN func. Neuron mean the number of node of each layer. Dropout mean the dropout ratio and Act denote the activation func of each layer.
type NetworkStruction struct {
	Neuron     []int
	Dropout    []float64
	Act        []ActivationFunc
	Bias       bool
	Normal     bool
	NormalSize float64
}

// Stock the fitting parameters.
type Parameter struct {
	Lr        float64
	Epoches   int
	BatchSize int
	Lossfunc  LossFunc
	Solver    string
}

// Initial the default parameter. It is not necessary.
func InitParameter() Parameter {
	return Parameter{
		Lr:        0.01,
		Epoches:   200,
		BatchSize: 500,
		Solver:    "RMSProp",
		Lossfunc:  RMSError,
	}
}

// Neural Network itself.
type NN struct {
	G *gorgonia.ExprGraph
	W []*gorgonia.Node
	B []*gorgonia.Node
	D []float64
	A []ActivationFunc

	Pred    *gorgonia.Node
	PredVal gorgonia.Value

	Normal     bool
	NormalSize float64
	FitStock   Stock
}

// Return the Nodes that should be optimized.
func (m *NN) Learnables() gorgonia.Nodes {
	n := gorgonia.NewNodeSet()
	for i := 0; i < len(m.W); i++ {
		n.Add(m.W[i])
	}
	for i := 0; i < len(m.B); i++ {
		n.Add(m.B[i])
	}

	return n.ToSlice().Nodes()
}

// Create the network.
func NewNN(g *gorgonia.ExprGraph, S NetworkStruction) *NN {
	// Set random seed
	rand.Seed(time.Now().Unix())
	var Ns gorgonia.Nodes
	var Bs gorgonia.Nodes
	for i := 0; i < len(S.Neuron)-1; i++ {
		Ns = append(Ns, gorgonia.NewMatrix(
			g,
			tensor.Float64,
			gorgonia.WithShape(S.Neuron[i], S.Neuron[i+1]),
			gorgonia.WithName("w"+strconv.Itoa(i)),
			gorgonia.WithInit(gorgonia.GlorotU(1)),
		))
	}
	if S.Bias {
		for i := 0; i < len(S.Neuron)-2; i++ {
			Bs = append(Bs, gorgonia.NewMatrix(
				g,
				tensor.Float64,
				gorgonia.WithShape(1, S.Neuron[i+1]),
				gorgonia.WithName("b"+strconv.Itoa(i)),
				gorgonia.WithInit(gorgonia.Uniform(-1, 1)),
			))
		}
	}

	return &NN{
		G:          g,
		W:          Ns,
		B:          Bs,
		D:          S.Dropout,
		A:          S.Act,
		Normal:     S.Normal,
		NormalSize: S.NormalSize,
	}
}

// Forward pass.
func (m *NN) Forward(x *gorgonia.Node) (err error) {
	l := make([]*gorgonia.Node, len(m.W)+1)
	ldot := make([]*gorgonia.Node, len(m.W))
	p := make([]*gorgonia.Node, len(m.W))

	// initial the first layer
	l[0] = x

	for i := 0; i < len(m.W); i++ {
		if len(m.B) != 0 && i < len(m.W)-1 {
			L1, err := gorgonia.Mul(l[i], m.W[i])
			if err != nil {
				panic(err)
			}
			ldot[i], err = gorgonia.BroadcastAdd(L1, m.B[i], nil, []byte{0})
			if err != nil {
				panic(err)
			}
		} else {
			ldot[i], err = gorgonia.Mul(l[i], m.W[i])
			if err != nil {
				log.Fatal("mul wrong ", err)
			}
		}

		drop, err := gorgonia.Dropout(ldot[i], m.D[i])
		if err != nil {
			log.Printf("Can't drop!")
		}
		p[i] = drop

		//activation function
		l[i+1] = gorgonia.Must(m.A[i](p[i]))

	}
	F := gorgonia.Must(m.A[len(m.A)-1](l[len(l)-1]))
	m.Pred = F
	gorgonia.Read(m.Pred, &m.PredVal)
	return
}

func (m NN) ValueToFloatSlice() (result [][]float64) {
	oneDimSlice := m.PredVal.Data().([]float64)
	outputShape := m.W[len(m.W)-1].Shape()[1]

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
func (n *NN) Predict(x [][]float64) (prediction_gen [][]float64) {
	// Define the needed parameters.
	inputShape := n.W[0].Shape()[0]
	sampleSize := len(x)
	batchSize := sampleSize
	if batchSize > sampleSize {
		batchSize = sampleSize
	} else if batchSize <= 1 {
		batchSize = 2
	}
	batches := int(math.Ceil(float64(sampleSize) / float64(batchSize)))

	//----------------------------------------------------------------------Create a New Model----------------------------------------------------------------------------------------------------
	var ww, bb []*gorgonia.Node
	new_G := gorgonia.NewGraph()
	for i := 0; i < len(n.W); i++ {
		xT := tensor.New(tensor.WithBacking(n.W[i].Value().Data()), tensor.WithShape(n.W[i].Shape()[0], n.W[i].Shape()[1]))
		w := gorgonia.NodeFromAny(new_G, xT, gorgonia.WithName("w"+strconv.Itoa(i)), gorgonia.WithShape(n.W[i].Shape()[0], n.W[i].Shape()[1]))

		ww = append(ww, w)
	}
	for i := 0; i < len(n.B); i++ {
		xT := tensor.New(tensor.WithBacking(n.B[i].Value().Data()), tensor.WithShape(n.B[i].Shape()[0], n.B[i].Shape()[1]))
		b := gorgonia.NodeFromAny(new_G, xT, gorgonia.WithName("b"+strconv.Itoa(i)), gorgonia.WithShape(n.B[i].Shape()[0], n.B[i].Shape()[1]))
		bb = append(bb, b)
	}

	var zero_drop []float64
	for i := 0; i < len(n.D); i ++{
		zero_drop = append(zero_drop, 0.)
	}
	m := NN{
		G:          new_G,
		W:          ww,
		B:          bb,
		D:          zero_drop,
		A:          n.A,
		Normal:     n.Normal,
		NormalSize: n.NormalSize,
	}
	//-----------------------------------------------------------------------------Create a New Model---------------------------------------------------------------------------------------------

	//Normalize the input data. And stock the information into m.FitStock.

	var input_x [][]float64
	if m.Normal {
		input_x, _, _ = Normalized(x, m.NormalSize)
	}else {
		input_x = x
	}

	x_oneDim := ToOneDimSlice(input_x)
	for i := 0; i < inputShape; i++ {
		x_oneDim = append(x_oneDim, x_oneDim[len(x_oneDim)-inputShape+i])
	}

	// Construct the input data tensor.
	xT := tensor.New(tensor.WithBacking(x_oneDim), tensor.WithShape(sampleSize+1, inputShape))

	// make prediction in batch.
	var prediction [][]float64

	// Start batches
	for b := 0; b < batches+1; b++ {
		start := b * batchSize
		end := start + batchSize
		over := 0
		if start >= sampleSize {
			break
		}
		if end > sampleSize {
			over = (end - sampleSize)
			end = sampleSize
		}

		if over == 1 {
			end += 1
		}
		//slice data Note: xT and xVal are same type but different size.
		xVal, err := xT.Slice(sli{start, end})
		if err != nil {
			log.Fatal(err, "Can't slice the data")
		}

		// Define input node.
		X := gorgonia.NewMatrix(m.G, tensor.Float64, gorgonia.WithShape(end-start, inputShape), gorgonia.WithName("X"))

		// Construct forward pass and record it using tape machine.
		m.Forward(X)

		// Dump it, still need tape machine to activate the process.
		gorgonia.Let(X, xVal)

		vm := gorgonia.NewTapeMachine(m.G, gorgonia.BindDualValues(m.Learnables()...))
		// Activate the tape machine.
		vm.RunAll()
		vm.Reset()

		// Append the result.
		if over == 1 {
			prediction = append(prediction, m.ValueToFloatSlice()[0])
		} else {
			prediction = append(prediction, m.ValueToFloatSlice()...)
		}
	}

	// generalize the output using the data which stock in m.FitStock.
	if m.Normal {
		prediction_gen = Generalize(prediction, n.FitStock.max_list_y, n.FitStock.min_list_y, m.NormalSize)
	}else{
		prediction_gen = prediction
	}
	return prediction_gen
}

func (m *NN) Fit(x_, y_ [][]float64, para Parameter) {
	//Normalize the input data. And stock the information into m.FitStock.
	S := Stock{}

	input_x := x_
	input_y := y_
	if m.Normal {
		n_x, max_list_x, min_list_x := Normalized(x_, m.NormalSize)
		input_x = n_x
		S.max_list_x = max_list_x
		S.min_list_x = min_list_x
		n_y, max_list_y, min_list_y := Normalized(y_, 2*m.NormalSize)
		input_y = n_y
		S.max_list_y = max_list_y
		S.min_list_y = min_list_y
	}

	// set S into struct m.
	m.FitStock = S

	// reshape data
	x_oneDim := ToOneDimSlice(input_x)
	y_oneDim := ToOneDimSlice(input_y)
	sampleSize := len(input_x)

	if sampleSize != len(input_y) {
		log.Fatal("x and y are not in the same size!")
	}

	// Define shapes
	inputShape := m.W[0].Shape()[0]
	outputShape := m.W[len(m.W)-1].Shape()[1]

	// batch size will not greater than sample size and won't less than 2. Since batch size equal to 1 will crash the model.
	batchSize := para.BatchSize
	if batchSize > sampleSize {
		batchSize = sampleSize
	} else if batchSize <= 1 {
		batchSize = 2
	}
	batches := sampleSize / batchSize

	// Construct the input data tensor and node.
	xT := tensor.New(tensor.WithBacking(x_oneDim), tensor.WithShape(sampleSize, inputShape))
	yT := tensor.New(tensor.WithBacking(y_oneDim), tensor.WithShape(sampleSize, outputShape))

	// costVal will be used outside the loop.
	var costVal gorgonia.Value

	// Since different optimizer are not the same type. We should rewrite code.
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
			if epoch%100 == 0 {
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
			if epoch == 500 {
				solver = gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(para.Lr/10))
			}
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
			if epoch%100 == 0 {
				fmt.Println("Iteration: ", epoch, "  Cost: ", costVal)
			}
			// Stock it.
			S.LossRecord = append(S.LossRecord, []float64{float64(epoch), costVal.Data().(float64)})
		}
		m.FitStock.LossRecord = S.LossRecord
	}

	// The training process is complete.
	log.Printf("training finish!")
}

// Separate the neural network apart. Since it has same graph as the original network, it can be trained separtely.
func (m *NN) Tear_apart(g *gorgonia.ExprGraph, layer_in, layer_out int) NN {
	m_new := NN{}
	// it should use the same graph
	m_new.G = g
	m_new.W = m.W[layer_in:layer_out]
	m_new.D = m.D[layer_in:layer_out]
	m_new.A = m.A[layer_in:layer_out]
	return m_new
}

func Example_Auto() {
	fmt.Println("start!")

	f_x := [][]float64{[]float64{1, 1, 1, 1}, []float64{2, 2, 2, 2}, []float64{3, 3, 3, 3}, []float64{4, 4, 4, 4}, []float64{5, 5, 5, 5}}
	f_m := [][]float64{[]float64{-1}, []float64{-2}, []float64{-3}, []float64{-4}, []float64{-5}}
	fmt.Println(f_m)

	g := gorgonia.NewGraph()

	S := NetworkStruction{
		Neuron:  []int{4, 2, 1, 2, 4},
		Dropout: []float64{0, 0, 0, 0, 0},
		Act:     []ActivationFunc{Mish, Linear, Linear, Linear, Linear},
	}
	para := InitParameter()

	m := NewNN(g, S)

	fmt.Println(m.W[1].Value())
	m.Fit(f_x, f_x, para)

	m1 := m.Tear_apart(g, 0, 2)
	m1.Predict(f_x)
	fmt.Println(m1.PredVal)

}

func Example_NN() {
	// fake data
	x := [][]float64{[]float64{1, 2, 3, 1}, []float64{4, 5, 6, 1}, []float64{4, 5, 6, 1}, []float64{4, 5, 6, 1}, []float64{4, 5, 6, 1}}
	//y := [][]float64{[]float64{1, 2}, []float64{5, 6}, []float64{5, 6}, []float64{5, 6}, []float64{5, 6}}
	y := [][]float64{[]float64{1}, []float64{5}, []float64{5}, []float64{5}, []float64{5}}

	// init graph
	g := gorgonia.NewGraph()

	// setup network struction
	S := NetworkStruction{
		Neuron:  []int{4, 4, 1},                                     // the front one should be input shape and  the last one should be output shape
		Dropout: []float64{0, 0},                                    // set each dropout layer
		Act:     []ActivationFunc{gorgonia.Rectify, Linear, Linear}, // can use act func directly from outside
		Bias:    true,
	}

	// create NN
	m := NewNN(g, S)

	// init training parameter
	para := InitParameter()

	// fit training data
	m.Fit(x, y, para)

	// set test data into NN
	pred := m.Predict(x)

	// show predition
	fmt.Println(pred)
}
