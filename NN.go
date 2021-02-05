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

// Stock :a struct that stock the fitting data such as mean and standard error of the data and also save the loss when optimizing.
type Stock struct {
	LossRecord [][]float64
	meanListX  []float64
	stdListX   []float64
	meanListY  []float64
	stdListY   []float64
}

// NetworkStruction :Use as the input of newNN func. Neuron mean the number of node of each layer. Dropout mean the dropout ratio and Act denote the activation func of each layer.
type NetworkStruction struct {
	Neuron       []int
	Dropout      []float64
	Act          []ActivationFunc
	Bias         bool
	Normal       bool
	L1reg        float64
	L2reg        float64
}

// TrainingParameter :Stock the fitting parameters.
type TrainingParameter struct {
	Lr        float64
	Epoches   int
	BatchSize int
	Lossfunc  LossFunc
	Solver    string
}

// InitParameter :Initial the default parameter. It is not necessary.
func InitParameter() TrainingParameter {
	return TrainingParameter{
		Lr:        0.01,
		Epoches:   200,
		BatchSize: 500,
		Solver:    "RMSProp",
		Lossfunc:  MSError,
	}
}

// paraDelivery : deliver training parameters to optimizer.
type paraDelivery struct {
	batches     int
	batchsize   int
	inputShape  int
	outputShape int
	costVal     gorgonia.Value
	para        TrainingParameter
	samplesize  int
	S           Stock
}

// NN : Neural Network itself.
type NN struct {
	G *gorgonia.ExprGraph
	W []*gorgonia.Node
	B []*gorgonia.Node
	D []float64
	A []ActivationFunc

	Pred    *gorgonia.Node
	PredVal gorgonia.Value

	Normal       bool
	L1reg        float64
	L2reg        float64
	FitStock     Stock
}

// Learnables : Return the Nodes that should be optimized.
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

// NewNN : Create the network.
func NewNN(g *gorgonia.ExprGraph, S NetworkStruction) *NN {
	// Set random seed
	rand.Seed(time.Now().Unix())
	var Ns, Bs gorgonia.Nodes
	for i := 0; i < len(S.Neuron)-1; i++ {
		Ns = append(Ns, gorgonia.NewMatrix(
			g,
			tensor.Float64,
			gorgonia.WithShape(S.Neuron[i], S.Neuron[i+1]),
			gorgonia.WithName("w"+strconv.Itoa(i)),
			gorgonia.WithInit(gorgonia.GlorotN(1)),
		))
	}
	if S.Bias {
		for i := 0; i < len(S.Neuron)-1; i++ {
			Bs = append(Bs, gorgonia.NewMatrix(
				g,
				tensor.Float64,
				gorgonia.WithShape(1, S.Neuron[i+1]),
				gorgonia.WithName("b"+strconv.Itoa(i)),
				gorgonia.WithInit(gorgonia.Zeroes()),
			))
		}
	}

	return &NN{
		G:            g,
		W:            Ns,
		B:            Bs,
		D:            S.Dropout,
		A:            S.Act,
		Normal:       S.Normal,
		L1reg:        S.L1reg,
		L2reg:        S.L2reg,
	}
}

// Forward : perform forward pass espectially for auto differential.
func (m *NN) Forward(x *gorgonia.Node) (err error) {
	l := make([]*gorgonia.Node, len(m.W)+1)
	ldot := make([]*gorgonia.Node, len(m.W))
	p := make([]*gorgonia.Node, len(m.W))

	// initial the first layer
	l[0] = x

	// W X + B
	for i := 0; i < len(m.W); i++ {
		if len(m.B) != 0 && i < len(m.W) {
			L1, err := gorgonia.Mul(l[i], m.W[i])
			if err != nil {
				fmt.Println(l[i].Shape(), m.W[i].Shape())
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

		// Dropout
		p[i], err = gorgonia.Dropout(ldot[i], m.D[i])
		if err != nil {
			log.Printf("Can't drop!")
		}

		//activation function
		l[i+1] = gorgonia.Must(m.A[i](p[i]))
	}

	m.Pred = gorgonia.Must(m.A[len(m.A)-1](l[len(l)-1]))
	gorgonia.Read(m.Pred, &m.PredVal)
	return
}

// ValueToFloatSlice : transfer the prediction in the network to [][]float64. A function for predict.
func (m NN) ValueToFloatSlice() [][]float64 {
	oneDimSlice := m.PredVal.Data().([]float64)
	outputShape := m.W[len(m.W)-1].Shape()[1]

	tmp := make([]float64, 0, outputShape+1)
	capabilityOfResult := int(float64(len(oneDimSlice))/float64(outputShape)) + 1
	result := make([][]float64, 0, capabilityOfResult)

	for i := 0; i < len(oneDimSlice); i++ {
		tmp = append(tmp, oneDimSlice[i])
		if len(tmp) == outputShape {
			result = append(result, tmp)
			tmp = make([]float64, 0, outputShape+1)
		}
	}
	return result
}

// CloneModel : copy the model structure for change the I/O shape.
func (m *NN) CloneModel(newG *gorgonia.ExprGraph) *NN {
	ww := make([]*gorgonia.Node, 0, len(m.W)+1)
	bb := make([]*gorgonia.Node, 0, len(m.B)+1)

	for i := 0; i < len(m.W); i++ {
		xT := tensor.New(tensor.WithBacking(m.W[i].Value().Data()), tensor.WithShape(m.W[i].Shape()[0], m.W[i].Shape()[1]))
		w := gorgonia.NodeFromAny(newG, xT, gorgonia.WithName("w"+strconv.Itoa(i)), gorgonia.WithShape(m.W[i].Shape()[0], m.W[i].Shape()[1]))

		ww = append(ww, w)
	}
	for i := 0; i < len(m.B); i++ {
		xT := tensor.New(tensor.WithBacking(m.B[i].Value().Data()), tensor.WithShape(m.B[i].Shape()[0], m.B[i].Shape()[1]))
		b := gorgonia.NodeFromAny(newG, xT, gorgonia.WithName("b"+strconv.Itoa(i)), gorgonia.WithShape(m.B[i].Shape()[0], m.B[i].Shape()[1]))
		bb = append(bb, b)
	}

	return &NN{
		G:            newG,
		W:            ww,
		B:            bb,
		D:            m.D,
		A:            m.A,
		Normal:       m.Normal,
		FitStock:     m.FitStock,
	}
}

// Predict : perform forward feed of neural network with respect to input x.
func (m *NN) Predict(x [][]float64) (predictionGen [][]float64) {
	// Define the needed parameters.
	inputShape := m.W[0].Shape()[0]
	sampleSize := len(x)
	batchSize, batches := CalBatch(sampleSize, sampleSize)

	// Create a New Model
	zeroDrop := make([]float64, 0, len(m.D)+1)
	for i := 0; i < len(m.D); i++ {
		zeroDrop = append(zeroDrop, 0.)
	}
	newG := gorgonia.NewGraph()
	m1 := m.CloneModel(newG)
	m1.D = zeroDrop

	//Normalize the input data. And stock the information into m1.FitStock.
	inputX := x
	if m1.Normal {
		inputX = NormalizeAdjust(x, m1.FitStock.meanListX, m1.FitStock.stdListX)
	}

	oneDimX := ToOneDimSlice(inputX)
	oneDimX = append(oneDimX, oneDimX[len(oneDimX)-inputShape:]...)

	// Construct the input data tensor.
	xT := tensor.New(tensor.WithBacking(oneDimX), tensor.WithShape(sampleSize+1, inputShape))

	// make prediction in batch.
	prediction := make([][]float64, 0, sampleSize+5)

	// Start batches
	for b := 0; b < batches; b++ {
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
			end++
		}
		//slice data Note: xT and xVal are same type but different size.
		xVal, err := xT.Slice(sli{start, end})
		if err != nil {
			log.Fatal(err, "Can't slice the data")
		}

		// Define input node.
		X := gorgonia.NewMatrix(m1.G, tensor.Float64, gorgonia.WithShape(end-start, inputShape), gorgonia.WithName("X"))

		// Construct forward pass and record it using tape machine.
		m1.Forward(X)

		// Dump it, still need tape machine to activate the process.
		gorgonia.Let(X, xVal)

		vm := gorgonia.NewTapeMachine(m1.G, gorgonia.BindDualValues(m1.Learnables()...))
		// Activate the tape machine.
		vm.RunAll()
		vm.Reset()

		// Append the result.
		if over == 1 {
			prediction = append(prediction, m1.ValueToFloatSlice()[0])
		} else {
			prediction = append(prediction, m1.ValueToFloatSlice()...)
		}

	}

	// generalize the output using the data which stock in m1.FitStock.
	if m1.Normal {
		predictionGen = Generalize(prediction, m1.FitStock.meanListY, m1.FitStock.stdListY)
	} else {
		predictionGen = prediction
	}
	return predictionGen
}

// Fit : training neural network.
func (m *NN) Fit(x, y [][]float64, para TrainingParameter) {
	newX := make([][]float64, 0, 2*len(x)+1)
	newY := make([][]float64, 0, 2*len(y)+1)

	for i := 0; i < len(x); i++ {
		newX = append(newX, x[i])
		newX = append(newX, x[i])
		newY = append(newY, y[i])
		newY = append(newY, y[i])
	}
	x = newX
	y = newY
	//============================================================
	inputX := x
	inputY := y

	//Normalize the input data. And stock the information into m.FitStock.
	S := Stock{}
	if m.Normal {
		inputX, S.meanListX, S.stdListX = Normalized(x)
		inputY, S.meanListY, S.stdListY = Normalized(y)
	}

	// set S into struct m.
	m.FitStock = S

	// reshape data
	flattenX := ToOneDimSlice(inputX)
	flattenY := ToOneDimSlice(inputY)
	sampleSize := len(inputX)

	if sampleSize != len(inputY) {
		panic("x and y are not in the same size!")
	}

	// Define shapes
	inputShape := m.W[0].Shape()[0]
	outputShape := m.W[len(m.W)-1].Shape()[1]

	// batch size will not greater than sample size and won't less than 2. Since batch size equal to 1 will crash the model.
	batchSize, batches := CalBatch(sampleSize, para.BatchSize)

	// Construct the input data tensor and node.
	xT := tensor.New(tensor.WithBacking(flattenX), tensor.WithShape(sampleSize, inputShape))
	yT := tensor.New(tensor.WithBacking(flattenY), tensor.WithShape(sampleSize, outputShape))

	// costVal will be used outside the loop.
	var costVal gorgonia.Value

	delivery := paraDelivery{
		batches:     batches,
		batchsize:   batchSize,
		inputShape:  inputShape,
		outputShape: outputShape,
		costVal:     costVal,
		para:        para,
		samplesize:  sampleSize,
		S:           S,
	}

	// Since different optimizer are not the same type. We should rewrite code.
	
	if para.Solver == "Adam" {
		m._AdamTrain(xT, yT, delivery)
	} /* else if para.Solver == "RMSProp" {
		m._RMSPropTrain(xT, yT, delivery)
	} */
	log.Printf("training finish!")
}

func (m *NN) _AdamTrain_(xT, yT *tensor.Dense, delivery paraDelivery) {
	batches := delivery.batches
	batchSize := delivery.batchsize
	inputShape := delivery.inputShape
	outputShape := delivery.outputShape
	costVal := delivery.costVal
	para := delivery.para
	sampleSize := delivery.samplesize
	S := delivery.S
	learningRate := para.Lr

	solver := gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learningRate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))
	// Start epoches training
	for epoch := 0; epoch < para.Epoches; epoch++ {
		if epoch == int(para.Epoches/2) {
			learningRate /= 10
			solver = gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learningRate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))
		}
		// Start batches...
		for b := 0; b < batches; b++ {

			// Handling the batch
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
				log.Fatal("Unable to update gradient")
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
		/* if epoch%10 == 0 {
			fmt.Println("Iteration: ", epoch, "  Cost: ", costVal)
		}  */

		// Stock it.
		S.LossRecord = append(S.LossRecord, []float64{float64(epoch), costVal.Data().(float64)})
	}
	m.FitStock.LossRecord = S.LossRecord
}

func (m *NN) _AdamTrain(xT, yT *tensor.Dense, delivery paraDelivery) {
	//batches := delivery.batches
	batchSize := delivery.batchsize
	inputShape := delivery.inputShape
	outputShape := delivery.outputShape
	costVal := delivery.costVal
	para := delivery.para
	sampleSize := delivery.samplesize
	S := delivery.S
	learningRate := para.Lr

	solver := gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learningRate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))
	// Start epoches training
	for epoch := 0; epoch < para.Epoches; epoch++ {
		if epoch == int(para.Epoches/2) {
			learningRate /= 10
			solver = gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learningRate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))
		}

		// randomly select
		choose := rand.Intn(int(float64(sampleSize) / 2.))
		start := 2 * choose
		end := 2*choose + 2

		xVal, err := xT.Slice(sli{start, end})
		if err != nil {
			log.Fatal(err, "Can't slice the X data")
		}
		yVal, err := yT.Slice(sli{start, end})
		if err != nil {
			log.Fatal(err, "Can't slice the Y data")
		}

		// Define input output node
		X := gorgonia.NewMatrix(m.G, tensor.Float64, gorgonia.WithShape(2, inputShape), gorgonia.WithName("X"))
		y := gorgonia.NewMatrix(m.G, tensor.Float64, gorgonia.WithShape(2, outputShape), gorgonia.WithName("y"))

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
			log.Fatal("Unable to update gradient")
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

		// Print cost
		/* if epoch%10 == 0 {
			fmt.Println("Iteration: ", epoch, "  Cost: ", costVal)
		} */

		// Stock it.
		S.LossRecord = append(S.LossRecord, []float64{float64(epoch), costVal.Data().(float64)})
	}
	m.FitStock.LossRecord = S.LossRecord
}

func (m *NN) _RMSPropTrain(xT, yT *tensor.Dense, delivery paraDelivery) {
	batches := delivery.batches
	batchSize := delivery.batchsize
	inputShape := delivery.inputShape
	outputShape := delivery.outputShape
	costVal := delivery.costVal
	para := delivery.para
	sampleSize := delivery.samplesize
	S := delivery.S
	learningRate := para.Lr

	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learningRate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))

	// Start epoches training
	for epoch := 1; epoch < para.Epoches+1; epoch++ {
		if epoch == int(para.Epoches/2) {
			learningRate /= 10
			solver = gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learningRate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))
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
				log.Fatal("Unable to update gradient")
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

		/* // Print cost
		if epoch%100 == 0 {
			fmt.Println("Iteration: ", epoch, "  Cost: ", costVal)
		} */

		// Stock it.
		S.LossRecord = append(S.LossRecord, []float64{float64(epoch), costVal.Data().(float64)})
	}
	m.FitStock.LossRecord = S.LossRecord
}

// TearApart : Separate the neural network apart. Since it has same graph as the original network, it can be trained separtely.
func (m *NN) TearApart(g *gorgonia.ExprGraph, layerIn, layerOut int) NN {
	newM := NN{}
	// it should use the same graph
	newM.G = g
	newM.W = m.W[layerIn:layerOut]
	newM.D = m.D[layerIn:layerOut]
	newM.A = m.A[layerIn:layerOut]
	return newM
}

// CrossFit : fit that return the best number of epoch.
func (m *NN) CrossFit(x, y [][]float64, xtest, ytest [][]float64, para TrainingParameter) (bestEpoch int, bestMAB float64) {
	newX := make([][]float64, 0, 2*len(x)+1)
	newY := make([][]float64, 0, 2*len(y)+1)
	for i := 0; i < len(x); i++ {
		newX = append(newX, x[i])
		newX = append(newX, x[i])
		newY = append(newY, y[i])
		newY = append(newY, y[i])
	}
	x = newX
	y = newY
	//============================================================

	inputX := x
	inputY := y

	//Normalize the input data. And stock the information into m.FitStock.
	S := Stock{}
	if m.Normal {
		inputX, S.meanListX, S.stdListX = Normalized(x)
		inputY, S.meanListY, S.stdListY = Normalized(y)
	}

	// set S into struct m.
	m.FitStock = S

	// reshape data
	flattenX := ToOneDimSlice(inputX)
	flattenY := ToOneDimSlice(inputY)
	sampleSize := len(inputX)

	if sampleSize != len(inputY) {
		panic("x and y are not in the same size!")
	}

	// Define shapes
	inputShape := m.W[0].Shape()[0]
	outputShape := m.W[len(m.W)-1].Shape()[1]

	// batch size will not greater than sample size and won't less than 2. Since batch size equal to 1 will crash the model.
	batchSize, batches := CalBatch(sampleSize, para.BatchSize)

	// Construct the input data tensor and node.
	xT := tensor.New(tensor.WithBacking(flattenX), tensor.WithShape(sampleSize, inputShape))
	yT := tensor.New(tensor.WithBacking(flattenY), tensor.WithShape(sampleSize, outputShape))

	// costVal will be used outside the loop.
	var costVal gorgonia.Value

	delivery := paraDelivery{
		batches:     batches,
		batchsize:   batchSize,
		inputShape:  inputShape,
		outputShape: outputShape,
		costVal:     costVal,
		para:        para,
		samplesize:  sampleSize,
		S:           S,
	}

	// Since different optimizer are not the same type. We should rewrite code.

	bestEpoch, bestMAB = m._CrossAdamTrain(xT, yT, xtest, ytest, delivery)

	log.Printf("training finish!")
	return
}

func (m *NN) _CrossAdamTrain__(xT, yT *tensor.Dense, xtest, ytest [][]float64, delivery paraDelivery) (int, float64) {
	batches := delivery.batches
	batchSize := delivery.batchsize
	inputShape := delivery.inputShape
	outputShape := delivery.outputShape
	costVal := delivery.costVal
	para := delivery.para
	sampleSize := delivery.samplesize
	S := delivery.S
	learningRate := para.Lr
	bestEpoch := para.Epoches
	bestMAB := 0.

	solver := gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learningRate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))
	var crossValiData [][]float64

	// Start epoches training
	for epoch := 0; epoch < para.Epoches; epoch++ {
		if epoch == int(para.Epoches/2) {
			learningRate /= 10
			solver = gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learningRate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))
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
				log.Fatal("Unable to update gradient")
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

		// cross validate
		if epoch%10 == 0 {
			//fmt.Println("Iteration: ", epoch, "  Cost: ", costVal)
			pred := m.Predict(xtest)
			// MAB
			MAB := 0.
			for k := 0; k < len(pred); k++ {
				AB := math.Abs(ytest[k][0] - pred[k][0])
				MAB += AB / float64(len(pred))
			}
			crossValiData = append(crossValiData, []float64{float64(epoch), MAB})
		}
		// Stock it.
		S.LossRecord = append(S.LossRecord, []float64{float64(epoch), costVal.Data().(float64)})
	}
	m.FitStock.LossRecord = S.LossRecord

	bestEpoch, bestMAB = 10, crossValiData[0][1]
	for k := 0; k < len(crossValiData); k++ {
		if bestMAB > crossValiData[k][1] {
			bestMAB = crossValiData[k][1]
			bestEpoch = int(crossValiData[k][0])
		}
	}
	fmt.Println("The epoches change to: ", bestEpoch)
	return bestEpoch, bestMAB
}
func (m *NN) _CrossAdamTrain(xT, yT *tensor.Dense, xtest, ytest [][]float64, delivery paraDelivery) (int, float64) {
	//batches := delivery.batches
	batchSize := delivery.batchsize
	inputShape := delivery.inputShape
	outputShape := delivery.outputShape
	costVal := delivery.costVal
	para := delivery.para
	sampleSize := delivery.samplesize
	S := delivery.S
	learningRate := para.Lr
	bestEpoch := para.Epoches
	bestMAB := 0.

	solver := gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learningRate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))

	crossValiData := make([][]float64, 0, int(float64(para.Epoches)/5.)+1)

	// Start epoches training
	for epoch := 0; epoch < para.Epoches; epoch++ {
		if epoch == int(para.Epoches/2) {
			learningRate /= 10
			solver = gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learningRate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))
		}

		// randomly select
		choose := rand.Intn(int(float64(sampleSize) / 2.))
		start := 2 * choose
		end := 2*choose + 2

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
		X := gorgonia.NewMatrix(m.G, tensor.Float64, gorgonia.WithShape(2, inputShape), gorgonia.WithName("X"))
		y := gorgonia.NewMatrix(m.G, tensor.Float64, gorgonia.WithShape(2, outputShape), gorgonia.WithName("y"))

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
			log.Fatal("Unable to update gradient")
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

		// cross validate
		if epoch%10 == 0 {
			//fmt.Println("Iteration: ", epoch, "  Cost: ", costVal)
			pred := m.Predict(xtest)
			// MAB
			MAB := 0.
			for k := 0; k < len(pred); k++ {
				AB := math.Abs(ytest[k][0] - pred[k][0])
				MAB += AB / float64(len(pred))
			}
			crossValiData = append(crossValiData, []float64{float64(epoch), MAB})

			// Stock it.
			S.LossRecord = append(S.LossRecord, []float64{float64(epoch), costVal.Data().(float64)})
		}
		m.FitStock.LossRecord = S.LossRecord
	}

	bestEpoch, bestMAB = 10, crossValiData[0][1]
	for k := 0; k < len(crossValiData); k++ {
		if bestMAB > crossValiData[k][1] {
			bestMAB = crossValiData[k][1]
			bestEpoch = int(crossValiData[k][0])
		}
	}
	fmt.Println("The epoches change to: ", bestEpoch)
	return bestEpoch, bestMAB
}

// ExampleAuto : example of autoencoder.
func ExampleAuto() {
	fmt.Println("start!")

	fakeX := [][]float64{[]float64{1, 1, 1, 1}, []float64{2, 2, 2, 2}, []float64{3, 3, 3, 3}, []float64{4, 4, 4, 4}, []float64{5, 5, 5, 5}}

	g := gorgonia.NewGraph()

	S := NetworkStruction{
		Neuron:  []int{4, 2, 1, 2, 4},
		Dropout: []float64{0, 0, 0, 0, 0},
		Act:     []ActivationFunc{gorgonia.Mish, Linear, Linear, Linear, Linear},
	}
	para := InitParameter()

	m := NewNN(g, S)

	fmt.Println(m.W[1].Value())
	m.Fit(fakeX, fakeX, para)

	m1 := m.TearApart(g, 0, 2)
	m1.Predict(fakeX)
	fmt.Println(m1.PredVal)

}

// ExampleNN : example for DNN.
func ExampleNN() {
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
