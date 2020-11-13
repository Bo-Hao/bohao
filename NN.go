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
	LossRecord  [][]float64
	mean_list_x []float64
	std_list_x  []float64
	mean_list_y []float64
	std_list_y  []float64
}

// Use as the input of newNN func. Neuron mean the number of node of each layer. Dropout mean the dropout ratio and Act denote the activation func of each layer.
type NetworkStruction struct {
	Neuron        []int
	Dropout       []float64
	Act           []ActivationFunc
	Bias          bool
	Normal        bool
	Normal_weight []float64
	L1reg         float64
	L2reg         float64
}

// Stock the fitting parameters.
type TrainingParameter struct {
	Lr        float64
	Epoches   int
	BatchSize int
	Lossfunc  LossFunc
	Solver    string
}

// Initial the default parameter. It is not necessary.
func InitParameter() TrainingParameter {
	return TrainingParameter{
		Lr:        0.01,
		Epoches:   200,
		BatchSize: 500,
		Solver:    "RMSProp",
		Lossfunc:  RMSError,
	}
}

type fit_delivery struct {
	batches     int
	batchsize   int
	inputShape  int
	outputShape int
	costVal     gorgonia.Value
	para        TrainingParameter
	samplesize  int
	S           Stock
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

	Normal        bool
	Normal_weight []float64
	L1reg         float64
	L2reg         float64
	FitStock      Stock
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
		G:             g,
		W:             Ns,
		B:             Bs,
		D:             S.Dropout,
		A:             S.Act,
		Normal:        S.Normal,
		Normal_weight: S.Normal_weight,
		L1reg:         S.L1reg,
		L2reg:         S.L2reg,
	}
}

// Forward pass.
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

func (n *NN) Clone_model(new_G *gorgonia.ExprGraph) *NN {
	var ww, bb []*gorgonia.Node
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

	return &NN{
		G:             new_G,
		W:             ww,
		B:             bb,
		D:             n.D,
		A:             n.A,
		Normal:        n.Normal,
		Normal_weight: n.Normal_weight,
		FitStock:      n.FitStock,
	}
}

func (n *NN) Predict(x [][]float64) (prediction_gen [][]float64) {
	// Define the needed parameters.
	inputShape := n.W[0].Shape()[0]
	sampleSize := len(x)
	batchSize, batches := Cal_Batch(sampleSize, sampleSize)

	// Create a New Model
	var zero_drop []float64
	for i := 0; i < len(n.D); i++ {
		zero_drop = append(zero_drop, 0.)
	}
	new_g := gorgonia.NewGraph()
	m := n.Clone_model(new_g)
	m.D = zero_drop

	//Normalize the input data. And stock the information into m.FitStock.
	input_x := x
	if m.Normal {
		input_x = Normalize_adjust(x, m.FitStock.mean_list_x, m.FitStock.std_list_x)
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
		prediction_gen = Generalize(prediction, n.FitStock.mean_list_y, n.FitStock.std_list_y)
	} else {
		prediction_gen = prediction
	}
	return prediction_gen
}

func (m *NN) Fit(x_, y_ [][]float64, para TrainingParameter) {
	input_x := x_
	input_y := y_

	//Normalize the input data. And stock the information into m.FitStock.
	S := Stock{}
	if m.Normal {
		input_x, S.mean_list_x, S.std_list_x = Normalized(x_, m.Normal_weight)
		input_y, S.mean_list_y, S.std_list_y = Normalized(y_, m.Normal_weight)
	}

	// set S into struct m.
	m.FitStock = S

	// reshape data
	x_oneDim := ToOneDimSlice(input_x)
	y_oneDim := ToOneDimSlice(input_y)
	sampleSize := len(input_x)

	if sampleSize != len(input_y) {
		panic("x and y are not in the same size!")
	}

	// Define shapes
	inputShape := m.W[0].Shape()[0]
	outputShape := m.W[len(m.W)-1].Shape()[1]

	// batch size will not greater than sample size and won't less than 2. Since batch size equal to 1 will crash the model.
	batchSize, batches := Cal_Batch(sampleSize, para.BatchSize)

	// Construct the input data tensor and node.
	xT := tensor.New(tensor.WithBacking(x_oneDim), tensor.WithShape(sampleSize, inputShape))
	yT := tensor.New(tensor.WithBacking(y_oneDim), tensor.WithShape(sampleSize, outputShape))

	// costVal will be used outside the loop.
	var costVal gorgonia.Value

	delivery := fit_delivery{
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
	if para.Solver == "RMSProp" {
		m._RMSPropTrain(xT, yT, delivery)

	} else if para.Solver == "Adam" {
		m._AdamTrain(xT, yT, delivery)
	}
}

func (m *NN) _AdamTrain(xT, yT *tensor.Dense, delivery fit_delivery) {
	batches := delivery.batches
	batchSize := delivery.batchsize
	inputShape := delivery.inputShape
	outputShape := delivery.outputShape
	costVal := delivery.costVal
	para := delivery.para
	sampleSize := delivery.samplesize
	S := delivery.S
	learning_rate := para.Lr

	solver := gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learning_rate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))
	// Start epoches training
	for epoch := 0; epoch < para.Epoches; epoch++ {
		if epoch == int(para.Epoches/2) {
			learning_rate /= 10
			solver = gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learning_rate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))
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

func (m *NN) _RMSPropTrain(xT, yT *tensor.Dense, delivery fit_delivery) {
	batches := delivery.batches
	batchSize := delivery.batchsize
	inputShape := delivery.inputShape
	outputShape := delivery.outputShape
	costVal := delivery.costVal
	para := delivery.para
	sampleSize := delivery.samplesize
	S := delivery.S
	learning_rate := para.Lr

	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learning_rate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))

	// Start epoches training
	for epoch := 1; epoch < para.Epoches+1; epoch++ {
		if epoch == int(para.Epoches/2) {
			learning_rate /= 10
			solver = gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learning_rate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))
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

func (m *NN) _TestOverfitting(x, y [][]float64) {
	input_x := x
	input_y := y

	split_x := make([][][]float64, 5)
	split_y := make([][][]float64, 5)
	stratumSize := int(float64(len(input_x)) / 5.)

	stratum := -1
	for i := 0; i < len(input_x); i++ {
		if i%stratumSize == 0 {
			stratum += 1
		}
		split_x[stratum] = append(split_x[stratum], input_x[i])
		split_y[stratum] = append(split_y[stratum], input_y[i])
	}

	for i := 0; i < 5; i++ {
		y_ := m.Predict(split_x[i])
		fmt.Println(RMSE(y_, split_y[i]))
	}
}

func RMSE(y_, y [][]float64) float64 {
	sum := 0.
	fmt.Println(len(y_), len(y))
	for i := 0; i < len(y); i++ {
		sum += math.Pow(y_[i][0]-y[i][0], 2)
	}
	return math.Sqrt(sum / float64(len(y)))
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

func (m *NN) SelfOrganFit(x_, y_ [][]float64, para TrainingParameter) {
	input_x := x_
	input_y := y_
	if len(input_x) != len(input_y) {
		panic("x and y are not in the same size!")
	}

	//Normalize the input data. And stock the information into m.FitStock.
	S := Stock{}
	if m.Normal {
		input_x, S.mean_list_x, S.std_list_x = Normalized(x_, m.Normal_weight)
		input_y, S.mean_list_y, S.std_list_y = Normalized(y_, m.Normal_weight)
	}

	// set S into struct m.
	m.FitStock = S

	// reshape data
	x_oneDim := ToOneDimSlice(input_x)
	y_oneDim := ToOneDimSlice(input_y)
	sampleSize := len(input_x)

	// Define shapes
	inputShape := m.W[0].Shape()[0]
	outputShape := m.W[len(m.W)-1].Shape()[1]

	// batch size will not greater than sample size and won't less than 2. Since batch size equal to 1 will crash the model.
	batchSize, batches := Cal_Batch(sampleSize, para.BatchSize)

	// Construct the input data tensor and node.
	xT := tensor.New(tensor.WithBacking(x_oneDim), tensor.WithShape(sampleSize, inputShape))
	yT := tensor.New(tensor.WithBacking(y_oneDim), tensor.WithShape(sampleSize, outputShape))

	// costVal will be used outside the loop.
	var costVal gorgonia.Value

	delivery := fit_delivery{
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
	if para.Solver == "RMSProp" {
		m._SelfOrganRMSPropTrain(xT, yT, delivery)

	} else if para.Solver == "Adam" {
		m._SelfOrganAdamTrain(xT, yT, delivery)
	}
	log.Printf("training finish!")
}

func (m *NN) _SelfOrganAdamTrain(xT, yT *tensor.Dense, delivery fit_delivery) {
	batches := delivery.batches
	batchSize := delivery.batchsize
	inputShape := delivery.inputShape
	outputShape := delivery.outputShape
	costVal := delivery.costVal
	para := delivery.para
	sampleSize := delivery.samplesize
	S := delivery.S
	learning_rate := para.Lr

	solver := gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learning_rate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))

	var movingMeanLoss []float64

	// Start epoches training
	for epoch := 0; epoch < para.Epoches; epoch++ {
		if epoch == int(para.Epoches/2) {
			learning_rate /= 10
			solver = gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learning_rate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))
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

		// Stock it.
		S.LossRecord = append(S.LossRecord, []float64{float64(epoch), costVal.Data().(float64)})

		// Self Organizing
		checkingPoint := 5.
		if epoch%int(checkingPoint) == 0 && len(S.LossRecord) > 4 {
			mean := 0.
			std := 0.
			for i := 0; i < int(checkingPoint); i++ {
				value := S.LossRecord[len(S.LossRecord)-1-i][1]
				mean += value / checkingPoint
				std += math.Pow(value, 2)
			}

			std = math.Sqrt((std - checkingPoint*mean) / checkingPoint)

			if len(movingMeanLoss) == 0 {
				movingMeanLoss = append(movingMeanLoss, mean)
			} else if math.Abs(mean-movingMeanLoss[len(movingMeanLoss)-1]) < std {
				fmt.Println("Training end early at:", epoch)
				break
			} else {
				movingMeanLoss = append(movingMeanLoss, mean)
			}
		}
	}
	m.FitStock.LossRecord = S.LossRecord
}

func (m *NN) _SelfOrganRMSPropTrain(xT, yT *tensor.Dense, delivery fit_delivery) {
	batches := delivery.batches
	batchSize := delivery.batchsize
	inputShape := delivery.inputShape
	outputShape := delivery.outputShape
	costVal := delivery.costVal
	para := delivery.para
	sampleSize := delivery.samplesize
	S := delivery.S
	learning_rate := para.Lr

	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learning_rate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))
	var movingMeanLoss []float64

	// Start epoches training
	for epoch := 1; epoch < para.Epoches+1; epoch++ {
		if epoch == int(para.Epoches/2) {
			learning_rate /= 10
			solver = gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learning_rate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))
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

		// Self Organizing
		checkingPoint := 5.
		if epoch%int(checkingPoint) == 0 && len(S.LossRecord) > 4 {
			mean := 0.
			std := 0.
			for i := 0; i < int(checkingPoint); i++ {
				value := S.LossRecord[len(S.LossRecord)-1-i][1]
				mean += value / checkingPoint
				std += math.Pow(value, 2)
			}

			std = math.Sqrt((std - checkingPoint*mean) / checkingPoint)

			if len(movingMeanLoss) == 0 {
				movingMeanLoss = append(movingMeanLoss, mean)
			} else if math.Abs(mean-movingMeanLoss[len(movingMeanLoss)-1]) < std {
				fmt.Println("Training end early at:", epoch)
				break
			} else {
				movingMeanLoss = append(movingMeanLoss, mean)
			}
		}
	}
	m.FitStock.LossRecord = S.LossRecord
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
		Act:     []ActivationFunc{gorgonia.Mish, Linear, Linear, Linear, Linear},
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
