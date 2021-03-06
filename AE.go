package bohao

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type AE struct {
	G              *gorgonia.ExprGraph
	W1, W2         *gorgonia.Node
	B1, B2, B3, B4 *gorgonia.Node
	Denoising      float64
	Dropout        []float64
	Acti           []ActivationFunc
	L1reg          float64
	L2reg          float64

	H1, H3 *gorgonia.Node

	Pred    *gorgonia.Node
	PredVal gorgonia.Value

	Normal     bool
	FitStock   Stock
}

type AE_Encoder struct {
	G      *gorgonia.ExprGraph
	W1, W2 *gorgonia.Node
	B1, B2 *gorgonia.Node
	Acti   []ActivationFunc

	Core    *gorgonia.Node
	CoreVal gorgonia.Value

	Normal     bool
	FitStock   Stock
}

type AE_Decoder struct {
	G      *gorgonia.ExprGraph
	W1, W2 *gorgonia.Node
	B3, B4 *gorgonia.Node
	Acti   []ActivationFunc

	Pred    *gorgonia.Node
	PredVal gorgonia.Value

	Normal     bool
	FitStock   Stock
}

type AE_Struction struct {
	InputShape  int
	HiddenShape int
	CoreShape   int

	Denoising  float64
	Dropout    []float64
	Acti       []ActivationFunc
	L1reg      float64
	L2reg      float64
	Normal     bool
}

func (m *AE) LearnablesPhase1() gorgonia.Nodes {
	return gorgonia.Nodes{
		m.W1, m.W2, m.B1, m.B2, m.B3, m.B4,
	}
}

func (m *AE) LearnablesPhase2() gorgonia.Nodes {
	return gorgonia.Nodes{
		m.W2, m.B2, m.B3,
	}
}

func NewAE(g *gorgonia.ExprGraph, S AE_Struction) *AE {
	// Set random seed
	rand.Seed(time.Now().Unix())

	w1 := gorgonia.NewMatrix(
		g,
		tensor.Float64,
		gorgonia.WithShape(S.InputShape, S.HiddenShape),
		gorgonia.WithName("W1"),
		gorgonia.WithInit(gorgonia.GlorotU(1)),
	)
	w2 := gorgonia.NewMatrix(
		g,
		tensor.Float64,
		gorgonia.WithShape(S.HiddenShape, S.CoreShape),
		gorgonia.WithName("W2"),
		gorgonia.WithInit(gorgonia.GlorotU(1)),
	)
	b1 := gorgonia.NewMatrix(
		g,
		tensor.Float64,
		gorgonia.WithShape(1, S.HiddenShape),
		gorgonia.WithName("B1"),
		gorgonia.WithInit(gorgonia.Zeroes()),
	)
	b2 := gorgonia.NewMatrix(
		g,
		tensor.Float64,
		gorgonia.WithShape(1, S.CoreShape),
		gorgonia.WithName("B2"),
		gorgonia.WithInit(gorgonia.Zeroes()),
	)

	b3 := gorgonia.NewMatrix(
		g,
		tensor.Float64,
		gorgonia.WithShape(1, S.HiddenShape),
		gorgonia.WithName("B3"),
		gorgonia.WithInit(gorgonia.Zeroes()),
	)
	b4 := gorgonia.NewMatrix(
		g,
		tensor.Float64,
		gorgonia.WithShape(1, S.InputShape),
		gorgonia.WithName("B4"),
		gorgonia.WithInit(gorgonia.Zeroes()),
	)

	return &AE{
		G:          g,
		W1:         w1,
		W2:         w2,
		B1:         b1,
		B2:         b2,
		B3:         b3,
		B4:         b4,
		Denoising:  S.Denoising,
		Dropout:    S.Dropout,
		Acti:       S.Acti,
		L1reg:      S.L1reg,
		L2reg:      S.L2reg,
		Normal:     S.Normal,
	}
}

// Clone_AE : 
func Clone_AE(m *AE) *AE {
	g := gorgonia.NewGraph()

	xw1T := tensor.New(tensor.WithBacking(m.W1.Value().Data()), tensor.WithShape(m.W1.Shape()[0], m.W1.Shape()[1]))
	w1 := gorgonia.NodeFromAny(g, xw1T, gorgonia.WithName("W1"), gorgonia.WithShape(m.W1.Shape()[0], m.W1.Shape()[1]))

	xw2T := tensor.New(tensor.WithBacking(m.W2.Value().Data()), tensor.WithShape(m.W2.Shape()[0], m.W2.Shape()[1]))
	w2 := gorgonia.NodeFromAny(g, xw2T, gorgonia.WithName("W2"), gorgonia.WithShape(m.W2.Shape()[0], m.W2.Shape()[1]))

	xb1T := tensor.New(tensor.WithBacking(m.B1.Value().Data()), tensor.WithShape(m.B1.Shape()[0], m.B1.Shape()[1]))
	b1 := gorgonia.NodeFromAny(g, xb1T, gorgonia.WithName("B1"), gorgonia.WithShape(m.B1.Shape()[0], m.B1.Shape()[1]))

	xb2T := tensor.New(tensor.WithBacking(m.B2.Value().Data()), tensor.WithShape(m.B2.Shape()[0], m.B2.Shape()[1]))
	b2 := gorgonia.NodeFromAny(g, xb2T, gorgonia.WithName("B2"), gorgonia.WithShape(m.B2.Shape()[0], m.B2.Shape()[1]))

	xb3T := tensor.New(tensor.WithBacking(m.B3.Value().Data()), tensor.WithShape(m.B3.Shape()[0], m.B3.Shape()[1]))
	b3 := gorgonia.NodeFromAny(g, xb3T, gorgonia.WithName("B3"), gorgonia.WithShape(m.B3.Shape()[0], m.B3.Shape()[1]))

	xb4T := tensor.New(tensor.WithBacking(m.B4.Value().Data()), tensor.WithShape(m.B4.Shape()[0], m.B4.Shape()[1]))
	b4 := gorgonia.NodeFromAny(g, xb4T, gorgonia.WithName("B4"), gorgonia.WithShape(m.B4.Shape()[0], m.B4.Shape()[1]))

	return &AE{
		G:          g,
		W1:         w1,
		W2:         w2,
		B1:         b1,
		B2:         b2,
		B3:         b3,
		B4:         b4,
		Dropout:    m.Dropout,
		Acti:       m.Acti,
		Normal:     m.Normal,
	}
}

// Forward : 
func (m *AE) Forward(x *gorgonia.Node) (err error) {
	l := make([]*gorgonia.Node, 5)
	lDot := make([]*gorgonia.Node, 4)
	lAdd := make([]*gorgonia.Node, 4)
	lDrop := make([]*gorgonia.Node, 4)
	var denoise1 *gorgonia.Node
	corruption := gorgonia.BinomialRandomNode(m.G, tensor.Float64, 1, m.Denoising)

	// Denoising layer 1
	l[0] = x
	if denoise1, err = gorgonia.HadamardProd(corruption, l[0]); err != nil {
		return
	}

	// layer 1
	if lDot[0], err = gorgonia.Mul(denoise1, m.W1); err != nil {
		log.Fatal("Can't Mul 1! ", err)
	}

	if lAdd[0], err = gorgonia.BroadcastAdd(lDot[0], m.B1, nil, []byte{0}); err != nil {
		log.Fatal("Can't Add 1! ", err)
	}

	if lDrop[0], err = gorgonia.Dropout(lAdd[0], m.Dropout[0]); err != nil {
		log.Fatal("Can't drop 1! ", err)
	}

	if l[1], err = m.Acti[0](lDrop[0]); err != nil {
		log.Fatal("Can't activate! ", err)
	}

	m.H1 = l[1]

	// layer 2
	if lDot[1], err = gorgonia.Mul(l[1], m.W2); err != nil {
		log.Fatal("Can't Mul 2! ", err)
	}

	if lAdd[1], err = gorgonia.BroadcastAdd(lDot[1], m.B2, nil, []byte{0}); err != nil {
		log.Fatal("Can't Add 2! ", err)
	}

	if lDrop[1], err = gorgonia.Dropout(lAdd[1], m.Dropout[1]); err != nil {
		log.Fatal("Can't drop 2! ", err)
	}

	if l[2], err = m.Acti[1](lDrop[1]); err != nil {
		log.Fatal("Can't activate 2! ", err)
	}

	// layer 3
	W2T, _ := gorgonia.Transpose(m.W2)
	if lDot[2], err = gorgonia.Mul(l[2], W2T); err != nil {
		log.Fatal("Can't Mul 3! ", err)
	}

	if lAdd[2], err = gorgonia.BroadcastAdd(lDot[2], m.B3, nil, []byte{0}); err != nil {
		log.Fatal("Can't Add 3! ", err)
	}

	if lDrop[2], err = gorgonia.Dropout(lAdd[2], m.Dropout[2]); err != nil {
		log.Fatal("Can't drop 3! ", err)
	}

	if l[3], err = m.Acti[2](lDrop[2]); err != nil {
		log.Fatal("Can't activate 3! ", err)
	}
	m.H3 = l[3]

	// layer 4
	W1T, _ := gorgonia.Transpose(m.W1)
	if lDot[3], err = gorgonia.Mul(l[3], W1T); err != nil {
		log.Fatal("Can't Mul 4! ", err)
	}

	if lAdd[3], err = gorgonia.BroadcastAdd(lDot[3], m.B4, nil, []byte{0}); err != nil {
		log.Fatal("Can't Add 4! ", err)
	}

	if lDrop[3], err = gorgonia.Dropout(lAdd[3], m.Dropout[3]); err != nil {
		log.Fatal("Can't drop 4! ", err)
	}

	if l[4], err = m.Acti[3](lDrop[3]); err != nil {
		log.Fatal("Can't activate 4! ", err)
	}

	m.Pred = l[4]
	gorgonia.Read(m.Pred, &m.PredVal)
	return
}

// Fit : train autoencoder.
func (m *AE) Fit(x [][]float64, para TrainingParameter) {
	inputX := x

	//Normalize the input data. And stock the information into m.FitStock.
	S := Stock{}
	if m.Normal {
		inputX, S.meanListX, S.stdListX = Normalized(x)
	}

	// set S into struct m.
	m.FitStock = S

	// reshape data
	flattenX := ToOneDimSlice(inputX)
	sampleSize := len(inputX)

	// Define shapes
	inputShape := m.W1.Shape()[0]

	// batch size will not greater than sample size and won't less than 2. Since batch size equal to 1 will crash the model.
	batchSize, batches := CalBatch(sampleSize, para.BatchSize)

	// Construct the input data tensor and node.
	xT := tensor.New(tensor.WithBacking(flattenX), tensor.WithShape(sampleSize, inputShape))

	// costVal will be used outside the loop.
	var costVal gorgonia.Value

	// Set up parameters for training.
	delivery := paraDelivery{
		batches:     batches,
		batchsize:   batchSize,
		inputShape:  inputShape,
		outputShape: inputShape,
		costVal:     costVal,
		para:        para,
		samplesize:  sampleSize,
		S:           S,
	}
	m._AdamTrain(xT, delivery)

	// Since different optimizer are not the same type. We should rewrite code.
	/* if para.Solver == "RMSProp" {
		m._RMSPropTrain(xT, delivery)

	} else if para.Solver == "Adam" {
		m._AdamTrain(xT, delivery)
	} */
	log.Printf("training finish!")
}

func (m *AE) _AdamTrain(xT *tensor.Dense, delivery paraDelivery) {
	batches := delivery.batches
	batchSize := delivery.batchsize
	inputShape := delivery.inputShape
	para := delivery.para
	sampleSize := delivery.samplesize
	S := delivery.S
	learningRate := para.Lr

	var costValPhase1, costValPhase2 gorgonia.Value
	// Start epoches training
	// Phase 1
	solver := gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learningRate), gorgonia.WithL1Reg(m.L1reg), gorgonia.WithL2Reg(m.L2reg))
	for epoch := 0; epoch < int(para.Epoches/2); epoch++ {
		if epoch == int(para.Epoches/4) {
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

			// Define input output node
			X := gorgonia.NewMatrix(m.G, tensor.Float64, gorgonia.WithShape(batchSize-over, inputShape), gorgonia.WithName("X"))

			// forward pass
			err = m.Forward(X)
			if err != nil {
				log.Fatal(err, "Forward pass fail")
			}

			// Define the loss function.
			costPhase1 := para.Lossfunc(m.Pred, X)

			// Record cost change
			gorgonia.Read(costPhase1, &costValPhase1)

			// Update the gradient.
			if _, err = gorgonia.Grad(costPhase1, m.LearnablesPhase1()...); err != nil {
				log.Fatal("Unable to update gradient 1. ", err)
			}

			// Define the tape machine to record the gradient change for the nodes which should be optimized or activated.
			vm := gorgonia.NewTapeMachine(m.G, gorgonia.BindDualValues(m.LearnablesPhase1()...))

			// Dump it
			gorgonia.Let(X, xVal)

			// Optimizing...
			vm.Reset()
			vm.RunAll()
			solver.Step(gorgonia.NodesToValueGrads(m.LearnablesPhase1()))
			vm.Reset()
		}

		// Print cost
		if epoch%100 == 0 {
			fmt.Println("Phase 1: Iteration: ", epoch, "  Cost: ", costValPhase1)
		}
		// Stock it.
		S.LossRecord = append(S.LossRecord, []float64{float64(epoch), costValPhase1.Data().(float64)})
	}
	m.FitStock.LossRecord = S.LossRecord

	// Phase 2
	m.Denoising = 0.
	learningRate = para.Lr
	solver = gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learningRate))
	for epoch := 0; epoch < para.Epoches-int(para.Epoches/2); epoch++ {
		if epoch == int(para.Epoches/4) {
			learningRate /= 10
			solver = gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learningRate))
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

			// Define input output node
			X := gorgonia.NewMatrix(m.G, tensor.Float64, gorgonia.WithShape(batchSize-over, inputShape), gorgonia.WithName("X"))

			// forward pass
			err = m.Forward(X)
			if err != nil {
				log.Fatal(err, "Forward pass fail")
			}

			// Define the loss function.
			costPhase2 := para.Lossfunc(m.H3, m.H1)

			// Record cost change
			gorgonia.Read(costPhase2, &costValPhase2)

			// Update the gradient.
			if _, err = gorgonia.Grad(costPhase2, m.LearnablesPhase2()...); err != nil {
				log.Fatal("Unable to update gradient 2.", err)
			}

			// Define the tape machine to record the gradient change for the nodes which should be optimized or activated.
			vm := gorgonia.NewTapeMachine(m.G, gorgonia.BindDualValues(m.LearnablesPhase2()...))

			// Dump it
			gorgonia.Let(X, xVal)

			// Optimizing...
			vm.Reset()
			vm.RunAll()
			solver.Step(gorgonia.NodesToValueGrads(m.LearnablesPhase2()))
			vm.Reset()
		}

		// Print cost
		if epoch%100 == 0 {
			fmt.Println("Phase 2: Iteration: ", epoch, "  Cost: ", costValPhase2)
		}
		// Stock it.
		S.LossRecord = append(S.LossRecord, []float64{float64(epoch), costValPhase2.Data().(float64)})
	}
	m.FitStock.LossRecord = S.LossRecord
}

func (m *AE) _encoder() *AE_Encoder {
	g := gorgonia.NewGraph()

	xw1T := tensor.New(tensor.WithBacking(m.W1.Value().Data()), tensor.WithShape(m.W1.Shape()[0], m.W1.Shape()[1]))
	w1 := gorgonia.NodeFromAny(g, xw1T, gorgonia.WithName("W1"), gorgonia.WithShape(m.W1.Shape()[0], m.W1.Shape()[1]))

	xw2T := tensor.New(tensor.WithBacking(m.W2.Value().Data()), tensor.WithShape(m.W2.Shape()[0], m.W2.Shape()[1]))
	w2 := gorgonia.NodeFromAny(g, xw2T, gorgonia.WithName("W2"), gorgonia.WithShape(m.W2.Shape()[0], m.W2.Shape()[1]))

	xb1T := tensor.New(tensor.WithBacking(m.B1.Value().Data()), tensor.WithShape(m.B1.Shape()[0], m.B1.Shape()[1]))
	b1 := gorgonia.NodeFromAny(g, xb1T, gorgonia.WithName("B1"), gorgonia.WithShape(m.B1.Shape()[0], m.B1.Shape()[1]))

	xb2T := tensor.New(tensor.WithBacking(m.B2.Value().Data()), tensor.WithShape(m.B2.Shape()[0], m.B2.Shape()[1]))
	b2 := gorgonia.NodeFromAny(g, xb2T, gorgonia.WithName("B2"), gorgonia.WithShape(m.B2.Shape()[0], m.B2.Shape()[1]))

	return &AE_Encoder{
		G:          g,
		W1:         w1,
		W2:         w2,
		B1:         b1,
		B2:         b2,
		Acti:       m.Acti[0:2],
		Normal:     m.Normal,
		FitStock:   m.FitStock,
	}
}

func (m *AE) _decoder() *AE_Decoder {
	g := gorgonia.NewGraph()

	xw1T := tensor.New(tensor.WithBacking(m.W1.Value().Data()), tensor.WithShape(m.W1.Shape()[0], m.W1.Shape()[1]))
	w1 := gorgonia.NodeFromAny(g, xw1T, gorgonia.WithName("W1"), gorgonia.WithShape(m.W1.Shape()[0], m.W1.Shape()[1]))

	xw2T := tensor.New(tensor.WithBacking(m.W2.Value().Data()), tensor.WithShape(m.W2.Shape()[0], m.W2.Shape()[1]))
	w2 := gorgonia.NodeFromAny(g, xw2T, gorgonia.WithName("W2"), gorgonia.WithShape(m.W2.Shape()[0], m.W2.Shape()[1]))

	xb3T := tensor.New(tensor.WithBacking(m.B3.Value().Data()), tensor.WithShape(m.B3.Shape()[0], m.B3.Shape()[1]))
	b3 := gorgonia.NodeFromAny(g, xb3T, gorgonia.WithName("B3"), gorgonia.WithShape(m.B3.Shape()[0], m.B3.Shape()[1]))

	xb4T := tensor.New(tensor.WithBacking(m.B4.Value().Data()), tensor.WithShape(m.B4.Shape()[0], m.B4.Shape()[1]))
	b4 := gorgonia.NodeFromAny(g, xb4T, gorgonia.WithName("B4"), gorgonia.WithShape(m.B4.Shape()[0], m.B4.Shape()[1]))

	return &AE_Decoder{
		G:          g,
		W1:         w1,
		W2:         w2,
		B3:         b3,
		B4:         b4,
		Acti:       m.Acti[2:],
		Normal:     m.Normal,
		FitStock:   m.FitStock,
	}
}

func (m *AE_Encoder) fwd(x *gorgonia.Node) (err error) {
	l := make([]*gorgonia.Node, 3)
	lDot := make([]*gorgonia.Node, 2)
	lAdd := make([]*gorgonia.Node, 2)

	l[0] = x

	// layer 1
	if lDot[0], err = gorgonia.Mul(l[0], m.W1); err != nil {
		log.Fatal("Can't Mul 1! ", err)
	}

	if lAdd[0], err = gorgonia.BroadcastAdd(lDot[0], m.B1, nil, []byte{0}); err != nil {
		log.Fatal("Can't Add 1! ", err)
	}

	if l[1], err = m.Acti[0](lAdd[0]); err != nil {
		log.Fatal("Can't activate! ", err)
	}

	// layer 2
	if lDot[1], err = gorgonia.Mul(l[1], m.W2); err != nil {
		log.Fatal("Can't Mul 2! ", err)
	}

	if lAdd[1], err = gorgonia.BroadcastAdd(lDot[1], m.B2, nil, []byte{0}); err != nil {
		log.Fatal("Can't Add 2! ", err)
	}

	if l[2], err = m.Acti[1](lAdd[1]); err != nil {
		log.Fatal("Can't activate 2! ", err)
	}

	m.Core = l[2]
	gorgonia.Read(m.Core, &m.CoreVal)

	return
}

func (m *AE_Decoder) fwd(c *gorgonia.Node) (err error) {
	l := make([]*gorgonia.Node, 3)
	lDot := make([]*gorgonia.Node, 2)
	lAdd := make([]*gorgonia.Node, 2)

	// Denoising layer 1
	l[0] = c

	// layer 3
	W2T, _ := gorgonia.Transpose(m.W2)
	if lDot[0], err = gorgonia.Mul(l[0], W2T); err != nil {
		log.Fatal("Can't Mul 3! ", err)
	}

	if lAdd[0], err = gorgonia.BroadcastAdd(lDot[0], m.B3, nil, []byte{0}); err != nil {
		log.Fatal("Can't Add 3! ", err)
	}

	if l[1], err = m.Acti[0](lAdd[0]); err != nil {
		log.Fatal("Can't activate 3! ", err)
	}

	// layer 4
	W1T, _ := gorgonia.Transpose(m.W1)
	if lDot[1], err = gorgonia.Mul(l[1], W1T); err != nil {
		log.Fatal("Can't Mul 4! ", err)
	}

	if lAdd[1], err = gorgonia.BroadcastAdd(lDot[1], m.B4, nil, []byte{0}); err != nil {
		log.Fatal("Can't Add 4! ", err)
	}

	if l[2], err = m.Acti[1](lAdd[1]); err != nil {
		log.Fatal("Can't activate 4! ", err)
	}

	m.Pred = l[2]
	gorgonia.Read(m.Pred, &m.PredVal)

	return
}

// Encode : 
func (m *AE) Encode(x [][]float64) [][]float64 {
	encoder := m._encoder()
	S := encoder.FitStock

	inputShape := encoder.W1.Shape()[0]
	sampleSize := len(x)
	batchSize, batches := CalBatch(sampleSize, sampleSize)

	//Normalize the input data. And stock the information into m.FitStock.
	inputX := x
	if encoder.Normal {
		inputX = NormalizeAdjust(x, S.meanListX, S.stdListX)
	}

	flattenX := ToOneDimSlice(inputX)
	for i := 0; i < inputShape; i++ {
		flattenX = append(flattenX, flattenX[len(flattenX)-inputShape+i])
	}

	// Construct the input data tensor.
	xT := tensor.New(tensor.WithBacking(flattenX), tensor.WithShape(sampleSize+1, inputShape))

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
			end ++
		}
		//slice data Note: xT and xVal are same type but different size.
		xVal, err := xT.Slice(sli{start, end})
		if err != nil {
			log.Fatal(err, "Can't slice the data")
		}

		// Define input node.
		X := gorgonia.NewMatrix(encoder.G, tensor.Float64, gorgonia.WithShape(end-start, inputShape), gorgonia.WithName("X"))

		// Construct forward pass and record it using tape machine.
		encoder.fwd(X)

		// Dump it, still need tape machine to activate the process.
		gorgonia.Let(X, xVal)

		vm := gorgonia.NewTapeMachine(encoder.G)
		// Activate the tape machine.
		vm.RunAll()
		vm.Reset()

		// Append the result.
		if over == 1 {
			prediction = append(prediction, Value2Float(encoder.CoreVal, encoder.W2.Shape()[1])[0])
		} else {
			prediction = append(prediction, Value2Float(encoder.CoreVal, encoder.W2.Shape()[1])...)
		}
	}
	return prediction
}

// Decode : 
func (m *AE) Decode(core [][]float64) (prediction_gen [][]float64) {
	decoder := m._decoder()
	S := decoder.FitStock

	inputShape := decoder.W2.Shape()[1]
	sampleSize := len(core)
	batchSize, batches := CalBatch(sampleSize, sampleSize)

	//Normalize the input data. And stock the information into m.FitStock.
	inputX := core

	flattenX := ToOneDimSlice(inputX)
	for i := 0; i < inputShape; i++ {
		flattenX = append(flattenX, flattenX[len(flattenX)-inputShape+i])
	}

	// Construct the input data tensor.
	xT := tensor.New(tensor.WithBacking(flattenX), tensor.WithShape(sampleSize+1, inputShape))

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
			end ++
		}
		//slice data Note: xT and xVal are same type but different size.
		xVal, err := xT.Slice(sli{start, end})
		if err != nil {
			log.Fatal(err, "Can't slice the data")
		}

		// Define input node.
		X := gorgonia.NewMatrix(decoder.G, tensor.Float64, gorgonia.WithShape(end-start, inputShape), gorgonia.WithName("X"))

		// Construct forward pass and record it using tape machine.
		decoder.fwd(X)

		// Dump it, still need tape machine to activate the process.
		gorgonia.Let(X, xVal)

		vm := gorgonia.NewTapeMachine(decoder.G)
		// Activate the tape machine.
		vm.RunAll()
		vm.Reset()

		// Append the result.
		if over == 1 {
			prediction = append(prediction, Value2Float(decoder.PredVal, decoder.W1.Shape()[0])[0])
		} else {
			prediction = append(prediction, Value2Float(decoder.PredVal, decoder.W1.Shape()[0])...)
		}
	}

	if m.Normal {
		prediction_gen = Generalize(prediction, S.meanListX, S.stdListX)
	} else {
		prediction_gen = prediction
	}
	return prediction_gen
}
