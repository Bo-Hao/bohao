package bohao

import (
	"fmt"
	"log"
	"strconv"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type VAE struct{
	G *gorgonia.ExprGraph
	sampleNum int
	Encoding_W []*gorgonia.Node
	Decoding_W []*gorgonia.Node

	EncodeDropout []float64
	DecodeDropout []float64

	EncodeAct []ActivationFunc
	DecodeAct []ActivationFunc

	estMean   *gorgonia.Node // mean
	estSd     *gorgonia.Node // standard deviation stored in log scale
	floatHalf *gorgonia.Node
	epsilon   *gorgonia.Node

	out     *gorgonia.Node
	outMean *gorgonia.Node
	outVar  *gorgonia.Node
	predVal gorgonia.Value
	Pred *gorgonia.Node
}


type VAEStruct struct {
	EncodeNeuron []int
	DecodeNeuron []int

	EncodeDropout []float64
	DecodeDropout []float64

	EncodeAct []ActivationFunc
	DecodeAct []ActivationFunc
}

func NewVAE(g *gorgonia.ExprGraph, S VAEStruct) (*VAE){
	var wEs gorgonia.Nodes
	for i :=0; i < len(S.EncodeNeuron) - 1; i ++{
		name := "w" + "E" + strconv.Itoa(i)
		wE := gorgonia.NewMatrix(
			g, 
			tensor.Float64, 
			gorgonia.WithShape(S.EncodeNeuron[i], S.EncodeNeuron[i + 1]), 
			gorgonia.WithName(name), 
			gorgonia.WithInit(gorgonia.GlorotU(1.0)), 
		)
		wEs = append(wEs, wE)
	}
	var wDs gorgonia.Nodes
	for i :=0; i < len(S.DecodeNeuron) - 1; i ++{
		name := "w" + "D" + strconv.Itoa(i)
		wD := gorgonia.NewMatrix(
			g, 
			tensor.Float64, 
			gorgonia.WithShape(S.DecodeNeuron[i], S.DecodeNeuron[i + 1]), 
			gorgonia.WithName(name), 
			gorgonia.WithInit(gorgonia.GlorotU(1.0)), 
		)
		wDs = append(wDs, wD)
	}
	lastindx := len(S.EncodeNeuron) - 1
	estMean := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(S.EncodeNeuron[lastindx], S.DecodeNeuron[0]), gorgonia.WithName("estMean"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	estSd := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(S.EncodeNeuron[lastindx], S.DecodeNeuron[0]), gorgonia.WithName("estSd"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))

	floatHalf := gorgonia.NewScalar(g, tensor.Float64, gorgonia.WithName("floatHalf"))
	gorgonia.Let(floatHalf, 0.5)

	

	return &VAE{
		G:  g,
		Encoding_W: wEs,
		Decoding_W: wDs,

		EncodeDropout: S.EncodeDropout,
		DecodeDropout: S.DecodeDropout,

		EncodeAct: S.EncodeAct,
		DecodeAct: S.DecodeAct,
		
		estMean:   estMean,
		estSd:     estSd,
		floatHalf: floatHalf,
		sampleNum : S.DecodeNeuron[0],
	}
}

func (m *VAE) Learnables() gorgonia.Nodes {
	n := gorgonia.NewNodeSet()
	for i := 0; i < len(m.Encoding_W); i ++{
		n.Add(m.Encoding_W[i])
	}
	for i := 0; i < len(m.Decoding_W); i ++{
		n.Add(m.Decoding_W[i])
	}
	n.Add(m.estMean)
	n.Add(m.estSd)
	return n.ToSlice().Nodes()
}

func (m *VAE) Forward(x *gorgonia.Node) (err error) {
	var sz *gorgonia.Node
	l := make([]*gorgonia.Node, len(m.Encoding_W)+len(m.Decoding_W) + 2)
	c := make([]*gorgonia.Node, len(m.Encoding_W)+len(m.Decoding_W))
	d := make([]*gorgonia.Node, len(m.Encoding_W)+len(m.Decoding_W))
	l[0] = x
	num := 0
	// Part 1 Encoding
	for i := 0; i < len(m.Encoding_W); i ++{
		if c[i], err = gorgonia.Mul(l[i], m.Encoding_W[i]); err != nil {
			log.Fatal(err, "Layer Convolution failed")
		}
		if d[i], err = m.EncodeAct[i](c[i]); err != nil {
			log.Fatal(err, "Layer activation failed")
		}
		if l[i + 1], err = gorgonia.Dropout(d[i], m.EncodeDropout[i]); err != nil {
			log.Fatal(err, "Layer Dropout failed")
		}
		num = i
	}

	// Part 2 Sampling
	outMean, err := gorgonia.Mul(l[num+1], m.estMean)
	if err != nil {
		log.Fatal(err, "outMean Multiplication failed")
	}
	outVar, err := gorgonia.HadamardProd(m.floatHalf, gorgonia.Must(gorgonia.Mul(l[num+1], m.estSd)))
	if err != nil {
		log.Fatal(err, "OutVar Multiplication failed")
	}

	if sz, err = gorgonia.Add(outMean, gorgonia.Must(gorgonia.HadamardProd(gorgonia.Must(gorgonia.Exp(outVar)), m.epsilon))); err != nil {
		log.Fatal(err, "Layer Sampling failed")
	}

	// Part 3 Decoding
	l[num + 2] = sz
	for i := 0; i < len(m.Decoding_W); i ++{
		if c[num + 1 + i], err = gorgonia.Mul(l[num + 2 + i], m.Decoding_W[i]); err != nil {
			log.Fatal(err, "Layer Convolution failed")
		}
		if d[num + 1 + i], err = m.DecodeAct[i](c[num + 1 + i]); err != nil {
			log.Fatal(err, "Layer activation failed")
		}
		if l[num + 3 + i], err = gorgonia.Dropout(d[num + 1 + i], m.DecodeDropout[i]); err != nil {
			log.Fatal(err, "Layer Dropout failed")
		}
	}
	lastindx := len(l) - 1
	m.out = l[lastindx]
	m.outMean = outMean
	m.outVar = outVar
	m.Pred = l[lastindx]
	gorgonia.Read(l[lastindx], &m.predVal)
	return
}

func (m *VAE) Fit(input_x, input_y [][]float64, para TrainingParameter) {
	S := Stock{}
	inputShape := len(input_x[0])
	outputShape := len(input_y[0])

	x_oneDim := ToOneDimSlice(input_x)
	y_oneDim := ToOneDimSlice(input_y)
	sampleSize := len(input_x)
	m.epsilon = gorgonia.GaussianRandomNode(m.G, tensor.Float64, 0, 1, sampleSize, m.sampleNum)

	if sampleSize != len(input_y) {
		log.Fatal("x and y are not in the same size!")
	}

	xT := tensor.New(tensor.WithBacking(x_oneDim), tensor.WithShape(sampleSize, inputShape))
	yT := tensor.New(tensor.WithBacking(y_oneDim), tensor.WithShape(sampleSize, outputShape))

	X := gorgonia.NewMatrix(m.G, tensor.Float64, gorgonia.WithShape(sampleSize, inputShape), gorgonia.WithName("X"))
	y := gorgonia.NewMatrix(m.G, tensor.Float64, gorgonia.WithShape(sampleSize, outputShape), gorgonia.WithName("y"))

	err := m.Forward(X)
	if err != nil {
		log.Fatal(err)
	}

	valueOne := gorgonia.NewScalar(m.G, tensor.Float64, gorgonia.WithName("valueOne"))
	valueTwo := gorgonia.NewScalar(m.G, tensor.Float64, gorgonia.WithName("valueTwo"))
	gorgonia.Let(valueOne, 1.0)
	gorgonia.Let(valueTwo, 2.0)

	klLoss, err := gorgonia.Div(
		gorgonia.Must(gorgonia.Sum(
			gorgonia.Must(gorgonia.Sub(
				gorgonia.Must(gorgonia.Add(
					gorgonia.Must(gorgonia.Square(m.outMean)),
					gorgonia.Must(gorgonia.Exp(m.outVar)))),
				gorgonia.Must(gorgonia.Add(m.outVar, valueOne)))))),
		valueTwo)
	if err != nil {
		log.Fatal(err)
	}

	valueLoss, err := gorgonia.Sum(gorgonia.Must(gorgonia.Square(gorgonia.Must(gorgonia.Sub(y, m.out)))))
	if err != nil {
		log.Fatal(err)
	}

	vaeCost := gorgonia.Must(gorgonia.Add(klLoss, valueLoss))

	var costVal gorgonia.Value
	gorgonia.Read(vaeCost, &costVal)

	if _, err = gorgonia.Grad(vaeCost, m.Learnables()...); err != nil {
		log.Fatal("Unable to upgrade gradient")
	}

	vm := gorgonia.NewTapeMachine(m.G, gorgonia.BindDualValues(m.Learnables()...))
	solver := gorgonia.NewAdaGradSolver(gorgonia.WithLearnRate(para.Lr))

	for epoch := 0; epoch < para.Epoches; epoch++ {
		gorgonia.Let(X, xT)
		gorgonia.Let(y, yT)

		vm.RunAll()
		
		solver.Step(gorgonia.NodesToValueGrads(m.Learnables()))
		vm.Reset()
		if epoch%10 == 0 {
			fmt.Println("Iteration: ", epoch, ". Cost: ", costVal)
		}
		S.LossRecord = append(S.LossRecord, []float64{float64(epoch), costVal.Data().(float64)})
	}
}

func (m *VAE) Predict(x [][]float64) {
	inputShape := m.Encoding_W[0].Shape()[0]

	/* input_x, _, _ := Normalized(x) */
	input_x := x

	x_oneDim := ToOneDimSlice(input_x)
	sampleSize := len(x)

	xT := tensor.New(tensor.WithBacking(x_oneDim), tensor.WithShape(sampleSize, inputShape))
	X := gorgonia.NewMatrix(m.G, tensor.Float64, gorgonia.WithShape(sampleSize, inputShape), gorgonia.WithName("X"))

	err := m.Forward(X)
	if err != nil {
		log.Fatal(err)
	}

	gorgonia.Let(X, xT)
	vm := gorgonia.NewTapeMachine(m.G, gorgonia.BindDualValues(m.Learnables()...))
	vm.RunAll()
	vm.Reset()
}

// Separate the decoding part
func (m *VAE) DecodingNet() (NN){
	return NN{
		G: m.G,
		W: m.Decoding_W,
		D: m.DecodeDropout,
		A: m.DecodeAct,
	}
} 


func Do_VAE() {
	// fake data 
	f_x := [][]float64{[]float64{0.1, 0.1, 0.1, 0.1}, []float64{0.2, 0.2, 0.2, 0.2}, []float64{0.3, 0.3, 0.3, 0.3}, []float64{0.4, 0.4, 0.4, 0.4}, []float64{0.5, 0.5, 0.5, 0.5}}
	
	g := gorgonia.NewGraph()

	// struction 
	S := VAEStruct{
		EncodeNeuron: []int{4, 3, 2},
		DecodeNeuron: []int{2, 3, 4},

		EncodeDropout: []float64{0, 0, 0},
		DecodeDropout: []float64{0, 0, 0},

		EncodeAct: []ActivationFunc{Linear, Linear, Linear},
		DecodeAct: []ActivationFunc{Linear, Linear, Linear},
	}

	m := NewVAE(g, S)
	para := InitParameter()
	para.Lr = 0.1

	// training for VAE
	m.Fit(f_x, f_x, para)

	// predicting 
	m.Predict(f_x)
	fmt.Println(m.predVal)

	// tear decoder apart
	md := m.DecodingNet()
	fmt.Println(md.W[0].Value())

	// random generate input for creating
	md.Predict([][]float64{[]float64{0.3, 0.7}, []float64{0, 1}, []float64{0.1, 0.9}})
	fmt.Println(md.PredVal)



}
