package bohao 

import(
	"fmt"
	"log"
	"strconv"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type DualNet struct{
	G *gorgonia.ExprGraph
	W []gorgonia.Nodes
	D [][]float64
	A [][]ActivationFunc

	S DualStruct

	Pred    *gorgonia.Node
	PredVal gorgonia.Value

	FitStock Stock
}

type DualStruct struct{
	Neuron [][]int
	Dropout [][]float64
	Act [][]ActivationFunc
}

func (m *DualNet) Learnables() gorgonia.Nodes{
	n := gorgonia.NewNodeSet()
	for i := 0; i < len(m.W); i++ {
		for j := 0; j < len(m.W[i]); j ++{
			n.Add(m.W[i][j])
		}	
	}
	return n.ToSlice().Nodes()
}

func NewDualNet(g *gorgonia.ExprGraph,S DualStruct) (*DualNet){
	var List_Ns []gorgonia.Nodes
	for i := 0; i < len(S.Neuron)-1; i++ {
		var Ns gorgonia.Nodes
		frontLen := len(S.Neuron[i])
		backLen := len(S.Neuron[i + 1])

		if frontLen == backLen {
			for j := 0; j < frontLen; j ++{
				Ns = append(Ns, gorgonia.NewMatrix(
					g,
					tensor.Float64,
					gorgonia.WithShape(S.Neuron[i][j], S.Neuron[i+1][j]),
					gorgonia.WithName("w"+strconv.Itoa(i)+strconv.Itoa(j)),
					gorgonia.WithInit(gorgonia.GlorotU(1)),
				))
			}
		} else if frontLen == backLen + 1 {
			for j := 0; j < 2; j ++{
				Ns = append(Ns, gorgonia.NewMatrix(
					g,
					tensor.Float64,
					gorgonia.WithShape(S.Neuron[i][j], S.Neuron[i+1][0]),
					gorgonia.WithName("w"+strconv.Itoa(i)+strconv.Itoa(j)),
					gorgonia.WithInit(gorgonia.GlorotU(1)),
				))
			}

		} else if frontLen + 1 == backLen {
			for j := 0; j < 2; j ++{
				Ns = append(Ns, gorgonia.NewMatrix(
					g,
					tensor.Float64,
					gorgonia.WithShape(S.Neuron[i][0], S.Neuron[i+1][j]),
					gorgonia.WithName("w"+strconv.Itoa(i)+strconv.Itoa(j)),
					gorgonia.WithInit(gorgonia.GlorotU(1)),
				))
			}
		}
		List_Ns = append(List_Ns, Ns)
	}
	
	return &DualNet{
		G: g,
		W: List_Ns,
		D: S.Dropout,
		A: S.Act,
		S: S,
	}
}

func (m *DualNet) Forward(x *gorgonia.Node)(err error) {
	var l, ldot, p [][]*gorgonia.Node
	l = append(l, []*gorgonia.Node{x})

	for i := 0; i < len(m.W); i ++{
		tmp1 := make([]*gorgonia.Node, len(m.W[i]))
		tmp2 := make([]*gorgonia.Node, len(m.W[i]))
		tmp3 := make([]*gorgonia.Node, len(m.W[i]))
		l = append(l, tmp1)
		ldot = append(ldot, tmp2)
		p = append(p, tmp3)
	}

	for i := 0; i < len(l) - 1; i++ {
		if len(l[i]) == len(l[i + 1]) {
			for j := 0; j < len(l[i]); j ++{
				ldot[i][j] = gorgonia.Must(gorgonia.Mul(l[i][j], m.W[i][j]))
				p[i][j], err = gorgonia.Dropout(ldot[i][j], m.D[i][j])
				if err != nil {
					log.Printf("Can't drop!")
				}

				//activation function
				l[i+1][j] = gorgonia.Must(m.A[i][j](p[i][j]))
				}
		}else if len(l[i]) == len(l[i + 1]) + 1 {
			// 1
			ldot[i][0] = gorgonia.Must(gorgonia.Mul(l[i][0], m.W[i][0]))
			p[i][0], err = gorgonia.Dropout(ldot[i][0], m.D[i][0])
			if err != nil {
				log.Printf("Can't drop!")
			}

			//activation function
			r1 := gorgonia.Must(m.A[i][0](p[i][0]))
				
			// 2
			ldot[i][1] = gorgonia.Must(gorgonia.Mul(l[i][1], m.W[i][1]))
			p[i][1], err = gorgonia.Dropout(ldot[i][1], m.D[i][1])
			if err != nil {
				log.Printf("Can't drop!")
			}

			//activation function
			r2 := gorgonia.Must(m.A[i][1](p[i][1]))

			// merge
			l[i+1][0] = gorgonia.Must(gorgonia.Add(r1, r2))
			
		}else if len(l[i]) + 1 == len(l[i + 1]) {
			
			// 1
			ldot[i][0] = gorgonia.Must(gorgonia.Mul(l[i][0], m.W[i][0]))
			p[i][0], err = gorgonia.Dropout(ldot[i][0], m.D[i][0])
			if err != nil {
				log.Printf("Can't drop!")
			}

			//activation function
			l[i + 1][0] = gorgonia.Must(m.A[i][0](p[i][0]))
				
			// 2
			ldot[i][1] = gorgonia.Must(gorgonia.Mul(l[i][0], m.W[i][1]))
			p[i][1], err = gorgonia.Dropout(ldot[i][1], m.D[i][1])
			if err != nil {
				log.Printf("Can't drop!")
			}

			//activation function
			l[i + 1][1] = gorgonia.Must(m.A[i][1](p[i][1]))
		}
	}
	fmt.Println(l)
	fmt.Println(ldot)
	fmt.Println(p)
	m.Pred = l[len(l)-1][0]
	gorgonia.Read(m.Pred, &m.PredVal)
	return
}

func (m DualNet) ValueToFloatSlice() (result [][]float64) {
	oneDimSlice := m.PredVal.Data().([]float64)
	outputShape := m.W[len(m.W)-1][0].Shape()[1]

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

func (m *DualNet) Predict(x [][]float64) {
	inputShape := m.W[0][0].Shape()[0]

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

func (m *DualNet) Fit(input_x, input_y [][]float64, para Parameter) {
	S := Stock{}
	inputShape := m.W[0][0].Shape()[0]
	outputShape := m.W[len(m.W)-1][0].Shape()[1]

	x_oneDim := ToOneDimSlice(input_x)
	y_oneDim := ToOneDimSlice(input_y)
	sampleSize := len(input_x)

	if sampleSize != len(input_y) {
		log.Fatal("x and y are not in the same size!")
	}
	xT := tensor.New(tensor.WithBacking(x_oneDim), tensor.WithShape(sampleSize, inputShape))
	yT := tensor.New(tensor.WithBacking(y_oneDim), tensor.WithShape(sampleSize, outputShape))

	X := gorgonia.NewMatrix(m.G, tensor.Float64, gorgonia.WithShape(sampleSize, inputShape), gorgonia.WithName("X"))
	y := gorgonia.NewMatrix(m.G, tensor.Float64, gorgonia.WithShape(sampleSize, outputShape), gorgonia.WithName("y"))

	m.Forward(X)
	cost := para.Lossfunc(m.Pred, y)

	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)

	if _, err = gorgonia.Grad(cost, m.Learnables()...); err != nil {
		fmt.Println(err)
		log.Fatal("Unable to upgrade gradient")
	}

	vm := gorgonia.NewTapeMachine(m.G, gorgonia.BindDualValues(m.Learnables()...))
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithLearnRate(para.Lr))
	

	for epoch := 0; epoch < para.Epoches; epoch++ {
		gorgonia.Let(X, xT)
		gorgonia.Let(y, yT)

		vm.RunAll()
		solver.Step(gorgonia.NodesToValueGrads(m.Learnables()))
		vm.Reset()
		if epoch%10 == 0 {
			//fmt.Println("Iteration: ", epoch, ". Cost: ", costVal)
		}
		S.LossRecord = append(S.LossRecord, []float64{float64(epoch), costVal.Data().(float64)})
	}
	m.FitStock.LossRecord = S.LossRecord
}

func Test_DualNet() {
	x := [][]float64{[]float64{1, 2, 3, 1}, []float64{4, 5, 6, 1}, []float64{4, 5, 6, 1}, []float64{4, 5, 6, 1}, []float64{4, 5, 6, 1}}
	y := [][]float64{[]float64{1, 2}, []float64{5, 6}, []float64{5, 6}, []float64{5, 6}, []float64{5, 6}}

	g := gorgonia.NewGraph()
	S := DualStruct{
		Neuron: [][]int{[]int{4}, []int{5, 3}, []int{2}},
		Dropout: [][]float64{[]float64{0, 0},  []float64{0, 0}}, 
		Act: [][]ActivationFunc{[]ActivationFunc{Linear, Linear}, []ActivationFunc{Linear, Linear}, []ActivationFunc{Linear} }, 
	}

	m := NewDualNet(g, S)
	// set test data into NN
	m.Predict(x)

	// show predition
	fmt.Println(m.PredVal)
	
	// init training parameter
	para := InitParameter()

	// fit training data
	m.Fit(x, y, para)
	
	// set test data into NN
	m.Predict(x)

	// show predition
	fmt.Println(m.PredVal)

}

func Merge_RMSError(Pred1, Pred2, y *gorgonia.Node) *gorgonia.Node {
	Pred := gorgonia.Must(gorgonia.Add(Pred1, Pred2))
	losses := gorgonia.Must(gorgonia.Square(gorgonia.Must(gorgonia.Sub(Pred, y))))
	cost := gorgonia.Must(gorgonia.Mean(losses))
	return cost
}
func Test_merge_NN() {

	x := [][]float64{[]float64{1, 2, 3, 1}, []float64{4, 5, 6, 1}, []float64{4, 5, 6, 1}, []float64{4, 5, 6, 1}, []float64{4, 5, 6, 1}}
	y := [][]float64{[]float64{1, 2}, []float64{5, 6}, []float64{5, 6}, []float64{5, 6}, []float64{5, 6}}

	x_oneDim := ToOneDimSlice(x)
	y_oneDim := ToOneDimSlice(y)
	sampleSize := len(x)
	xT := tensor.New(tensor.WithBacking(x_oneDim), tensor.WithShape(sampleSize, 4))
	yT := tensor.New(tensor.WithBacking(y_oneDim), tensor.WithShape(sampleSize, 2))


	g := gorgonia.NewGraph()

	S1 := NetworkStruction{
		Neuron: []int{4, 3, 2},
		Dropout: []float64{0, 0}, 
		Act: []ActivationFunc{Linear, Linear, Linear},
	}
	m1 := NewNN(g, S1)

	S2 := NetworkStruction{
		Neuron: []int{4, 2, 2},
		Dropout: []float64{0, 0}, 
		Act: []ActivationFunc{Linear, Linear, Linear},
	}
	m2 := NewNN(g, S2)

	
	X := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(5, 4), gorgonia.WithName("X"))
	Y := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(5, 2), gorgonia.WithName("y"))

	m1.Forward(X)
	m2.Forward(X)

	cost1 := Merge_RMSError(m1.Pred, m2.Pred, Y)
	cost2 := Merge_RMSError(m1.Pred, m2.Pred, Y)
	
	var costVal1 gorgonia.Value
	gorgonia.Read(cost1, &costVal1)

	var costVal2 gorgonia.Value
	gorgonia.Read(cost2, &costVal2)

	if _, err = gorgonia.Grad(cost1, m1.Learnables()...); err != nil {
		fmt.Println(err, "1")
		log.Fatal("Unable to upgrade gradient")
	}

	if _, err = gorgonia.Grad(cost2, m2.Learnables()...); err != nil {
		fmt.Println(err, "2")
		log.Fatal("Unable to upgrade gradient")
	}

	vm1 := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m1.Learnables()...))
	vm2 := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m2.Learnables()...))
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithLearnRate(0.01))

	for epoch := 0; epoch < 200; epoch++ {
		gorgonia.Let(X, xT)
		gorgonia.Let(Y, yT)

		vm1.RunAll()
		solver.Step(gorgonia.NodesToValueGrads(m1.Learnables()))
		vm1.Reset()

		vm2.RunAll()
		solver.Step(gorgonia.NodesToValueGrads(m2.Learnables()))
		vm2.Reset()
		if epoch%10 == 0 {
			fmt.Println("Iteration: ", epoch, ". Cost: ", costVal1)
		}
		
	}

}