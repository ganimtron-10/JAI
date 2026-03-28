package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/sugarme/tokenizer/pretrained"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/gonum/blas/gonum"
)

func init() {
	blas32.Use(gonum.Implementation{})
}

type Config struct {
	HiddenSize       int     `json:"hidden_size"`
	IntermediateSize int     `json:"intermediate_size"`
	NumLayers        int     `json:"num_hidden_layers"`
	NumHeads         int     `json:"num_attention_heads"`
	NumKVHeads       int     `json:"num_key_value_heads"`
	NormEPS          float32 `json:"rms_norm_eps"`
	VocabSize        int     `json:"vocab_size"`
	MaxContext       int     `json:"max_position_embeddings"`
}

type KVCache struct {
	K [][][]float32 // [layer][position][kv_dim]
	V [][][]float32 // [layer][position][kv_dim]
}

type SamplerConfig struct {
	Temperature float32
	TopK        int
	TopP        float32
	RepPenalty  float32
}

type JAI struct {
	Config  Config
	Weights map[string][]float32
	Cache   *KVCache
	Pos     int // Tracks current position in the KV cache across the REPL
}

func MatMulVector(input []float32, weight []float32, rows, cols int, output []float32) {
	a := blas32.General{Rows: rows, Cols: cols, Stride: cols, Data: weight}
	x := blas32.Vector{N: cols, Inc: 1, Data: input}
	y := blas32.Vector{N: rows, Inc: 1, Data: output}
	blas32.Gemv(blas.NoTrans, 1.0, a, x, 0.0, y)
}

func RMSNorm(x []float32, weight []float32, eps float32) {
	var sum float32
	for _, v := range x {
		sum += v * v
	}
	invStdDev := 1.0 / float32(math.Sqrt(float64(sum/float32(len(x))+eps)))
	for i := range x {
		x[i] = (x[i] * invStdDev) * weight[i]
	}
}

func ApplyRoPE(q, k []float32, pos, headDim int) {
	half := headDim / 2
	for i := 0; i < half; i++ {
		freq := 1.0 / math.Pow(10000.0, float64(i*2)/float64(headDim))
		val := float64(pos) * freq
		cos := float32(math.Cos(val))
		sin := float32(math.Sin(val))

		if q != nil {
			q0, q1 := q[i], q[i+half]
			q[i] = q0*cos - q1*sin
			q[i+half] = q0*sin + q1*cos
		}
		if k != nil {
			k0, k1 := k[i], k[i+half]
			k[i] = k0*cos - k1*sin
			k[i+half] = k0*sin + k1*cos
		}
	}
}

func SiLU(x []float32) {
	for i, v := range x {
		x[i] = v * (1.0 / (1.0 + float32(math.Exp(float64(-v)))))
	}
}

func Sample(logits []float32, cfg SamplerConfig) int {
	if cfg.Temperature == 0.0 {
		// Greedy
		maxIdx, maxVal := 0, logits[0]
		for i, v := range logits {
			if v > maxVal {
				maxVal, maxIdx = v, i
			}
		}
		return maxIdx
	}

	// Temperature
	for i := range logits {
		logits[i] /= cfg.Temperature
	}

	var maxVal float32 = -math.MaxFloat32
	for _, v := range logits {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float32
	for i, v := range logits {
		logits[i] = float32(math.Exp(float64(v - maxVal)))
		sum += logits[i]
	}
	for i := range logits {
		logits[i] /= sum
	}

	type prob struct {
		id int
		p  float32
	}
	probs := make([]prob, len(logits))
	for i, p := range logits {
		probs[i] = prob{id: i, p: p}
	}

	// Sort descending
	sort.Slice(probs, func(i, j int) bool { return probs[i].p > probs[j].p })

	if cfg.TopK > 0 && cfg.TopK < len(probs) {
		probs = probs[:cfg.TopK]
	}

	if cfg.TopP > 0.0 && cfg.TopP < 1.0 {
		var cumSum float32
		for i, p := range probs {
			cumSum += p.p
			if cumSum > cfg.TopP {
				probs = probs[:i+1]
				break
			}
		}
	}

	// Re-normalize after truncations
	sum = 0
	for _, p := range probs {
		sum += p.p
	}
	for i := range probs {
		probs[i].p /= sum
	}

	r := rand.Float32()
	var c float32
	for _, p := range probs {
		c += p.p
		if r <= c {
			return p.id
		}
	}
	return probs[len(probs)-1].id
}

func LoadJAI(modelPath, configPath string) (*JAI, error) {
	confBuf, _ := os.ReadFile(configPath)
	var conf Config
	json.Unmarshal(confBuf, &conf)

	if conf.MaxContext == 0 {
		conf.MaxContext = 2048
	}
	if conf.IntermediateSize == 0 {
		conf.IntermediateSize = 1536
	}

	f, _ := os.Open(modelPath)
	defer f.Close()
	info, _ := f.Stat()
	data, _ := syscall.Mmap(int(f.Fd()), 0, int(info.Size()), syscall.PROT_READ, syscall.MAP_SHARED)

	headerSize := binary.LittleEndian.Uint64(data[:8])
	var headerMap map[string]interface{}
	json.Unmarshal(data[8:8+headerSize], &headerMap)

	weights := make(map[string][]float32)
	dataStart := 8 + headerSize

	fmt.Print("Loading & Converting Weights Concurrently... ")
	startT := time.Now()

	var wg sync.WaitGroup
	var mu sync.Mutex

	for name, meta := range headerMap {
		if name == "__metadata__" {
			continue
		}
		m := meta.(map[string]interface{})
		offsets := m["data_offsets"].([]interface{})
		start := uint64(offsets[0].(float64)) + dataStart
		end := uint64(offsets[1].(float64)) + dataStart

		wg.Add(1)

		go func(name string, start, end uint64) {
			defer wg.Done()
			rawData := data[start:end]
			count := len(rawData) / 2
			floatData := make([]float32, count)
			for i := 0; i < count; i++ {
				bits := binary.LittleEndian.Uint16(rawData[i*2 : i*2+2])
				floatData[i] = math.Float32frombits(uint32(bits) << 16)
			}
			mu.Lock()
			weights[name] = floatData
			mu.Unlock()
		}(name, start, end)
	}
	wg.Wait()
	fmt.Printf("Done in %v\n", time.Since(startT))

	cache := &KVCache{
		K: make([][][]float32, conf.NumLayers),
		V: make([][][]float32, conf.NumLayers),
	}
	kvDim := (conf.HiddenSize / conf.NumHeads) * conf.NumKVHeads
	for i := 0; i < conf.NumLayers; i++ {
		cache.K[i] = make([][]float32, conf.MaxContext)
		cache.V[i] = make([][]float32, conf.MaxContext)
		for j := 0; j < conf.MaxContext; j++ {
			cache.K[i][j] = make([]float32, kvDim)
			cache.V[i][j] = make([]float32, kvDim)
		}
	}

	return &JAI{Config: conf, Weights: weights, Cache: cache, Pos: 0}, nil
}

func (j *JAI) forward(tokenID int) []float32 {
	hSize := j.Config.HiddenSize
	headDim := hSize / j.Config.NumHeads
	kvDim := headDim * j.Config.NumKVHeads

	x := make([]float32, hSize)
	copy(x, j.Weights["model.embed_tokens.weight"][tokenID*hSize:(tokenID+1)*hSize])

	for l := 0; l < j.Config.NumLayers; l++ {
		residual := make([]float32, hSize)
		copy(residual, x)

		// Attention
		RMSNorm(x, j.Weights[fmt.Sprintf("model.layers.%d.input_layernorm.weight", l)], j.Config.NormEPS)

		q, k, v := make([]float32, hSize), make([]float32, kvDim), make([]float32, kvDim)
		MatMulVector(x, j.Weights[fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", l)], hSize, hSize, q)
		MatMulVector(x, j.Weights[fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", l)], kvDim, hSize, k)
		MatMulVector(x, j.Weights[fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", l)], kvDim, hSize, v)

		for h := 0; h < j.Config.NumHeads; h++ {
			ApplyRoPE(q[h*headDim:(h+1)*headDim], nil, j.Pos, headDim)
		}
		for h := 0; h < j.Config.NumKVHeads; h++ {
			ApplyRoPE(nil, k[h*headDim:(h+1)*headDim], j.Pos, headDim)
		}

		copy(j.Cache.K[l][j.Pos], k)
		copy(j.Cache.V[l][j.Pos], v)

		attnOut := make([]float32, hSize)
		scale := 1.0 / float32(math.Sqrt(float64(headDim)))

		for h := 0; h < j.Config.NumHeads; h++ {
			kvHeadIdx := h / (j.Config.NumHeads / j.Config.NumKVHeads)
			qHead := q[h*headDim : (h+1)*headDim]
			scores := make([]float32, j.Pos+1)

			for t := 0; t <= j.Pos; t++ {
				kHead := j.Cache.K[l][t][kvHeadIdx*headDim : (kvHeadIdx+1)*headDim]
				var score float32
				for i := 0; i < headDim; i++ {
					score += qHead[i] * kHead[i]
				}
				scores[t] = score * scale
			}

			var maxS float32 = -math.MaxFloat32
			for _, s := range scores {
				if s > maxS {
					maxS = s
				}
			}
			var sumS float32
			for i, s := range scores {
				scores[i] = float32(math.Exp(float64(s - maxS)))
				sumS += scores[i]
			}
			for i := range scores {
				scores[i] /= sumS
			}

			outHead := attnOut[h*headDim : (h+1)*headDim]
			for t := 0; t <= j.Pos; t++ {
				vHead := j.Cache.V[l][t][kvHeadIdx*headDim : (kvHeadIdx+1)*headDim]
				for i := 0; i < headDim; i++ {
					outHead[i] += scores[t] * vHead[i]
				}
			}
		}

		projOut := make([]float32, hSize)
		MatMulVector(attnOut, j.Weights[fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", l)], hSize, hSize, projOut)
		for i := range x {
			x[i] = residual[i] + projOut[i]
		}

		// MLP (SwiGLU)
		copy(residual, x)
		RMSNorm(x, j.Weights[fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", l)], j.Config.NormEPS)

		gate, up := make([]float32, j.Config.IntermediateSize), make([]float32, j.Config.IntermediateSize)
		MatMulVector(x, j.Weights[fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", l)], j.Config.IntermediateSize, hSize, gate)
		MatMulVector(x, j.Weights[fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", l)], j.Config.IntermediateSize, hSize, up)

		SiLU(gate)
		for i := range gate {
			gate[i] *= up[i]
		}

		mlpOut := make([]float32, hSize)
		MatMulVector(gate, j.Weights[fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", l)], hSize, j.Config.IntermediateSize, mlpOut)
		for i := range x {
			x[i] = residual[i] + mlpOut[i]
		}
	}

	RMSNorm(x, j.Weights["model.norm.weight"], j.Config.NormEPS)
	logits := make([]float32, j.Config.VocabSize)

	headWeight, ok := j.Weights["lm_head.weight"]
	if !ok {
		headWeight = j.Weights["model.embed_tokens.weight"]
	}

	MatMulVector(x, headWeight, j.Config.VocabSize, hSize, logits)
	return logits
}

func main() {
	modelDir := "models/smollm2-135m-instruct"
	jai, err := LoadJAI(modelDir+"/model.safetensors", modelDir+"/config.json")
	if err != nil {
		panic(err)
	}

	tk, _ := pretrained.FromFile(modelDir + "/tokenizer.json")

	sampler := SamplerConfig{
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.9,
		RepPenalty:  1.15,
	}

	reader := bufio.NewReader(os.Stdin)
	fmt.Println("========================================")
	fmt.Println(" JAI (Just another AI Inference Engine) ")
	fmt.Println(" Type 'quit' to exit.")
	fmt.Println("========================================")

	contextHistory := make([]int, 0)

	for {
		fmt.Print("\nUser: ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if input == "quit" || input == "exit" {
			break
		}

		// Wrap users input as using instruct model
		prompt := fmt.Sprintf("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", input)
		en, _ := tk.EncodeSingle(prompt)
		contextHistory = append(contextHistory, en.Ids...)

		fmt.Print("JAI: ")
		var logits []float32

		// Prefill Phase
		for i, id := range en.Ids {
			logits = jai.forward(id)
			if i < len(en.Ids)-1 {
				jai.Pos++
			}
		}

		// Decode Phase
		for i := 0; i < 200; i++ {
			if jai.Pos >= jai.Config.MaxContext-1 {
				fmt.Print("\n[Context Limit Reached]")
				break
			}

			if sampler.RepPenalty > 1.0 {
				for _, ctxID := range contextHistory {
					if logits[ctxID] > 0 {
						logits[ctxID] /= sampler.RepPenalty
					} else {
						logits[ctxID] *= sampler.RepPenalty
					}
				}
			}

			nextToken := Sample(logits, sampler)

			// SmolLM-specific EOS tokens (0: <|endoftext|>, 2: <|im_end|>)
			if nextToken == 0 || nextToken == 2 {
				break
			}

			word := tk.Decode([]int{nextToken}, true)
			fmt.Print(word)

			contextHistory = append(contextHistory, nextToken)
			jai.Pos++
			logits = jai.forward(nextToken)
		}

		jai.Pos++
	}
}
