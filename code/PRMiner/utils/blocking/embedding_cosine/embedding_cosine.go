package embedding_cosine

import "math"

func Cosine(left, right []float64) float64 {
	return math.Abs(dot(left, right)) / (length(left) * length(right))
}

func dot(a, b []float64) float64 {
	var s float64 = 0
	for i, f := range a {
		s += f * b[i]
	}
	return s
}

func length(a []float64) float64 {
	return math.Sqrt(dot(a, a))
}
