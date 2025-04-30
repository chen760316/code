package embedding_cosine

import (
	"fmt"
	"testing"
)

func TestName(t *testing.T) {
	fmt.Println(Cosine([]float64{1}, []float64{1}))
	fmt.Println(Cosine([]float64{1}, []float64{-1}))
	fmt.Println(Cosine([]float64{0, 1}, []float64{1, 0}))
}
