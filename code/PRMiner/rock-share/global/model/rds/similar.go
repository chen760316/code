package rds

type SimilarPair struct {
	LeftTableId   string
	LeftColumnId  string
	RightTableId  string
	RightColumnId string
	SimilarType   string
	Threshold     float64
}
