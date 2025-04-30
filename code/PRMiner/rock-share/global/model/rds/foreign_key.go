package rds

import (
	comm_util "github.com/bovinae/common/util"
)

type HSOriginalTableInfo struct {
	TableId     int64
	ColumnInfos []HSColumnInfo
	Values      map[string][]interface{}
}

type HSColumnInfo struct {
	ColumnId   int64
	ColumnName string
	ColumnType string
}

type HorizontalSuggestion struct {
	LeftTableId   int64
	RightTableId  int64
	ColumnIdPairs []HSColumnIdPair
}
type HSColumnIdPair struct {
	LeftColumnId  int64
	RightColumnId int64
	Similarity    float64
}

type HorizontalSuggestionInternal struct {
	*TablePair
	HSColumnIdPair
}

type TablePair struct {
	LeftTableId  int64
	RightTableId int64
	Similarity   float64 // 两表字段相似度最大值
}

type SortColumn []any

func (a SortColumn) Len() int      { return len(a) }
func (a SortColumn) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a SortColumn) Less(i, j int) bool {
	return comm_util.CompareAny(a[i], a[j]) == comm_util.LESS
}
