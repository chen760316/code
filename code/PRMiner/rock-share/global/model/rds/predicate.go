package rds

import (
	"strings"
)

type Column struct {
	ColumnIndex int
	TableId     string
	ColumnId    string
	ColumnType  string
	IsML        bool
	JoinTableId string // 多表join生成大宽表使用
	RoleName    string `json:"RoleName"` // 数据角色名称
}

type Predicate struct {
	PredicateStr              string
	LeftColumn                Column
	RightColumn               Column
	ConstantValue             interface{}
	ConstantIndexValue        int32
	SymbolType                string
	PredicateType             int    // 常数谓词是0,非常数谓词是1
	UDFName                   string // jaccard sentence-bert...
	Threshold                 float64
	Support                   float64
	Intersection              [][2][]int32 `json:"-"` // [按值分组][t0或t1][行号]
	UdfIndex                  int
	LeftColumnVectorFilePath  string
	RightColumnVectorFilePath string
}

// ConvertColumnID 列名转换 /
func (predicate Predicate) ConvertColumnID(columnMap *map[string]string) string {
	columnValueMap := *columnMap
	// 左侧
	leftColumnId := predicate.LeftColumn.ColumnId
	replace := strings.Replace(predicate.PredicateStr, leftColumnId, columnValueMap[leftColumnId], 1)
	if predicate.PredicateType == 0 {
		return replace
	}

	rightColumnId := predicate.RightColumn.ColumnId
	replace = strings.Replace(replace, rightColumnId, columnValueMap[rightColumnId], 1)

	return replace
}
