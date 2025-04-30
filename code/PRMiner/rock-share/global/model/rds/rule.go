package rds

type Rule struct {
	RuleId        int
	TableId       string
	Ree           string
	LhsPredicates []Predicate
	LhsColumns    []Column
	Rhs           Predicate
	RhsColumn     Column
	CR            float64
	FTR           float64
	RuleType      int // 0单行;1多行;2正则;3多项式;4ER;5最优/时序
	XSupp         int
	XySupp        int
	XSatisfyCount int
	XSatisfyRows  []int32      `json:"-"`
	XIntersection [][2][]int32 `json:"-"` // [按值分组][t0或t1][行号]
	CfdStatus     int          // -1降低;0不变;1提升
	IsUserDefine  bool         // 标志是否是用户自定义规则
}
