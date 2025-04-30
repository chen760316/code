package enum

type RuleType int

const (
	// SingleTableSingleRow 单表单行规则
	SingleTableSingleRow RuleType = iota
	// SingleTableMultiRow 单表多行规则
	SingleTableMultiRow
	// SingleTableMultiRowWithConst 单表多行规则加常数
	SingleTableMultiRowWithConst
	// SingleTableMultiRowWithML 单表多行规则加ML
	SingleTableMultiRowWithML
	// SingleTableMultiRowWithSimilar 单表多行规则加similar
	SingleTableMultiRowWithSimilar
	// MultiTableSingleRow 多表单行规则
	MultiTableSingleRow
	// MultiTableMultiRow 多表多行规则
	MultiTableMultiRow
	// MultiTableMultiRowWithConst 多表多行规则加常数
	MultiTableMultiRowWithConst
	// MultiTableMultiRowWithML 多表多行规则加ML
	MultiTableMultiRowWithML
	// MultiTableMultiRowWithSimilar 多表多行规则加similar
	MultiTableMultiRowWithSimilar
	// MultiTableMultiRowMultiY 多表多行规则Y跨列
	MultiTableMultiRowMultiY
)

const (
	DISCOVER  = "DISCOVER"
	CUSTOMIZE = "CUSTOMIZE"
)
