package enum

const (
	RULE_UNUSED  = 0
	RULE_IGNORED = 1
	RULE_USED    = 2
	RULE_HIDE    = 3 // 修改单元格可能会使一些规则不被展示,但是回退又要展示出来,所以加一个规则的状态
)
