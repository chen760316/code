package rds

import "rds-shenglin/rock-share/global/enum"

type ScoreRee struct {
	ScoreTaskId int64
	DataId      int64
	ColumnId    int64
	RowIds      map[int64]bool  // 错误行号
	RuleId      int64           // 规则编号
	RuleName    string          // 规则名称
	RuleContent string          // 规则内容。若是ree规则，则填写编号
	SixAttr     enum.OldSixEnum // 六性
}
