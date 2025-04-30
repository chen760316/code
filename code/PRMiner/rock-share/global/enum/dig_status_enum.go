package enum

import "rds-shenglin/rock-share/base/logger"

/*
digRulesStatus规则挖掘状态：
DIG_EXEC 挖掘中
DIG_FINISH 挖掘完成
DIG_FAIL 挖掘失败
*/

const (
	DIG_EXEC   = "DIG_EXEC"
	DIG_FINISH = "DIG_FINISH"
	DIG_FAIL   = "DIG_FAIL"
)

//  任务状态  转化为  挖掘状态
// TABLE_SYN 同步中。-->    挖掘中 DIG_EXEC
// TABLE_FAIL 表同步失败 。--> 挖掘失败
// TABLE_EXEC 表执行中（包含规则发现、查错） --> 挖掘完成
// TABLE_FINISH 表执行完成.--> 挖掘完成
// TABLE_KILL 表废弃

// 任务状态  转化为  挖掘状态
func TaskStatusToDigStatus(s string) string {
	switch s {
	case TABLE_SYN:
		return DIG_EXEC
	case TABLE_FAIL:
		return DIG_FAIL
	case TABLE_EXEC:
		return DIG_EXEC
	case TABLE_FINISH:
		return DIG_FINISH
	case TABLE_KILL:
		return DIG_FAIL
	default:
		logger.Errorf("UNKNOWN enum:%s", s)
		return "UNKNOWN"
	}
}
