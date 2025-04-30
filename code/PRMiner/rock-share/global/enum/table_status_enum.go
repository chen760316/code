package enum

// 任务状态枚举
// TABLE_SYN 同步中
// TABLE_FAIL 表同步失败
// TABLE_EXEC 表执行中（包含规则发现、查错） .已经同步完成
// TABLE_FINISH 表执行成功.(已经同步完成)
// TABLE_EXEC_FAIL 表执行失败.
// TABLE_KILL 表废弃
// TABLE_STOPPED 退出工程的话 停止任务
const (
	TABLE_SYN       = "TABLE_SYN"
	TABLE_FAIL      = "TABLE_FAIL"
	TABLE_TODO      = "TABLE_TODO"
	TABLE_EXEC      = "TABLE_EXEC"
	TABLE_FINISH    = "TABLE_FINISH"
	TABLE_EXEC_FAIL = "TABLE_EXEC_FAIL"
	TABLE_KILL      = "TABLE_KILL"
	TABLE_STOPPED   = "TABLE_STOPPED"
)
