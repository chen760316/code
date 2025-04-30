package enum

// 标准化任务状态枚举
// TASK_CONFIG 配置中
// TASK_EXEC 进行中
// TASK_FINISH 已完成
// TASK_FAILED 执行失败

const (
	TASK_CONFIG = "TASK_CONFIG"
	TASK_EXEC   = "TASK_EXEC"
	TASK_FINISH = "TASK_FINISH"
	TASK_FAILED = "TASK_FAILED"
)

// 标准化任务字段可配置类型枚举
// CONFIG_ALL 全部字段
// CONFIG_YES 可配置字段
// CONFIG_NO 不可配置字段

const (
	CONFIG_ALL = "CONFIG_ALL"
	CONFIG_YES = "CONFIG_YES"
	CONFIG_NO  = "CONFIG_NO"
)
