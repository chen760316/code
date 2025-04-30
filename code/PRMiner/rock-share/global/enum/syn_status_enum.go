package enum

// 同步状态。
/*
SYN_EXEC 同步中
SYN_SUCCESS 同步成功
SYN_FAIL 同步失败
*/
const (
	SYN_EXEC    = "SYN_EXEC"
	SYN_SUCCESS = "SYN_SUCCESS"
	SYN_FAIL    = "SYN_FAIL"
)

// 表失败状态枚举
/*
TABLE_FAIL_SYN 表同步失败
TABLE_FAIL_EXEC 表执行失败
*/
const (
	TABLE_FAIL_SYN  = "TABLE_FAIL_SYN"
	TABLE_FAIL_EXEC = "TABLE_FAIL_EXEC"
)

// 同步状态
func SynStatus(s string) string {
	switch s {
	case SYN_EXEC:
		return SYN_EXEC
	case SYN_SUCCESS:
		return SYN_SUCCESS
	case SYN_FAIL:
		return SYN_FAIL
	default:
		return ""
	}
}

// 任务状态。
// TABLE_SYN 同步中
// TABLE_FAIL 表同步失败
// TABLE_EXEC 表执行中（包含规则发现、查错） .同步完成
// TABLE_FINISH 表执行完成.(同步完成)
// TABLE_EXEC_FAIL 表执行失败
// TABLE_KILL 表废弃

// 任务状态  转化为  同步状态
func TaskStatusToSynStatus(s string) string {
	switch s {
	case TABLE_SYN:
		return SYN_EXEC
	case TABLE_FAIL:
		return SYN_FAIL
	case TABLE_EXEC:
		return SYN_SUCCESS
	case TABLE_FINISH:
		return SYN_SUCCESS
	case TABLE_EXEC_FAIL:
		return SYN_FAIL //前端非要
	case TABLE_KILL:
		return ""
	default:
		return ""
	}
}

// 任务状态   转为   表失败状态枚举
func TaskStatusToTableFailStatus(s string) string {
	switch s {
	case TABLE_SYN:
		return ""
	case TABLE_FAIL:
		return TABLE_FAIL_SYN
	case TABLE_EXEC:
		return ""
	case TABLE_FINISH:
		return ""
	case TABLE_EXEC_FAIL:
		return TABLE_FAIL_EXEC
	case TABLE_KILL:
		return ""
	default:
		return ""
	}
}
