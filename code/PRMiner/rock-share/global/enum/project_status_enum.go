package enum

/*
工程状态枚举

PROJECT_DESIGN 配置中。
PROJECT_SYN 加载中。（同步中）
PROJECT_EXEC 处理中。（执行中）
PROJECT_FINISH 处理结束
PROJECT_FAIL 加载失败
*/
const (
	PROJECT_DESIGN = "PROJECT_DESIGN"
	PROJECT_SYN    = "PROJECT_SYN"
	PROJECT_EXEC   = "PROJECT_EXEC"
	PROJECT_FINISH = "PROJECT_FINISH"
	PROJECT_FAIL   = "PROJECT_FAIL"
)

func ProjectStatus(p string) string {
	switch p {
	case PROJECT_DESIGN:
		return PROJECT_DESIGN
	case PROJECT_SYN:
		return PROJECT_SYN
	case PROJECT_EXEC:
		return PROJECT_EXEC
	case PROJECT_FINISH:
		return PROJECT_FINISH
	case PROJECT_FAIL:
		return PROJECT_FAIL
	default:
		return ""
	}
}
