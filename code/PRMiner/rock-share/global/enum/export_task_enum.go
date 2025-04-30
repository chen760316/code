package enum

// 导出任务状态
const (
	EXPORT_TODO     = "TODO" //该状态表示新的导出任务正在队列中等待
	EXPORT_RUNNING  = "RUNNING"
	EXPORT_FINISHED = "FINISHED"
	EXPORT_FAILED   = "FAILED"
	EXPORT_STOPPED  = "STOPPED"
	EXPORT_QUIT     = "QUIT" //该状态表示因为退出当前工程导致正在执行的任务停止
)

// 导出任务详情状态
const (
	EXPORT_DETAIL_TODO     = "TODO"
	EXPORT_DETAIL_RUNNING  = "RUNNING"
	EXPORT_DETAIL_FINISHED = "FINISHED"
	EXPORT_DETAIL_FAILED   = "FAILED"
)
