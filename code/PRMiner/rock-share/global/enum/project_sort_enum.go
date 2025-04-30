package enum

func ProjectSortString(s string) string {

	switch s {
	case "name":
		return "name"
	case "createTime":
		return "create_time"
	case "updateTime":
		return "update_time"
	case "status":
		return "status"
	default:
		return "update_time"
	}
}
