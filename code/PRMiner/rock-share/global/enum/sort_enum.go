package enum

import "strings"

//type Sort string

const (
	DESC = "DESC"
	ASC  = "ASC"
)

func SortString(p string) string {
	s := strings.ToUpper(p)
	switch s {
	case DESC:
		return DESC
	default:
		return ASC
	}
}
