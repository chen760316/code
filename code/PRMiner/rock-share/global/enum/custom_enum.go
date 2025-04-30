package enum

import (
	"unicode/utf8"

	"rds-shenglin/rock-share/base/logger"
)

// 用户输入的字符，最大支持字符长度
const MaxStringLength = 128

// 是否超出最大支持字符个数
func IsExceedMaxStringLength(s string) bool {
	length := utf8.RuneCountInString(s)
	if length > MaxStringLength {
		logger.Warn(s, "长度", length, "超过最大支持字符长度", MaxStringLength)
		return true
	}
	return false
}
