package enum

import "rds-shenglin/rock-share/base/logger"

const (
	STRING  = "字符串"
	DATE    = "日期"
	DIGITAL = "数字"
	BOOLEAN = "布尔"
	MONEY   = "货币"
)

const (
	ENSTRING  = "string"
	ENDATE    = "data"
	ENDIGITAL = "number"
	ENBOOLEAN = "boolean"
	ENMONEY   = "currency"
)

func FieldTypeToDisplay(s string, language string) string {
	if language == CHINESE {
		switch s {
		case "STRING":
			return STRING
		case "DATE":
			return DATE
		case "DIGITAL":
			return DIGITAL
		case "BOOLEAN":
			return BOOLEAN
		case "MONEY":
			return MONEY
		default:
			logger.Errorf("UNKNOWN enum:%s", s)
			return "UNKNOWN"
		}
	} else if language == ENGLISH {
		switch s {
		case "STRING":
			return ENSTRING
		case "DATE":
			return ENDATE
		case "DIGITAL":
			return ENDIGITAL
		case "BOOLEAN":
			return ENBOOLEAN
		case "MONEY":
			return ENMONEY
		default:
			logger.Errorf("UNKNOWN enum:%s", s)
			return "UNKNOWN"
		}
	}
	return "UNKNOWN"
}
