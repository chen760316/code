package enum

// 质量评分三种规则类型枚举
type QuantityRuleEnum = string

const (
	TemplateRule QuantityRuleEnum = "TEMPLATE_RULE"
	RoleRule     QuantityRuleEnum = "ROLE_RULE"
	ReeRule      QuantityRuleEnum = "REE_RULE"
)

// 模版规则类型枚举
type TemplateRuleEnum = string

const (
	EmptyTemplate  TemplateRuleEnum = "EMPTY_TEMPLATE"
	EnumTemplate   TemplateRuleEnum = "ENUM_TEMPLATE"
	RegexTemplate  TemplateRuleEnum = "REGEX_TEMPLATE"
	LengthTemplate TemplateRuleEnum = "LENGTH_TEMPLATE"
	NumberTemplate TemplateRuleEnum = "NUMBER_TEMPLATE"
)

func GetNameByTEMPLATE(templateType TemplateRuleEnum) (name string) {
	switch templateType {
	case EmptyTemplate:
		name = "空值检查"
	case EnumTemplate:
		name = "枚举值检查"
	case RegexTemplate:
		name = "正则表达式检查"
	case LengthTemplate:
		name = "长度检查"
	case NumberTemplate:
		name = "数值检查"
	}
	return
}

func GetDescByTEMPLATE(templateType string) (desc string) {
	switch templateType {
	case EmptyTemplate:
		desc = "不能存在空值"
	case EnumTemplate:
		desc = "不在枚举值中"
	case RegexTemplate:
		desc = "正则表达式无法匹配"
	case LengthTemplate:
		desc = "长度不符合要求"
	case NumberTemplate:
		desc = "数值不符合要求"
	}
	return
}

// 六性枚举
type OldSixEnum = string

const (
	Accuracy     OldSixEnum = "ACCURACY"     //准确性
	Completeness OldSixEnum = "COMPLETENESS" //完整性
	Consistency  OldSixEnum = "CONSISTENCY"  //一致性
	Timeliness   OldSixEnum = "TIMELINESS"   //及时性
	Uniqueness   OldSixEnum = "UNIQUENESS"   //唯一性
	Validity     OldSixEnum = "VALIDITY"     //可信度
)

//操作符枚举

type OperatorEnum = string

const (
	GreaterThan    OperatorEnum = ">"
	LessThan       OperatorEnum = "<"
	NotLessThan    OperatorEnum = ">="
	NotGreaterThan OperatorEnum = "<="
	Equal          OperatorEnum = "="
	NotEqual       OperatorEnum = "!="
	Between        OperatorEnum = "in"
)

// 任务状态枚举
const (
	Running  = "RUNNING"
	Finished = "TASK_FINISH"
	Failed   = "FAILED"
)
