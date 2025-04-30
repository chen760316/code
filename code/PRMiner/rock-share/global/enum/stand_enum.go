package enum

// 数据角色 枚举（复用抽图组）

// StandType
// 标准化操作类型 枚举
type StandType int

const (
	Neatness StandType = iota
	Detection
	Enhancement
	InformationExtraction
)

// 规则类型 枚举
const (
	Regex      = "regex"      //正则规则
	Similarity = "similarity" //相似度规则
	Function   = "function"   //自定义函数
)

const (
	SingleReeType     = 0  //单行
	MultipleReeType   = 1  //多行
	RegexReeType      = 2  //正则
	PolynomialReeType = 3  //多项式
	ErReeType         = 4  //ER
	SequentialReeType = 5  //时序
	DetectionReeType  = 10 //检测类型
	NeatnessReeType   = 11 //规整类型
)

type StandTypeInfo struct {
	EnumName    string
	Description string
}

var StandTypeInfos = map[StandType]StandTypeInfo{
	Neatness:              {"NEATNESS", "规整"},
	Detection:             {"DETECTION", "检测"},
	Enhancement:           {"ENHANCEMENT", "增强"},
	InformationExtraction: {"INFORMATION_EXTRACTION", "信息抽取"},
}

func (t StandType) EnumName() string {
	if rule, ok := StandTypeInfos[t]; ok {
		return rule.Description
	}
	return ""
}

// StandRule
// 标准化规则 枚举
type StandRule int

const (
	Rule1 StandRule = iota
	Rule2
	Rule3
	Rule4
	Rule5
	Rule6
	Rule7
	Rule8
)

type PCNStandRuleInfo struct {
	Tag              string
	Type             StandType
	Operation        string
	ErrorDescription string
	Execution        bool
}
