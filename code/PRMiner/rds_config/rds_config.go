package rds_config

import (
	"math"
)

const CubeSize = 0

const ChanSize = 8

const MAXCpuNum = 16

const GinPort = "19123"

const PositiveSize = 100

// 相似度谓词相关配置
const (
	RegexCheckNum = 100
	RegexRatio    = 0.75
	JaroWinkler   = "jaro-winkler"
)

// 匹配列类型的正则表达式
const (
	AddressRegex = "^.+(市|区|镇|县|屯|).+(路|街|组|苑|村|小区|巷|区).+(栋|大厦|号|号楼|单元|楼|室|户).*$"
	CompanyRegex = "^.+(有限|公司|中心|大学|学校|医院|合作社|协会).*$"
)

const (
	RatioOfColumn = 0.1 //常数谓词频率
	TopK          = 50  //Topk谓词数
	RuleCoverage  = 0.8 // 当规则最大覆盖率不满足时,加入相似度谓词
)

// 开关
const (
	SaveIntersection = false
	UseStorage       = false
)

const (
	MultiFilterRowSize = 1000
	FilterCondition    = 16
)

// 谓词类型
const (
	Equal        = "="
	GreaterE     = ">="
	Less         = "<"
	NotEqual     = "!="
	Similar      = "similar"
	Regular      = "regular"
	RegularRight = "true"
	Poly         = "poly"
)

// 存储中的数据类型
const (
	StringType = "string"
	IntType    = "int64"
	FloatType  = "float64"
	BoolType   = "bool"
	TimeType   = "time"
	TextType   = "text"
	IndexType  = "index" // 索引类型
)

// 系统中现在没有enum类型, string类型列unique值<某个阈值时该列为enum
const (
	EnumType    = "enum"
	GroupByEnum = "enum"
	ListType    = "list"
)

const (
	JoinCrossPredicate  = 0
	OtherCrossPredicate = 1
)

// CR FTR
const (
	FTR = 0.9
	CR  = 0.05
)

// NilIndex nil 值索引
const NilIndex = int32(-1)

const PredicateRuleLimit = 15

const DecisionTreeThreshold = float64(0.0)

const DecisionTreeYNilReplace = math.MaxFloat64

var DecisionTreeSampleThreshold2Ratio = map[int]float64{
	2000000:  0.5,
	10000000: 0.1,
}

const ForeignKeyPredicateType = 2

const UdfColumnPrefix = "$"
const UdfColumnConn = "@"

const (
	NormalRuleFind = iota
	ScoreReeFind
)
