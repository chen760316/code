package rds_config

const (
	DecisionTreeMaxRowSize = 5000
	PredicateSupportLimit  = float64(0.95)
	TopKLayer              = 2
	TopKSize               = 200
	TreeLevel              = 3
	Confidence             = float64(0.9)
	Support                = float64(0.00001)
	DecisionTreeMaxDepth   = 3
	EnumSize               = 100
	SimilarThreshold       = float64(0.9)
	SchemaMappingThreshold = float64(0.8)
	TableRuleLimit         = 20
	RdsSize                = 50000000
	DecisionNodeSize       = 20
	EnableCrossTable       = false
	EnableErRule           = false
	EnableTimeRule         = false
	EnableDecisionTree     = true
	EnableML               = false
	EnableSimilar          = false
	EnableEnum             = true
	EnableNum              = false

	RdsSizeNoLimit = -1
)
