package main

import (
	"rds-shenglin/rds_config"
	"rds-shenglin/rock-share/base/logger"
	"rds-shenglin/rock-share/global/enum"
	"rds-shenglin/rock-share/global/model/rds"
	"rds-shenglin/utils"
	"runtime"
	"sync"
)

var GlobalVariable = make(map[int64]*GlobalV) //[taskID,GlobalV]
var GlobalVariableLock sync.RWMutex           //并发写入操作,加锁
// var specialColumns = map[string]bool{"id": true, "row_id": true, "update_time": true, "${df}": true, "${mk}": true}
var specialColumns = map[string]bool{"row_id": true, "update_time": true, "${df}": true, "${mk}": true}

type GlobalV struct {
	TaskId                  int64
	Support                 float64
	Confidence              float64
	PredicateSupportLimit   float64
	CrossTablePredicateSupp float64
	TreeLevel               int
	SingleRuleSize          int
	MultiRuleSize           int
	StopTask                bool
	HasError                bool
	TopKLayer               int
	TopKSize                int
	DecisionTreeMaxRowSize  int
	DecisionTreeMaxDepth    int
	EnumSize                int
	SimilarThreshold        float64
	MLThreshold             float64
	TableRuleLimit          int
	RdsSize                 int
	DecisionNodeSize        int

	enableErRule       bool
	enableTimeRule     bool
	enableDecisionTree bool
	enableML           bool
	enableSimilar      bool
	enableEnum         bool
	enableNum          bool

	TablesId               []string
	TablesIndex            map[string]int
	TablesName             map[string]string
	RowSizes               map[string]int
	ColumnsType            map[string]map[string]string
	TableRealValues        map[string]map[string][]interface{}           // 一个数据表的 表名 -> 列名 -> 值（slice)
	TableUdfBlockingValues map[string]map[string][][]int32               // 一个数据表的 表名 -> 列名 -> 行号 -> block结果（[]int32）
	TableIndexValues       map[string]map[string][]int32                 // 一个数据表的 表名 -> 列名 -> 索引值（值排序之后的次序, slice）
	Index2Value            map[string]map[string]map[int32]interface{}   // 一个数据表的 表名 -> 列名 -> [索引值 ->值]
	Value2Index            map[string]map[string]map[interface{}]int32   // 一个数据表的 表名 -> 列名 -> [值 -> 索引值]
	PLI                    map[string]map[string]map[interface{}][]int32 // tableId -> column -> value -> 有序rowIds
	UdfBlockingPLI         map[string]map[string]map[int32][]int32       // tableId -> column -> index -> 有序rowIds
	IndexPLI               map[string]map[string]map[int32][]int32

	UDFInfos []rds.UDFTabCol

	Table2DecisionY      map[string][]string
	TrainDataColumnsType map[string]map[string]string
	HasMultiRowDecision  bool

	Predicates           []rds.Predicate
	CrossTablePredicates [2][]rds.Predicate

	Table2JoinTables map[string]map[string]struct{}

	AbandonedRules []string

	rules            []rds.Rule
	WriteRulesChan   chan rds.Rule // 需要保存的规则
	NeedExeRulesChan chan rds.Rule // 需要查错的规则
	FinishedWriteDB  bool
	FinishedExe      bool

	RuleSizeLock                      sync.Mutex // 统计规则数量的锁
	DecisionTreeSampleThreshold2Ratio map[int]float64

	RowConflictSizeLock sync.Mutex
	RowConflictSize     map[string][]int32 // tableId -> 99999

	ColumnId2TableId map[string]string

	TableRules map[string]map[enum.RuleType][]rds.Rule

	skipYColumns map[string]bool
	skipColumns  []string

	Eids              map[string]string
	EidPredicates     []rds.Predicate
	ErTableRealValues map[string]map[string][]interface{}
	ErRowSizes        map[string]int

	MutexGroup map[string][]int
	GroupSize  int

	RuleFindType int

	SchemaMappingThreshold float64
	PrimaryKeys            map[string]bool
}

func InitRdsGlobalV(taskId int64, table Table, support, confidence float64) *GlobalV {
	GlobalVariableLock.Lock()
	gv, flag := GlobalVariable[taskId]
	if !flag {
		gv = &GlobalV{
			taskId,
			rds_config.Support,
			rds_config.Confidence,
			rds_config.PredicateSupportLimit,
			0.1,
			rds_config.TreeLevel,
			0,
			0,
			false,
			false,
			rds_config.TopKLayer,
			rds_config.TopKSize,
			rds_config.DecisionTreeMaxRowSize,
			rds_config.DecisionTreeMaxDepth,
			rds_config.EnumSize,
			rds_config.SimilarThreshold,
			0.9,
			rds_config.TableRuleLimit,
			rds_config.RdsSize,
			rds_config.DecisionNodeSize,
			rds_config.EnableErRule,
			rds_config.EnableTimeRule,
			rds_config.EnableDecisionTree,
			rds_config.EnableML,
			rds_config.EnableSimilar,
			rds_config.EnableEnum,
			rds_config.EnableNum,
			[]string{"0"},
			map[string]int{},
			map[string]string{},
			map[string]int{},
			map[string]map[string]string{},
			map[string]map[string][]interface{}{},
			map[string]map[string][][]int32{},
			map[string]map[string][]int32{},
			map[string]map[string]map[int32]interface{}{},
			map[string]map[string]map[interface{}]int32{},
			map[string]map[string]map[interface{}][]int32{},
			map[string]map[string]map[int32][]int32{},
			map[string]map[string]map[int32][]int32{},
			nil,
			map[string][]string{},
			map[string]map[string]string{},
			true,
			nil,
			[2][]rds.Predicate{},
			map[string]map[string]struct{}{},
			[]string{},
			[]rds.Rule{},
			make(chan rds.Rule, 1000),
			make(chan rds.Rule, 1000),
			false,
			false,
			sync.Mutex{},
			make(map[int]float64),
			sync.Mutex{},
			map[string][]int32{},
			map[string]string{},
			map[string]map[enum.RuleType][]rds.Rule{},
			map[string]bool{},
			nil,
			map[string]string{},
			[]rds.Predicate{},
			map[string]map[string][]interface{}{},
			map[string]int{},
			map[string][]int{},
			0,
			rds_config.NormalRuleFind,
			rds_config.SchemaMappingThreshold,
			map[string]bool{},
		}
		// 读取高级配置
		if support > 0 {
			gv.Support = support
		}
		if confidence > 0 {
			gv.Confidence = confidence
		}
		tablePath := table.Path
		tableId := "0"
		tableName := "0"
		gv.TablesIndex[tableId] = 0
		gv.TrainDataColumnsType[tableId] = map[string]string{}
		tableColumnsType := table.ColumnsType
		data, tableRows, e := utils.ReadCSVToMap(tablePath)
		if e != nil {
			logger.Errorf("error:%v", e)
		}
		gv.RowConflictSize[tableId] = make([]int32, tableRows)
		gv.TableRealValues[tableId] = data
		for column := range tableColumnsType {
			gv.ColumnId2TableId[column] = tableId
		}
		gv.TablesName[tableId], gv.RowSizes[tableId], gv.ColumnsType[tableId] = tableName, tableRows, tableColumnsType
		LoadDataCreatePli(gv)
		CreateIndex(gv)
		GlobalVariable[taskId] = gv
	}
	GlobalVariableLock.Unlock()
	return gv
}

func (gv *GlobalV) Stop() bool {
	return gv.StopTask
}

func GetGv(taskId int64) *GlobalV {
	GlobalVariableLock.Lock()
	defer GlobalVariableLock.Unlock()
	return GlobalVariable[taskId]
}

var TaskCh = GenTokenChan(0.5) // 构建谓词,构建树,规则执行共享一组协程（类似于信号量）
var CalCh = GenTokenChan(1)    // 计算不满足要求的节点(决策树)共享一组协程（类似于信号量）

func GenTokenChan(coefficient float64) chan struct{} {
	// 后端那边直接限制了cpu的核数要留1个
	cpuNum := runtime.NumCPU() - 1
	if cpuNum <= 0 {
		cpuNum = 1
	}
	tokenNum := int(coefficient * float64(cpuNum))
	if tokenNum <= 0 {
		tokenNum = 1
	}
	ch := make(chan struct{}, tokenNum)
	for i := 0; i < tokenNum; i++ {
		ch <- struct{}{}
	}
	return ch
}

const udfColPrefix = rds_config.UdfColumnPrefix
const udfColConn = rds_config.UdfColumnConn

// LoadDataCreatePli 加载需要的列的数据。需要先调用 LoadSchema 和 SetBlockingInfo
func LoadDataCreatePli(gv *GlobalV) {
	for _, tableId := range gv.TablesId {
		//gv.TableRealValues[tableId] = map[string][]interface{}{}
		gv.TableUdfBlockingValues[tableId] = map[string][][]int32{}
		gv.PLI[tableId] = map[string]map[interface{}][]int32{}
		gv.UdfBlockingPLI[tableId] = map[string]map[int32][]int32{}
		for column := range gv.ColumnsType[tableId] {
			values := gv.TableRealValues[tableId][column]

			pli := map[interface{}][]int32{}
			for rowId, value := range values {
				pli[value] = append(pli[value], int32(rowId))
			}
			gv.PLI[tableId][column] = pli
			logger.Infof("finish create pli tableId:%v, columnId:%v", tableId, column)
		}
		logger.Infof("finish create pli tableId:%v", tableId)
	}
}

// CreateIndex 先 LoadSchema 然后 SetBlockingInfo 然后 LoadDataCreatePli
func CreateIndex(gv *GlobalV) {
	idProvider := map[interface{}]int32{}
	idProvider[nil] = rds_config.NilIndex
	idProvider[""] = rds_config.NilIndex
	providerCnt := int32(0)

	gv.TableIndexValues = map[string]map[string][]int32{}
	for tableName, columnValues := range gv.TableRealValues {
		gv.TableIndexValues[tableName] = map[string][]int32{}
		for columnName, values := range columnValues {
			var columnIndexes = make([]int32, len(values))
			for i, value := range values {
				index, ok := idProvider[value]
				if !ok {
					idProvider[value] = providerCnt
					index = providerCnt
					providerCnt++
				}
				columnIndexes[i] = index
			}
			gv.TableIndexValues[tableName][columnName] = columnIndexes
			logger.Infof("finish create index tableId:%v, columnId:%v", tableName, columnName)
		}
		logger.Infof("finish create index tableId:%v", tableName)
	}

	gv.IndexPLI = map[string]map[string]map[int32][]int32{}
	for tableName, columnPLI := range gv.PLI {
		gv.IndexPLI[tableName] = map[string]map[int32][]int32{}
		for columnName, pli := range columnPLI {
			indexPli := map[int32][]int32{}
			for value, rowIds := range pli {
				indexPli[idProvider[value]] = rowIds
			}
			gv.IndexPLI[tableName][columnName] = indexPli
			logger.Infof("finish create index pli tableId:%v, columnId:%v", tableName, columnName)
		}
		logger.Infof("finish create index pli tableId:%v", tableName)
	}

	//task.TableValues = nil
	//task.PLI = nil
}

func ClearMemory(taskId int64) {
	GlobalVariableLock.Lock()
	_, ok := GlobalVariable[taskId]
	if !ok {
		logger.Warnf("[ClearMemory] 无法拿到任务:%d的GlobalVariable", taskId)
	}
	GlobalVariable[taskId] = nil
	delete(GlobalVariable, taskId)
	GlobalVariableLock.Unlock()
	runtime.GC()
}
