package main

import (
	"math"
	"rds-shenglin/decision_tree"
	"rds-shenglin/rds_config"
	"sort"

	"rds-shenglin/rock-share/base/logger"
	"rds-shenglin/rock-share/global/model/rds"
	"rds-shenglin/utils/train_data_util"
)

func ExecuteSingleRowDecisionTree(tableId string, yColumn string, columnsType map[string]string, tableLength int, gv *GlobalV) int {
	logger.Infof("ExecuteSingleRowDecisionTree, taskId: %v, tableId: %s , column :%s", gv.TaskId, tableId, yColumn)

	//取抽样数据，生成训练数据
	tableName := tableId
	columnType := columnsType
	sampleRatio := float64(1)
	sampleThreshold2Ratio := gv.DecisionTreeSampleThreshold2Ratio
	limit := 0
	if len(sampleThreshold2Ratio) == 0 {
		limit = gv.DecisionTreeMaxRowSize
		sampleThreshold2Ratio = rds_config.DecisionTreeSampleThreshold2Ratio
	}

	sortThreshold := make([]int, 0)
	for threshold := range sampleThreshold2Ratio {
		sortThreshold = append(sortThreshold, threshold)
	}
	sort.Slice(sortThreshold, func(i, j int) bool {
		return sortThreshold[i] > sortThreshold[j]
	})
	for _, threshold := range sortThreshold {
		if tableLength > threshold {
			sampleRatio = sampleThreshold2Ratio[threshold]
			break
		}
	}
	logger.Debugf("ExecuteSingleRowDecisionTree, taskId: %v, tableId: %s, tableLength: %v, sampleRatio: %v", gv.TaskId, tableId, tableLength, sampleRatio)
	sampleData := GetTableSampleData(tableId, columnType, limit, sampleRatio, gv)

	trainDataInput := make(map[string]map[string][]interface{})
	trainDataInput["t0"] = sampleData

	index2Table := make(map[string]string)
	index2Table["t0"] = tableName

	dataType := make(map[string]map[string]string)
	dataType[tableId] = columnType

	ratioMap := make(map[string]float64)
	ratioMap["t0"] = float64(1)
	columns, decisionTreeInput, yValue2Num, trainDataColumnType := train_data_util.GenerateTrainData([]map[string]map[string][]interface{}{trainDataInput}, []string{tableName}, index2Table, dataType, gv.enableEnum, gv.enableNum, false, rds.Predicate{
		LeftColumn: rds.Column{
			TableId: tableName,
			//TableAlias: "t0",
			ColumnId:   yColumn,
			ColumnType: columnType[yColumn],
		},
		PredicateType: 0,
	}, ratioMap, gv.DecisionTreeMaxRowSize)
	trainDataLength := 0
	if len(columns) > 0 {
		trainDataLength = len(decisionTreeInput)
	}
	logger.Debugf("get train data, taskId: %v, y: %s, trainTableLength: %v", gv.TaskId, yColumn, trainDataLength)
	//3.决策树
	rules, xSupports, xySupports, err := decision_tree.DecisionTreeWithDataInput(nil, rds.Predicate{}, columns, decisionTreeInput,
		trainDataColumnType, index2Table, yValue2Num, gv.TaskId, gv.Support, gv.Confidence, gv.DecisionTreeMaxDepth, &gv.StopTask)
	//规则入库
	ruleSize := 0
	for i, rule := range rules {
		if !checkMutex(rule.LhsPredicates, rule.Rhs, gv.MutexGroup, gv.GroupSize) {
			continue
		}
		gv.RuleSizeLock.Lock()
		ruleSize++
		gv.rules = append(gv.rules, rule)
		gv.RuleSizeLock.Unlock()
		logger.Infof("taskId: %v, y:%s, find decision tree rule: %v, rowSize: %v, xSupp: %v, xySupp: %v", gv.TaskId, yColumn, rule.Ree, len(decisionTreeInput), xSupports[i], xySupports[i])
	}
	if err != nil {
		logger.Errorf("execute decision tree error", err)
	}

	return ruleSize
}

func GetTableSampleData(tableId string, columnType map[string]string, limit int, ratio float64, gv *GlobalV) map[string][]interface{} {
	logger.Debugf("Decision Tree, get table sample data, tableId: %s, limit: %v, sampleRatio: %v", tableId, limit, ratio)
	length := gv.RowSizes[tableId] //当前节点该表数据行数, 目前是全表
	if limit != 0 {
		if limit > length {
			limit = length
		}
	} else {
		limitFloat := ratio * float64(length)
		limit = int(math.Ceil(limitFloat))
	}
	sampleData := map[string][]interface{}{}
	for columnName, theColumnType := range columnType {
		// 相似度ML不参与决策树生成的谓词
		//if strings.HasPrefix(columnName, udfColPrefix) {
		if theColumnType == rds_config.IndexType {
			continue
		}
		sampleData[columnName] = gv.TableRealValues[tableId][columnName][:limit]
	}
	return sampleData
}
