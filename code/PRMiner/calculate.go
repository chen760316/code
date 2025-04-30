package main

import (
	"rds-shenglin/decision_tree"
	"rds-shenglin/rds_config"
	"rds-shenglin/rock-share/base/logger"
	"rds-shenglin/rock-share/global/enum"
	"rds-shenglin/rock-share/global/model/rds"
	"rds-shenglin/utils"
	"rds-shenglin/utils/train_data_util"
	"sort"
	"strconv"
	"time"
)

func SampleIntersection(predicates []rds.Predicate, cubeSize int, taskId int64, isRds bool) [][][]int32 {
	var intersection [][][]int32
	var fkPredicate []rds.Predicate
	var innerPredicate []rds.Predicate
	for _, p := range predicates {
		if p.PredicateType == rds_config.ForeignKeyPredicateType {
			fkPredicate = append(fkPredicate, p)
		} else if p.PredicateType == 1 {
			innerPredicate = append(innerPredicate, p)
		}
	}
	for _, lh := range predicates {
		intersection = CalIntersection(intersection, lh, taskId, isRds)
		if len(intersection) < 1 {
			break
		}
	}
	if len(intersection) > cubeSize && cubeSize > 0 {
		intersection = intersection[:cubeSize]
	}
	return intersection
}

func CalIntersectionSupport(intersection [][][]int32, useTableIndexes []int) int {
	if len(intersection) < 1 {
		return 0
	}

	var supp = 0
	//var wg sync.WaitGroup
	//var lock sync.Mutex
	//for _, idPairs := range intersection {
	//	wg.Add(1)
	//	<-CalCh
	//	go func(idPairs [][]int32) {
	//		defer func() {
	//			wg.Done()
	//			CalCh <- struct{}{}
	//			if err := recover(); err != nil {
	//				s := string(debug.Stack())
	//				logger.Error("recover.err:%v, stack:%v", err, s)
	//			}
	//		}()
	//		var temp = 1
	//		for _, index := range useTableIndexes {
	//			temp *= len(idPairs[index])
	//		}
	//		lock.Lock()
	//		supp += temp
	//		lock.Unlock()
	//	}(idPairs)
	//}
	//wg.Wait()

	group := make(map[int][]int)
	for _, index := range useTableIndexes {
		key := index / 2
		group[key] = append(group[key], index)
	}

	for _, idPairs := range intersection {
		var temp = 1
		for _, indexes := range group {
			t := 0
			if len(indexes) == 1 {
				t = len(idPairs[indexes[0]])
			} else {
				left := len(idPairs[indexes[0]])
				right := len(idPairs[indexes[1]])
				if left > right {
					t = (left - 1) * right
				} else {
					t = left * (right - 1)
				}
			}
			temp *= t
		}
		//for _, index := range useTableIndexes {
		//	temp *= len(idPairs[index])
		//}
		supp += temp
	}

	return supp
}

func GeneratePredicateIntersection(predicate rds.Predicate, taskId int64, isRds bool) [][][]int32 {
	var intersection [][][]int32
	isConstant := predicate.PredicateType == 0
	if isConstant {
		leftTableId := predicate.LeftColumn.TableId
		leftColumnName := predicate.LeftColumn.ColumnId
		leftColumnType := predicate.LeftColumn.ColumnType
		if leftColumnType == rds_config.IndexType {
			leftColumnName = udfColPrefix + strconv.Itoa(predicate.UdfIndex) + udfColConn + leftColumnName
		}
		constantValue := predicate.ConstantValue
		leftTableIndex, _ := utils.GetPredicateColumnIndex(predicate)
		gv := GetGv(taskId)
		var leftPli = gv.PLI[leftTableId][leftColumnName]
		var rows []int32
		switch predicate.SymbolType {
		case rds_config.GreaterE:
			rows = GetGreaterOrLessValueIndex(constantValue, true, true, leftPli, leftColumnType)
		case rds_config.Less:
			rows = GetGreaterOrLessValueIndex(constantValue, false, true, leftPli, leftColumnType)
		case rds_config.NotEqual:
			rows = GetNotEqualValueIndex(constantValue, leftPli)
		default:
			rows = leftPli[constantValue]
		}
		idPairs := make([][]int32, leftTableIndex+1)
		idPairs[leftTableIndex] = rows
		intersection = append(intersection, idPairs)
		return intersection
	}

	leftTableId := predicate.LeftColumn.TableId
	rightTableId := predicate.RightColumn.TableId
	leftColumnName := predicate.LeftColumn.ColumnId
	rightColumnName := predicate.RightColumn.ColumnId
	leftColumnType := predicate.LeftColumn.ColumnType
	rightColumnType := predicate.RightColumn.ColumnType
	if leftColumnType == rds_config.IndexType {
		leftColumnName = udfColPrefix + strconv.Itoa(predicate.UdfIndex) + udfColConn + leftColumnName
	}
	if rightColumnType == rds_config.IndexType {
		rightColumnName = udfColPrefix + strconv.Itoa(predicate.UdfIndex) + udfColConn + rightColumnName
	}
	leftTableIndex, rightTableIndex := utils.GetPredicateColumnIndex(predicate)
	maxTableIndex := rightTableIndex
	if leftTableIndex > rightTableIndex {
		maxTableIndex = leftTableIndex
	}
	gv := GetGv(taskId)
	var leftPli, rightPli map[int32][]int32
	if isRds || predicate.SymbolType == enum.Equal {
		leftPli, rightPli = gv.IndexPLI[leftTableId][leftColumnName], gv.IndexPLI[rightTableId][rightColumnName]
	} else {
		leftPli, rightPli = gv.UdfBlockingPLI[leftTableId][leftColumnName], gv.UdfBlockingPLI[rightTableId][rightColumnName]
	}
	isSameColumn := leftTableId == rightTableId && leftColumnName == rightColumnName
	for value, leftIds := range leftPli {
		if value == rds_config.NilIndex {
			continue
		}
		idPairs := make([][]int32, maxTableIndex+1)
		if isSameColumn {
			idPairs[leftTableIndex] = leftIds
			idPairs[rightTableIndex] = leftIds
		} else {
			rightIds := rightPli[value]
			if len(rightIds) > 0 {
				idPairs[leftTableIndex] = leftIds
				idPairs[rightTableIndex] = rightIds
			} else {
				continue
			}
		}
		intersection = append(intersection, idPairs)
	}
	return intersection
}

func CalIntersection(intersection [][][]int32, predicate rds.Predicate, taskId int64, isRds bool) [][][]int32 {
	if len(intersection) < 1 {
		return GeneratePredicateIntersection(predicate, taskId, isRds)
	}
	isConstant := predicate.PredicateType == 0
	if isConstant {
		var resultIntersection [][][]int32
		switch predicate.SymbolType {
		case rds_config.GreaterE:
			resultIntersection = calculateGreaterOrLess(intersection, predicate, taskId, true, true)
		case rds_config.Less:
			resultIntersection = calculateGreaterOrLess(intersection, predicate, taskId, false, false)
		case rds_config.NotEqual:
			resultIntersection = calculateNotEqual(intersection, predicate, taskId)
		default:
			resultIntersection = calculateEqual(intersection, predicate, taskId)
		}
		return resultIntersection
	}

	leftTableId := predicate.LeftColumn.TableId
	rightTableId := predicate.RightColumn.TableId
	leftColumnName := predicate.LeftColumn.ColumnId
	rightColumnName := predicate.RightColumn.ColumnId
	leftColumnType := predicate.LeftColumn.ColumnType
	rightColumnType := predicate.RightColumn.ColumnType
	if leftColumnType == rds_config.IndexType {
		leftColumnName = udfColPrefix + strconv.Itoa(predicate.UdfIndex) + udfColConn + leftColumnName
	}
	if rightColumnType == rds_config.IndexType {
		rightColumnName = udfColPrefix + strconv.Itoa(predicate.UdfIndex) + udfColConn + rightColumnName
	}

	leftTableIndex, rightTableIndex := utils.GetPredicateColumnIndex(predicate)

	gv := GetGv(taskId)
	leftValues := gv.TableIndexValues[leftTableId][leftColumnName]
	rightValues := gv.TableIndexValues[rightTableId][rightColumnName]
	var leftPli, rightPli = gv.IndexPLI[leftTableId][leftColumnName], gv.IndexPLI[rightTableId][rightColumnName]

	var newIdPairSize = len(intersection[0])
	newIdPairSize = utils.Max(newIdPairSize, leftTableIndex+1)
	newIdPairSize = utils.Max(newIdPairSize, rightTableIndex+1)

	var leftPad = leftTableIndex >= len(intersection[0]) || len(intersection[0][leftTableIndex]) < 1
	var rightPad = rightTableIndex >= len(intersection[0]) || len(intersection[0][rightTableIndex]) < 1

	var noLimitIndexes []int
	for tableIndex := range intersection[0] {
		if tableIndex != leftTableIndex && tableIndex != rightTableIndex {
			noLimitIndexes = append(noLimitIndexes, tableIndex)
		}
	}

	var resultIntersection [][][]int32

	for _, idPairs := range intersection {
		var tempIdPairs [][][]int32
		value2Index := make(map[int32]int)
		for tableIndex, ids := range idPairs {
			if tableIndex == leftTableIndex || tableIndex == rightTableIndex {
				for _, rowId := range ids {
					var value int32
					if tableIndex == leftTableIndex {
						value = leftValues[rowId]
					} else {
						value = rightValues[rowId]
					}
					if value == rds_config.NilIndex {
						continue
					}
					index, ok := value2Index[value]
					if !ok {
						var pad []int32
						if rightPad && tableIndex == leftTableIndex {
							pad = rightPli[value]
							if len(pad) == 0 {
								continue
							}
						}
						if leftPad && tableIndex == rightTableIndex {
							pad = leftPli[value]
							if len(pad) == 0 {
								continue
							}
						}
						value2Index[value] = len(tempIdPairs)
						index = len(tempIdPairs)
						tempIdPairs = append(tempIdPairs, make([][]int32, newIdPairSize))
						if leftPad {
							tempIdPairs[index][leftTableIndex] = pad
						}
						if rightPad {
							tempIdPairs[index][rightTableIndex] = pad
						}
					}
					tempIdPairs[index][tableIndex] = append(tempIdPairs[index][tableIndex], rowId)
				}
			}
		}

		for k := range tempIdPairs {
			flag := false
			for tableIndex, rowIds := range tempIdPairs[k] {
				if tableIndex == leftTableIndex || tableIndex == rightTableIndex {
					if len(rowIds) < 1 {
						flag = true
						continue
					}
				}
			}
			if flag {
				continue
			}
			for _, tableIndex := range noLimitIndexes {
				tempIdPairs[k][tableIndex] = idPairs[tableIndex]
			}
		}
		resultIntersection = append(resultIntersection, tempIdPairs...)
	}
	return resultIntersection
}

func GetGreaterOrLessValueIndex(constantValue interface{}, greater bool, hasEqual bool, pli map[interface{}][]int32, columnType string) []int32 {
	var satisfyRows []int32
	for currentValue, rowIds := range pli {
		if currentValue == nil || currentValue == "" {
			continue
		}
		if columnType == rds_config.FloatType || columnType == rds_config.IntType {
			compareResult, err := utils.CompareTo(currentValue, constantValue, columnType)
			if err != nil {
				logger.Errorf("[GetNotEqualValueIndex] interface比较失败,error=%v", err)
				continue
			}
			if greater && compareResult == 1 {
				satisfyRows = append(satisfyRows, rowIds...)
			}
			if !greater && compareResult == -1 {
				satisfyRows = append(satisfyRows, rowIds...)
			}
		}
		if hasEqual && currentValue == constantValue {
			satisfyRows = append(satisfyRows, rowIds...)
		}
	}
	return satisfyRows
}

func GetNotEqualValueIndex(constantValue interface{}, pli map[interface{}][]int32) []int32 {
	var satisfyRows []int32
	for currentValue, rowIds := range pli {
		if currentValue == nil || currentValue == "" {
			continue
		}
		if currentValue != constantValue {
			satisfyRows = append(satisfyRows, rowIds...)
		}
	}
	return satisfyRows
}

func calculateGreaterOrLess(intersection [][][]int32, predicate rds.Predicate, taskId int64, greater bool, hasEqual bool) [][][]int32 {
	var resultIntersection [][][]int32
	gv := GetGv(taskId)
	leftTableId := predicate.LeftColumn.TableId
	leftColumnName := predicate.LeftColumn.ColumnId
	leftColumnType := predicate.LeftColumn.ColumnType
	if leftColumnType == rds_config.IndexType {
		leftColumnName = udfColPrefix + strconv.Itoa(predicate.UdfIndex) + udfColConn + leftColumnName
	}
	leftValues := gv.TableRealValues[leftTableId][leftColumnName]
	constantValue := predicate.ConstantValue
	lTableIndex, _ := utils.GetPredicateColumnIndex(predicate)
	for _, idPairs := range intersection {
		newIdPairs := make([][]int32, len(idPairs))
		for tableIndex, ids := range idPairs {
			if lTableIndex == tableIndex {
				var satisfyIds []int32
				for _, id := range ids {
					currentValue := leftValues[id]
					if currentValue == nil || currentValue == "" {
						continue
					}
					if leftColumnType == rds_config.FloatType || leftColumnType == rds_config.IntType {
						compareResult, err := utils.CompareTo(currentValue, constantValue, leftColumnType)
						if err != nil {
							logger.Errorf("[GetNotEqualValueIndex] interface比较失败,error=%v", err)
							continue
						}
						if greater && compareResult == 1 {
							satisfyIds = append(satisfyIds, id)
						}
						if !greater && compareResult == -1 {
							satisfyIds = append(satisfyIds, id)
						}
					}
					if hasEqual && currentValue == constantValue {
						satisfyIds = append(satisfyIds, id)
					}
				}
				newIdPairs[tableIndex] = satisfyIds
			} else {
				newIdPairs[tableIndex] = ids
			}
		}
		resultIntersection = append(resultIntersection, newIdPairs)
	}
	return resultIntersection
}

func calculateNotEqual(intersection [][][]int32, predicate rds.Predicate, taskId int64) [][][]int32 {
	var resultIntersection [][][]int32
	gv := GetGv(taskId)
	leftTableId := predicate.LeftColumn.TableId
	leftColumnName := predicate.LeftColumn.ColumnId
	leftColumnType := predicate.LeftColumn.ColumnType
	if leftColumnType == rds_config.IndexType {
		leftColumnName = udfColPrefix + strconv.Itoa(predicate.UdfIndex) + udfColConn + leftColumnName
	}
	leftValues := gv.TableIndexValues[leftTableId][leftColumnName]
	constantIndexValue := getIndexValueByValue(predicate.ConstantValue, taskId, leftTableId, leftColumnName)
	lTableIndex, _ := utils.GetPredicateColumnIndex(predicate)
	for _, idPairs := range intersection {
		newIdPairs := make([][]int32, len(idPairs))
		for tableIndex, ids := range idPairs {
			if lTableIndex == tableIndex {
				var satisfyIds []int32
				for _, id := range ids {
					currentIndexValue := leftValues[id]
					if currentIndexValue == rds_config.NilIndex {
						continue
					}
					if currentIndexValue != constantIndexValue {
						satisfyIds = append(satisfyIds, id)
					}
				}
				newIdPairs[tableIndex] = satisfyIds
			} else {
				newIdPairs[tableIndex] = ids
			}
		}
		resultIntersection = append(resultIntersection, newIdPairs)
	}
	return resultIntersection
}

func calculateEqual(intersection [][][]int32, predicate rds.Predicate, taskId int64) [][][]int32 {
	var resultIntersection [][][]int32
	gv := GetGv(taskId)
	leftTableId := predicate.LeftColumn.TableId
	leftColumnName := predicate.LeftColumn.ColumnId
	leftColumnType := predicate.LeftColumn.ColumnType
	if leftColumnType == rds_config.IndexType {
		leftColumnName = udfColPrefix + strconv.Itoa(predicate.UdfIndex) + udfColConn + leftColumnName
	}
	leftValues := gv.TableIndexValues[leftTableId][leftColumnName]
	constantIndexValue := getIndexValueByValue(predicate.ConstantValue, taskId, leftTableId, leftColumnName)
	lTableIndex, _ := utils.GetPredicateColumnIndex(predicate)
	for _, idPairs := range intersection {
		newIdPairs := make([][]int32, len(idPairs))
		var hasAppend = false
		for tableIndex, ids := range idPairs {
			if lTableIndex == tableIndex {
				var satisfyIds []int32
				for _, id := range ids {
					var currentIndexValue int32
					if leftValues == nil {
						currentIndexValue = id
					} else {
						currentIndexValue = leftValues[id]
					}
					if currentIndexValue == rds_config.NilIndex {
						continue
					}
					if currentIndexValue == constantIndexValue {
						satisfyIds = append(satisfyIds, id)
						hasAppend = true
					}
				}
				newIdPairs[tableIndex] = satisfyIds
			} else {
				newIdPairs[tableIndex] = ids
			}
		}
		if hasAppend {
			resultIntersection = append(resultIntersection, newIdPairs)
		}
	}
	return resultIntersection
}

func getIndexValueByValue(value interface{}, taskId int64, tableId string, columnName string) (indexValue int32) {

	gv := GetGv(taskId)
	rows := gv.PLI[tableId][columnName][value]
	if len(rows) == 0 {
		logger.Warnf("[GetIndexValueByValue] 找不到taskId:%d|tableId:%s|columnName:%s|value:%v对应的rows", taskId, tableId, columnName, value)
		return indexValue
	}
	indexValue = gv.TableIndexValues[tableId][columnName][rows[0]]
	return indexValue
}

func GetTableIndex2tableId(lhs []rds.Predicate) map[int]string {
	r := map[int]string{}
	for _, p := range lhs {
		lid, rid := utils.GetPredicateColumnIndex(p)
		r[lid] = p.LeftColumn.TableId
		r[rid] = p.RightColumn.TableId
	}
	return r
}

func checkNode(node *TaskTree) bool {
	// 检查lhs和lhs+rhs是否都是连通图,两者有一个不满足则不计算该节点
	// rhs中出现过的列索引必须在lhs中也出现
	lhs := node.Lhs
	tmpP := make([]rds.Predicate, len(lhs)+1)
	copy(tmpP, lhs)
	tmpP[len(lhs)] = node.Rhs
	m := make(map[int]bool)
	for _, pre := range node.Lhs {
		m[pre.LeftColumn.ColumnIndex] = true
		m[pre.RightColumn.ColumnIndex] = true
	}
	if utils.CheckPredicatesIsConnectedGraph(lhs) && utils.CheckPredicatesIsConnectedGraph(tmpP) && m[node.Rhs.LeftColumn.ColumnIndex] && m[node.Rhs.RightColumn.ColumnIndex] {
		return true
	}
	logger.Debugf("lhs:%v, rhs:%v, 不满足计算要求", utils.GetLhsStr(lhs), node.Rhs.PredicateStr)
	return false
}

func calNodes(nodes []*TaskTree, gv *GlobalV) ([]bool, []bool, []bool, int) {
	hasRules, prunes, isDeletes := make([]bool, len(nodes)), make([]bool, len(nodes)), make([]bool, len(nodes))
	calCnt := 0
	//var wg sync.WaitGroup
	//var lock sync.Mutex
	for i, node := range nodes {
		//t := time.Now().UnixMilli()
		//logger.Infof("cal lhs:%v, rhs:%v", utils.GetLhsStr(node.Lhs), node.Rhs.PredicateStr)
		// 检查lhs和lhs+rhs是否都是连通图,两者有一个不满足则不计算该节点
		if !checkNode(node) {
			hasRules[i] = false
			prunes[i] = false
			isDeletes[i] = false
			continue
		}
		// 检测规则中是否存在互斥的列
		if !checkMutex(node.Lhs, node.Rhs, gv.MutexGroup, gv.GroupSize) {
			hasRules[i] = false
			prunes[i] = true
			isDeletes[i] = false
			continue
		}

		// 跨表谓词中表内谓词的supp太大,过滤掉
		if !utils.CheckSatisfyPredicateSupp(node.Lhs, node.TableId2index, gv.CrossTablePredicateSupp) {
			hasRules[i] = false
			prunes[i] = false
			isDeletes[i] = false
			continue
		}
		calCnt++

		rowSize, xSupp, xySupp, _ := CalRule(node.Lhs, node.Rhs, gv)
		var support, confidence float64 = 0, 0
		if rowSize > 0 {
			support = float64(xySupp) / float64(rowSize)
		}
		if xSupp > 0 {
			confidence = float64(xySupp) / float64(xSupp)
		}

		node.Confidence = confidence

		var hasRule, prune, isDelete, isAbandoned bool

		hasRule = gv.Support <= support && gv.Confidence <= confidence
		prune = hasRule || gv.Support > support || rds_config.DecisionTreeThreshold > confidence
		isDelete = hasRule

		if hasRule {
			px := utils.GetLhsStr(node.Lhs)
			ree := px + "->" + node.Rhs.PredicateStr
			//ree = utils.GenerateMultiTableIdStr(node.TableId2index) + ree
			ree = utils.GenerateMultiTableIdStrNew2(node.TableId2index, node.Lhs) + " ^ " + ree

			if utils.IsAbandoned(ree, gv.AbandonedRules) {
				logger.Infof("规则:%v 曾经被废弃过,不再进行后续操作", ree)
				isAbandoned = true
			} else {
				ruleType := 1

				rule := rds.Rule{
					TableId:       "0",
					Ree:           ree,
					LhsPredicates: node.Lhs,
					LhsColumns:    utils.GetPredicatesColumn(node.Lhs),
					Rhs:           node.Rhs,
					RhsColumn:     node.Rhs.LeftColumn,
					CR:            support,
					FTR:           confidence,
					RuleType:      ruleType,
					XSupp:         xSupp,
					XySupp:        xySupp,
					XSatisfyCount: 0,
					XSatisfyRows:  nil,
					XIntersection: nil,
				}

				logger.Infof("%v find rule: %v, rowSize: %v, xySupp: %v, xSupp: %v", gv.TaskId, ree, rowSize, xySupp, xSupp)
				//gv.WriteRulesChan <- rule
				gv.RuleSizeLock.Lock()
				gv.rules = append(gv.rules, rule)
				gv.RuleSizeLock.Unlock()
			}
		}
		if isAbandoned {
			hasRule = false
			prune = true
			isDelete = true
		}
		hasRules[i] = hasRule
		prunes[i] = prune
		isDeletes[i] = isDelete
	}
	return hasRules, prunes, isDeletes, calCnt
}

func CalRule(lhs []rds.Predicate, rhs rds.Predicate, gv *GlobalV) (int, int, int, [][][]int32) {
	logger.Debugf("start cal rule:%v->%v", utils.GetLhsStr(lhs), rhs.PredicateStr)
	taskId := gv.TaskId
	lhs = utils.SortPredicatesRelated(lhs)
	//lhs = utils.SortPredicatesCrossF(lhs)
	var fkPredicates, singlePredicates []rds.Predicate
	for _, lh := range lhs {
		if lh.PredicateType == rds_config.ForeignKeyPredicateType {
			fkPredicates = append(fkPredicates, lh)
		} else if lh.PredicateType == 1 {
			singlePredicates = append(singlePredicates, lh)
		}
	}
	var xSupp, xySupp, rowSize int
	var intersection [][][]int32
	var usedTableIndex []int
	for _, lh := range lhs {
		lTid, rTid := utils.GetPredicateColumnIndex(lh)
		usedTableIndex = append(usedTableIndex, lTid)
		if rTid >= 0 {
			usedTableIndex = append(usedTableIndex, rTid)
		}
		intersection = CalIntersection(intersection, lh, taskId, true)
		if len(intersection) < 1 {
			break
		}
	}
	usedTableIndex = utils.Distinct(usedTableIndex)
	xSupp = CalIntersectionSupport(intersection, usedTableIndex)
	if xSupp == 0 {
		return 0, 0, 0, nil
	}

	xyIntersection := CalIntersection(intersection, rhs, taskId, true)
	xySupp = CalIntersectionSupport(xyIntersection, usedTableIndex)

	tableIndex2tableId := GetTableIndex2tableId(lhs)
	rowSize = 1
	for _, tableId := range tableIndex2tableId {
		rowSize *= gv.RowSizes[tableId]
	}
	if len(fkPredicates) == len(lhs) {
		rowSize = xSupp
	}

	logger.Debugf("finish cal rule:%v->%v, rowSize:%v, xSupp:%v, xySupp:%v", utils.GetLhsStr(lhs), rhs.PredicateStr, rowSize, xSupp, xySupp)

	return rowSize, xSupp, xySupp, nil
}

func CalDecisionNodes(gv *GlobalV, nodes []*TaskTree) int {
	taskId := gv.TaskId
	support := gv.Support
	confidence := gv.Confidence
	tableColumnType := gv.ColumnsType
	decisionTreeMaxDepth := gv.DecisionTreeMaxDepth

	// 把节点按照规则的confidence进行排序
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].Confidence > nodes[j].Confidence
	})

	ruleSize := 0
	i := 1
	for _, node := range nodes {
		if i > gv.DecisionNodeSize {
			break
		}
		i++
		if !checkNode(node) {
			continue
		}
		t := time.Now().UnixMilli()
		lhs := node.Lhs
		rhs := node.Rhs
		if isSingleLineCrossTable(lhs, rhs) {
			continue
		}
		logger.Debugf("cal not satisfy node lhs:%v, rhs:%v", utils.GetLhsStr(lhs), rhs.PredicateStr)
		index2table, _ := generateIndex2TableNew(append(lhs, rhs))
		tableId2index := generateTableId2tableIndex(lhs)
		// lhs的交集结果
		logger.Debugf("get %v intersection", utils.GetLhsStr(lhs))

		intersection := SampleIntersection(lhs, rds_config.CubeSize, taskId, true)
		if len(intersection) < 1 {
			logger.Infof("lhs:%v, intersection size is 0", utils.GetLhsStr(lhs), rhs.PredicateStr)
			continue
		}
		t1 := time.Now().UnixMilli() - t
		logger.Debugf("generate train data lhs:%v, rhs:%v", utils.GetLhsStr(lhs), rhs.PredicateStr)
		filterRatio := make(map[string]float64)
		for index := range index2table {
			filterRatio[index] = 1
		}
		columns, columnsType := train_data_util.GenerateHeader(tableId2index, node.TrainDataColumnsType, node.TrainDataColumns, rhs)
		if rhs.PredicateType == 3 {
			columns, columnsType = train_data_util.FilterHeader(columnsType, lhs)
		}
		trainData := train_data_util.GenerateTrainDataFromIntersectionNew(intersection, index2table, tableColumnType, gv.enableEnum, gv.enableNum, false, rhs, filterRatio, gv.TableRealValues, columns, columnsType, gv.DecisionTreeMaxRowSize)
		// 生成的训练数据有时候长度为0了,可能是因为抽样的原因?
		if len(trainData) < 1 {
			logger.Infof("lhs:%v, rhs:%v, train data size is 0", utils.GetLhsStr(lhs), rhs.PredicateStr)
			continue
		}
		t2 := time.Now().UnixMilli() - t - t1
		logger.Debugf("lhs:%v, rhs:%v, train data row size:%v, column size:%v", utils.GetLhsStr(lhs), rhs.PredicateStr, len(trainData), len(columns))

		// 调用决策树
		curRuleSize := 0
		rules, _, _, err := decision_tree.DecisionTreeWithDataInput(lhs, rhs, columns, trainData, columnsType, index2table, nil, taskId, support, confidence, decisionTreeMaxDepth, &gv.StopTask)
		if err != nil {
			logger.Errorf("execute decision tree err:%v", err)
			continue
		}
		for _, rule := range rules {
			if !checkMutex(rule.LhsPredicates, rule.Rhs, gv.MutexGroup, gv.GroupSize) {
				continue
			}
			logger.Infof("taskId: %v, find decision tree rule: %v, rowSize: %v, xSupp: %v, xySupp: %v", gv.TaskId, rule.Ree, len(trainData), rule.XSupp, rule.XySupp)
			gv.RuleSizeLock.Lock()
			curRuleSize++
			gv.rules = append(gv.rules, rule)
			gv.RuleSizeLock.Unlock()
		}
		ruleSize += curRuleSize
		t3 := time.Now().UnixMilli() - t - t1 - t2

		logger.Debugf("finish cal not satisfy node lhs:%v, rhs:%v, rule size:%v spent time:%vms:"+
			"cal intersection time:%v, generate train data time:%v, decision tree time:%v",
			utils.GetLhsStr(lhs), rhs.PredicateStr, curRuleSize, time.Now().UnixMilli()-t, t1, t2, t3)
	}
	return ruleSize
}

func isSingleLineCrossTable(lhs []rds.Predicate, rhs rds.Predicate) bool {
	if rhs.PredicateType != 3 {
		return false
	}
	for _, p := range lhs {
		if p.PredicateType != 3 {
			return false
		}
	}
	return true
}

func generateIndex2TableNew(predicates []rds.Predicate) (map[string]string, []string) {
	index2Table := make(map[string]string)
	tableMap := make(map[string]bool)
	for _, predicate := range predicates {
		leftIndex := "t" + strconv.Itoa(predicate.LeftColumn.ColumnIndex)
		rightIndex := "t" + strconv.Itoa(predicate.RightColumn.ColumnIndex)
		leftTableId := predicate.LeftColumn.TableId
		rightTableId := predicate.RightColumn.TableId
		index2Table[leftIndex] = leftTableId
		index2Table[rightIndex] = rightTableId
		tableMap[leftTableId] = true
		tableMap[rightTableId] = true
	}
	tableArr := make([]string, len(tableMap))
	i := 0
	for tableId := range tableMap {
		tableArr[i] = tableId
		i++
	}
	return index2Table, tableArr
}

func generateTableId2tableIndex(predicates []rds.Predicate) map[string]int {
	result := make(map[string]int)
	for _, predicate := range predicates {
		leftIndex, rightIndex := utils.GetPredicateColumnIndex(predicate)
		result[predicate.LeftColumn.TableId] = leftIndex / 2
		result[predicate.RightColumn.TableId] = rightIndex / 2
	}
	return result
}
