package main

import (
	"fmt"
	"github.com/yourbasic/bit"
	"rds-shenglin/rds_config"
	"rds-shenglin/rock-share/base/logger"
	"rds-shenglin/rock-share/global/model/rds"
	"rds-shenglin/utils"
	"sort"
	"strconv"
	"time"
)

func ExecuteCheckErrorRds(rule rds.Rule, gv *GlobalV) (support float64, confidence float64, conflictDataAgg map[string]map[int32]int32, conflictSize int) {
	logger.Infof("task id:%v, 执行查错:%v", gv.TaskId, rule.Ree)
	startTime := time.Now().UnixMilli()
	taskId := gv.TaskId
	sortLhs := utils.SortPredicatesRelated(rule.LhsPredicates)
	intersections := SampleIntersection(sortLhs, 0, taskId, false)

	lTid, rTid := utils.GetPredicateColumnIndex(rule.Rhs)
	if len(intersections) < 1 {
		return 0, 0, conflictDataAgg, 0
	}

	mlPredicates := utils.GetMLPredicate(rule.LhsPredicates)
	var xSupp int
	var xySupp int

	for _, intersection := range intersections {
		lRows := intersection[lTid]
		var rRows []int32
		if rTid >= 0 {
			rRows = intersection[rTid]
		}
		var opposite int
		var positive int
		var conflictData map[string]map[int32]int32
		if rule.RuleType == 5 { // best
			//opposite, positive, conflictData = calcBestOpposite(lRows, rRows, rule.Rhs, gv)
		} else if rule.RuleType == 4 {
			if len(mlPredicates) == 0 {
				opposite, positive, conflictData = calcOpposite3(lRows, rRows, rule.Rhs, gv)
			}
		} else {
			if len(mlPredicates) == 0 {
				opposite, positive, conflictData = calcOpposite2(lRows, rRows, rule.Rhs, gv)
			}
		}
		conflictSize += opposite
		xySupp += positive

		if conflictDataAgg == nil {
			conflictDataAgg = conflictData
		} else {
			for t, m := range conflictData {
				ov, ok := conflictDataAgg[t]
				if !ok {
					conflictDataAgg[t] = m
				} else {
					for r, c := range m {
						ov[r] += c
					}
				}
			}
		}
	}

	if len(mlPredicates) == 0 {
		xSupp = getXSuppNew(sortLhs, intersections)
	}
	isConsistent := (xSupp - xySupp) == conflictSize
	if !isConsistent {
		logger.Warnf("[ExecuteCheckError] taskId:%d, (xSupp:%d - xySupp:%d) != conflict:%d", taskId, xSupp, xySupp, conflictSize)
		xSupp = xySupp + conflictSize
	}
	rowSize := getRowSize(gv, sortLhs)

	if rowSize > 0 {
		support = float64(xySupp) / float64(rowSize)
	}
	if xSupp > 0 {
		confidence = float64(xySupp) / float64(xSupp)
	}
	// 重新查错时需要更新
	rule.CR = support
	rule.FTR = confidence

	totalTime := time.Now().UnixMilli() - startTime
	logger.Infof("[ExecuteCheckError] taskId:%d, finish check error, totalTime:%d(ms), rowSize:%d|xSupp:%d|xySupp:%d|conflict:%d",
		taskId, totalTime, rowSize, xSupp, xySupp, conflictSize)
	return support, confidence, conflictDataAgg, conflictSize
}

func getColumnIndexValuesGroupBy(rowIds []int32, tableId string, column string, gv *GlobalV) (map[int32]int, map[int32]*bit.Set) {
	columValues := gv.TableIndexValues[tableId][column]
	groupBy := make(map[int32]int)
	value2RowSet := make(map[int32]*bit.Set)
	for _, rowId := range rowIds {
		indexValue := columValues[rowId]
		if count, ok := groupBy[indexValue]; ok {
			groupBy[indexValue] = count + 1
			value2RowSet[indexValue] = value2RowSet[indexValue].Add(int(rowId))
		} else {
			groupBy[indexValue] = 1
			value2RowSet[indexValue] = bit.New(int(rowId))
		}
	}
	return groupBy, value2RowSet
}

// calcOpposite3 最小的不是冲突
func calcOpposite3(lRows []int32, rRows []int32, rhs rds.Predicate, gv *GlobalV) (opposite int, positive int, conflictData map[string]map[int32]int32) {
	conflictData = make(map[string]map[int32]int32)
	if rhs.PredicateType == 0 { // 不应该进来
		tableId := rhs.LeftColumn.TableId
		column := rhs.LeftColumn.ColumnId
		constValueRowIds := gv.PLI[tableId][column][rhs.ConstantValue]
		if len(constValueRowIds) == 0 {
			return 0, 0, conflictData
		}
		colIndexValues := gv.TableIndexValues[tableId][column]
		constantIndexValue := colIndexValues[constValueRowIds[0]]
		conflict := make(map[int32]int32, len(lRows))
		for _, rowId := range lRows {
			currentIndexValue := colIndexValues[rowId]
			if currentIndexValue == constantIndexValue {
				positive++
			} else {
				conflict[rowId] = 1
				opposite++
			}
		}
		conflictData[tableId] = conflict
	} else { // multi row
		lRowNum := len(lRows)
		rRowNum := len(rRows)
		if lRowNum == 0 || rRowNum == 0 {
			return 0, 0, conflictData
		}
		lTableId := rhs.LeftColumn.TableId
		rTableId := rhs.RightColumn.TableId
		if lTableId == rTableId { // same table
			var realValueRowIdsMap = map[any]map[int32]struct{}{} // value -> rows
			var lRealValues = gv.TableRealValues[lTableId][rhs.LeftColumn.ColumnId]
			var rRealValues = gv.TableRealValues[rTableId][rhs.RightColumn.ColumnId]
			var totalRowIds = map[int32]struct{}{}
			for _, lRowId := range lRows {
				var value = lRealValues[lRowId]
				if realValueRowIdsMap[value] == nil {
					realValueRowIdsMap[value] = map[int32]struct{}{}
				}
				realValueRowIdsMap[value][lRowId] = struct{}{}
				totalRowIds[lRowId] = struct{}{}
			}
			for _, rRowId := range rRows {
				var value = rRealValues[rRowId]
				if realValueRowIdsMap[value] == nil {
					realValueRowIdsMap[value] = map[int32]struct{}{}
				}
				realValueRowIdsMap[value][rRowId] = struct{}{}
				totalRowIds[rRowId] = struct{}{}
			}
			if len(realValueRowIdsMap) == 0 {
				return 0, 0, conflictData
			} else if len(realValueRowIdsMap) == 1 { // only one
				for value, rowIds := range realValueRowIdsMap {
					if value == nil || value == "" { // all is conflict
						conflict := make(map[int32]int32, rRowNum)
						for rowId := range rowIds {
							conflict[rowId] = int32(len(rowIds) - 1)
						}
						conflictData[lTableId] = conflict
						return 0, len(rowIds) * (len(rowIds) - 1), conflictData
					} else { // all is not conflict
						return 0, len(rowIds) * (len(rowIds) - 1), conflictData
					}
				}
				return 0, 0, conflictData // no reach
			}
			var valueRowIds = utils.MapIKVs(realValueRowIdsMap)
			sort.Slice(valueRowIds, func(i, j int) bool { // desc
				var vi, vj float64
				var ok bool
				if valueRowIds[i].Key == nil || valueRowIds[i].Key == "" {
					vi = 10e10
				} else {
					vi, ok = toFloat64(valueRowIds[i].Key)
					if !ok {
						logger.Warnf("number error %v", valueRowIds[i].Key)
					}
				}
				if valueRowIds[j].Key == nil || valueRowIds[j].Key == "" {
					vi = 10e10
				} else {
					vj, ok = toFloat64(valueRowIds[j].Key)
					if !ok {
						logger.Warnf("number error %v", valueRowIds[j].Key)
					}
				}
				return vi < vj
			})
			var conflict = make(map[int32]int32, rRowNum)
			conflictData[lTableId] = conflict

			var nilNumber = len(realValueRowIdsMap[nil])
			for i, indexValueRowIds := range valueRowIds {
				value := indexValueRowIds.Key
				rowIds := indexValueRowIds.Value
				if value == nil || value == "" { // all is conflict
					for rowId := range rowIds {
						opposite += 0
						conflict[rowId] = int32(len(totalRowIds) - 1)
					}
				} else { // skip self conflict
					for rowId := range rowIds {
						positive += len(rowIds) - 1 // self
						opposite += len(totalRowIds) - len(rowIds) - nilNumber
						if i == 0 { // first and skip conflictData
						} else {
							conflict[rowId] = int32(len(totalRowIds) - len(rowIds))
						}
					}
				}
			}
			return opposite, positive, conflictData
		} else { // 不应该进来
			// multi table
			// 1000085(t0)^1000086(t2) ^ t0.1001441=t2.1001447->t0.1001442=t2.1001448
			var valueRowIdsMap = map[int32][2]map[int32]struct{}{} // index -> [l|r] -> rowIds
			var lTotalRowIds, rTotalRowIds = map[int32]struct{}{}, map[int32]struct{}{}
			var lIndexValues = gv.TableIndexValues[lTableId][rhs.LeftColumn.ColumnId]
			var rIndexValues = gv.TableIndexValues[rTableId][rhs.RightColumn.ColumnId]
			for _, lRowId := range lRows {
				var indexValue = lIndexValues[lRowId]
				if valueRowIdsMap[indexValue][0] == nil {
					valueRowIdsMap[indexValue] = [2]map[int32]struct{}{make(map[int32]struct{}), make(map[int32]struct{})}
				}
				valueRowIdsMap[indexValue][0][lRowId] = struct{}{}
				lTotalRowIds[lRowId] = struct{}{}
			}
			for _, rRowId := range rRows {
				var indexValue = rIndexValues[rRowId]
				if valueRowIdsMap[indexValue][1] == nil {
					valueRowIdsMap[indexValue] = [2]map[int32]struct{}{make(map[int32]struct{}), make(map[int32]struct{})}
				}
				valueRowIdsMap[indexValue][1][rRowId] = struct{}{}
				rTotalRowIds[rRowId] = struct{}{}
			}
			if len(valueRowIdsMap) == 0 {
				return 0, 0, conflictData
			} else if len(valueRowIdsMap) == 1 { // only one
				for indexValue, lrRowIds := range valueRowIdsMap {
					if len(lrRowIds[0]) > 0 && len(lrRowIds[1]) > 0 {
						if indexValue == rds_config.NilIndex { // all conflict
							opposite = len(lrRowIds[0]) * len(lrRowIds[1])
							var lConflict, rConflict = make(map[int32]int32), make(map[int32]int32)
							conflictData[lTableId] = lConflict
							conflictData[rTableId] = rConflict
							for lRowId := range lrRowIds[0] {
								lConflict[lRowId] = int32(len(lrRowIds[1]))
							}
							for rRowId := range lrRowIds[1] {
								rConflict[rRowId] = int32(len(lrRowIds[0]))
							}
						} else { // all same
							positive = len(lrRowIds[1]) * len(lrRowIds[0])
						}
					}
				}
				return opposite, positive, conflictData
			}
			// sort
			var valueRowIds = utils.MapKVs(valueRowIdsMap)
			sort.Slice(valueRowIds, func(i, j int) bool {
				return len(valueRowIds[i].Value[0])+len(valueRowIds[i].Value[1]) > len(valueRowIds[j].Value[0])+len(valueRowIds[j].Value[1])
			})
			var maxIsNil = valueRowIds[0].Key == rds_config.NilIndex
			var maxAndNextIsSame = len(valueRowIds[0].Value[0])+len(valueRowIds[0].Value[1]) == len(valueRowIds[1].Value[0])+len(valueRowIds[1].Value[1])
			var lConflict, rConflict = make(map[int32]int32), make(map[int32]int32)
			conflictData[lTableId] = lConflict
			conflictData[rTableId] = rConflict

			for i, indexValueRowIds := range valueRowIds {
				indexValue := indexValueRowIds.Key
				lrRowIds := indexValueRowIds.Value
				if indexValue == rds_config.NilIndex { // all is conflict
					opposite += len(lrRowIds[0]) * len(rTotalRowIds)
					for lRowId := range lrRowIds[0] {
						if i == 0 && (!(maxIsNil || maxAndNextIsSame)) { // first and skip
						} else {
							lConflict[lRowId] = int32(len(rTotalRowIds))
						}
					}
					for rRowId := range lrRowIds[1] {
						if i == 0 && (!(maxIsNil || maxAndNextIsSame)) { // first and skip
						} else {
							rConflict[rRowId] = int32(len(lTotalRowIds))
						}
					}
				} else { // skip same
					positive += len(lrRowIds[0]) * len(valueRowIdsMap[indexValue][1])
					opposite += len(lrRowIds[0]) * (len(rTotalRowIds) - len(valueRowIdsMap[indexValue][1]))
					for lRowId := range lrRowIds[0] {
						if i == 0 && (!(maxIsNil || maxAndNextIsSame)) { // first and skip
						} else {
							lConflict[lRowId] = int32(len(rTotalRowIds) - len(valueRowIdsMap[indexValue][1]))
						}
					}
					for rRowId := range lrRowIds[1] {
						if i == 0 && (!(maxIsNil || maxAndNextIsSame)) { // first and skip
						} else {
							rConflict[rRowId] = int32(len(lTotalRowIds) - len(valueRowIdsMap[indexValue][0]))
						}
					}
				}
			}
			return opposite, positive, conflictData
		}
	}
	return opposite, positive, conflictData
}

// calcOpposite2 出现次数最多不是冲突
func calcOpposite2(lRows []int32, rRows []int32, rhs rds.Predicate, gv *GlobalV) (opposite int, positive int, conflictData map[string]map[int32]int32) {
	conflictData = make(map[string]map[int32]int32)
	if rhs.PredicateType == 0 {
		tableId := rhs.LeftColumn.TableId
		column := rhs.LeftColumn.ColumnId
		constValueRowIds := gv.PLI[tableId][column][rhs.ConstantValue]
		if len(constValueRowIds) == 0 {
			return 0, 0, conflictData
		}
		colIndexValues := gv.TableIndexValues[tableId][column]
		constantIndexValue := colIndexValues[constValueRowIds[0]]
		conflict := make(map[int32]int32, len(lRows))
		for _, rowId := range lRows {
			currentIndexValue := colIndexValues[rowId]
			if currentIndexValue == constantIndexValue {
				positive++
			} else {
				conflict[rowId] = 1
				opposite++
			}
		}
		conflictData[tableId] = conflict
	} else { // multi row
		lRowNum := len(lRows)
		rRowNum := len(rRows)
		if lRowNum == 0 || rRowNum == 0 {
			return 0, 0, conflictData
		}
		lTableId := rhs.LeftColumn.TableId
		rTableId := rhs.RightColumn.TableId
		if lTableId == rTableId { // same table
			var valueRowIdsMap = map[int32]map[int32]struct{}{} // statistics index->rowIdSet
			var lIndexValues = gv.TableIndexValues[lTableId][rhs.LeftColumn.ColumnId]
			var rIndexValues = gv.TableIndexValues[rTableId][rhs.RightColumn.ColumnId]
			var totalRowIds = map[int32]struct{}{}
			for _, lRowId := range lRows {
				var indexValue = lIndexValues[lRowId]
				if valueRowIdsMap[indexValue] == nil {
					valueRowIdsMap[indexValue] = map[int32]struct{}{}
				}
				valueRowIdsMap[indexValue][lRowId] = struct{}{}
				totalRowIds[lRowId] = struct{}{}
			}
			for _, rRowId := range rRows {
				var indexValue = rIndexValues[rRowId]
				if valueRowIdsMap[indexValue] == nil {
					valueRowIdsMap[indexValue] = map[int32]struct{}{}
				}
				valueRowIdsMap[indexValue][rRowId] = struct{}{}
				totalRowIds[rRowId] = struct{}{}
			}
			if len(valueRowIdsMap) == 0 {
				return 0, 0, conflictData
			} else if len(valueRowIdsMap) == 1 { // only one
				for indexValue, rowIds := range valueRowIdsMap {
					if indexValue == rds_config.NilIndex { // all is conflict
						conflict := make(map[int32]int32, rRowNum)
						for rowId := range rowIds {
							conflict[rowId] = int32(len(rowIds) - 1)
						}
						conflictData[lTableId] = conflict
						return len(rowIds) * (len(rowIds) - 1), 0, conflictData
					} else { // all is not conflict
						return 0, len(rowIds) * (len(rowIds) - 1), conflictData
					}
				}
				return 0, 0, conflictData // no reach
			}
			var valueRowIds = utils.MapKVs(valueRowIdsMap)
			sort.Slice(valueRowIds, func(i, j int) bool { // desc
				return len(valueRowIds[i].Value) > len(valueRowIds[j].Value)
			})
			var maxIsNil = valueRowIds[0].Key == rds_config.NilIndex
			var maxAndNextIsSame = len(valueRowIds[0].Value) == len(valueRowIds[1].Value)
			var conflict = make(map[int32]int32, rRowNum)
			conflictData[lTableId] = conflict

			for i, indexValueRowIds := range valueRowIds {
				indexValue := indexValueRowIds.Key
				rowIds := indexValueRowIds.Value
				if indexValue == rds_config.NilIndex { // all is conflict
					for rowId := range rowIds {
						opposite += len(totalRowIds) - 1
						if i == 0 && (!(maxIsNil || maxAndNextIsSame)) { // first and skip conflictData
						} else {
							conflict[rowId] = int32(len(totalRowIds) - 1)
						}
					}
				} else { // skip self conflict
					for rowId := range rowIds {
						positive += len(rowIds) - 1 // self
						opposite += len(totalRowIds) - len(rowIds)
						if i == 0 && (!(maxIsNil || maxAndNextIsSame)) { // first and skip conflictData
						} else {
							conflict[rowId] = int32(len(totalRowIds) - len(rowIds))
						}
					}
				}
			}
			return opposite, positive, conflictData
		} else {
			// multi table
			// 1000085(t0)^1000086(t2) ^ t0.1001441=t2.1001447->t0.1001442=t2.1001448
			var valueRowIdsMap = map[int32][2]map[int32]struct{}{} // index -> [l|r] -> rowIds
			var lTotalRowIds, rTotalRowIds = map[int32]struct{}{}, map[int32]struct{}{}
			var lIndexValues = gv.TableIndexValues[lTableId][rhs.LeftColumn.ColumnId]
			var rIndexValues = gv.TableIndexValues[rTableId][rhs.RightColumn.ColumnId]
			for _, lRowId := range lRows {
				var indexValue = lIndexValues[lRowId]
				if valueRowIdsMap[indexValue][0] == nil {
					valueRowIdsMap[indexValue] = [2]map[int32]struct{}{make(map[int32]struct{}), make(map[int32]struct{})}
				}
				valueRowIdsMap[indexValue][0][lRowId] = struct{}{}
				lTotalRowIds[lRowId] = struct{}{}
			}
			for _, rRowId := range rRows {
				var indexValue = rIndexValues[rRowId]
				if valueRowIdsMap[indexValue][1] == nil {
					valueRowIdsMap[indexValue] = [2]map[int32]struct{}{make(map[int32]struct{}), make(map[int32]struct{})}
				}
				valueRowIdsMap[indexValue][1][rRowId] = struct{}{}
				rTotalRowIds[rRowId] = struct{}{}
			}
			if len(valueRowIdsMap) == 0 {
				return 0, 0, conflictData
			} else if len(valueRowIdsMap) == 1 { // only one
				for indexValue, lrRowIds := range valueRowIdsMap {
					if len(lrRowIds[0]) > 0 && len(lrRowIds[1]) > 0 {
						if indexValue == rds_config.NilIndex { // all conflict
							opposite = len(lrRowIds[0]) * len(lrRowIds[1])
							var lConflict, rConflict = make(map[int32]int32), make(map[int32]int32)
							conflictData[lTableId] = lConflict
							conflictData[rTableId] = rConflict
							for lRowId := range lrRowIds[0] {
								lConflict[lRowId] = int32(len(lrRowIds[1]))
							}
							for rRowId := range lrRowIds[1] {
								rConflict[rRowId] = int32(len(lrRowIds[0]))
							}
						} else { // all same
							positive = len(lrRowIds[1]) * len(lrRowIds[0])
						}
					}
				}
				return opposite, positive, conflictData
			}
			// sort
			var valueRowIds = utils.MapKVs(valueRowIdsMap)
			sort.Slice(valueRowIds, func(i, j int) bool {
				return len(valueRowIds[i].Value[0])+len(valueRowIds[i].Value[1]) > len(valueRowIds[j].Value[0])+len(valueRowIds[j].Value[1])
			})
			var maxIsNil = valueRowIds[0].Key == rds_config.NilIndex
			var maxAndNextIsSame = len(valueRowIds[0].Value[0])+len(valueRowIds[0].Value[1]) == len(valueRowIds[1].Value[0])+len(valueRowIds[1].Value[1])
			var lConflict, rConflict = make(map[int32]int32), make(map[int32]int32)
			conflictData[lTableId] = lConflict
			conflictData[rTableId] = rConflict

			for i, indexValueRowIds := range valueRowIds {
				indexValue := indexValueRowIds.Key
				lrRowIds := indexValueRowIds.Value
				if indexValue == rds_config.NilIndex { // all is conflict
					opposite += len(lrRowIds[0]) * len(rTotalRowIds)
					for lRowId := range lrRowIds[0] {
						if i == 0 && (!(maxIsNil || maxAndNextIsSame)) { // first and skip
						} else {
							lConflict[lRowId] = int32(len(rTotalRowIds))
						}
					}
					for rRowId := range lrRowIds[1] {
						if i == 0 && (!(maxIsNil || maxAndNextIsSame)) { // first and skip
						} else {
							rConflict[rRowId] = int32(len(lTotalRowIds))
						}
					}
				} else { // skip same
					positive += len(lrRowIds[0]) * len(valueRowIdsMap[indexValue][1])
					opposite += len(lrRowIds[0]) * (len(rTotalRowIds) - len(valueRowIdsMap[indexValue][1]))
					for lRowId := range lrRowIds[0] {
						if i == 0 && (!(maxIsNil || maxAndNextIsSame)) { // first and skip
						} else {
							lConflict[lRowId] = int32(len(rTotalRowIds) - len(valueRowIdsMap[indexValue][1]))
						}
					}
					for rRowId := range lrRowIds[1] {
						if i == 0 && (!(maxIsNil || maxAndNextIsSame)) { // first and skip
						} else {
							rConflict[rRowId] = int32(len(lTotalRowIds) - len(valueRowIdsMap[indexValue][0]))
						}
					}
				}
			}
			return opposite, positive, conflictData
		}
	}
	return opposite, positive, conflictData
}

func toFloat64(a any) (float64, bool) {
	if a == nil {
		return 0, false
	}
	if f, ok := a.(float64); ok {
		return f, true
	}
	if b, ok := a.(time.Time); ok {
		return float64(b.UnixMilli()), true
	}
	f, err := strconv.ParseFloat(fmt.Sprintf("%v", a), 64)
	if err != nil {
		return 0, false
	}
	return f, true
}

func getXSuppNew(lhs []rds.Predicate, intersection [][][]int32) (xSupp int) {
	var usedTableIndex []int
	for _, lh := range lhs {
		lTid, rTid := utils.GetPredicateColumnIndex(lh)
		usedTableIndex = append(usedTableIndex, lTid)
		if rTid != -1 {
			usedTableIndex = append(usedTableIndex, rTid)
		}
	}
	usedTableIndex = utils.Distinct(usedTableIndex)
	return CalIntersectionSupport(intersection, usedTableIndex)
}

func getRowSize(gv *GlobalV, lhs []rds.Predicate) int {
	var rowSize int
	tableIndex2tableId := GetTableIndex2tableId(lhs)
	rowSize = 1
	for tid, tableId := range tableIndex2tableId {
		if tid == -1 {
			continue
		}
		rowSize *= gv.RowSizes[tableId]
	}
	return rowSize
}
