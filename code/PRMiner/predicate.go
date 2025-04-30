package main

import (
	"fmt"
	"golang.org/x/exp/maps"
	"rds-shenglin/rds_config"
	"rds-shenglin/rock-share/base/logger"
	"rds-shenglin/rock-share/global/enum"
	"rds-shenglin/rock-share/global/model/rds"
	"rds-shenglin/utils"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
)

func GenerateInnerPredicateNew(gv *GlobalV) {
	// 构建表内谓词
	var wg sync.WaitGroup
	predicateSupport := make(map[string]float64)
	predicateSupportLock := &sync.RWMutex{}
	// 遍历表
	for tableIndex, tableId := range gv.TablesId {
		tableColumnsType := gv.ColumnsType[tableId]
		rowSize := gv.RowSizes[tableId]
		// 遍历列
		for columnId, columnType := range maps.Clone(tableColumnsType) {
			// 判断是否需要跳过该列
			if _, ok := specialColumns[columnId]; ok {
				continue
			}
			<-TaskCh
			wg.Add(1)
			go func(columnId, columnType, tableId string, rowSize, tableIndex int) {
				defer func() {
					wg.Done()
					TaskCh <- struct{}{}
					if err := recover(); err != nil {
						gv.HasError = true
						s := string(debug.Stack())
						logger.Error("recover.err:%v, stack:%v", err, s)
					}
				}()
				cardinality := len(gv.IndexPLI[tableId][columnId])
				logger.Infof("tableId=%v, columnId=%s, cardinality=%d", tableId, columnId, cardinality)
				// 计算谓词的support
				nonConstPredSupport := 0
				for value, rows := range gv.IndexPLI[tableId][columnId] {
					if value == rds_config.NilIndex { //过滤掉=nil和=""的常数谓词
						continue
					}
					if len(rows) == 0 {
						continue
					}
					lines := len(rows)
					if lines > 1 {
						nonConstPredSupport += lines * (lines - 1)
					}
				}

				if columnType == rds_config.StringType {
					if cardinality < gv.EnumSize {
						columnType = rds_config.EnumType
					} else {
						columnType = rds_config.TextType
					}
					predicateSupportLock.Lock()
					gv.ColumnsType[tableId][columnId] = columnType
					predicateSupportLock.Unlock()
				}
				support := (float64(nonConstPredSupport)) / float64(rowSize*rowSize)

				if gv.Eids[tableId] == columnId {
					predicateStr := utils.GeneratePredicateStr(columnId, columnId, tableIndex*2, tableIndex*2+1)
					column := rds.Column{ColumnId: columnId, ColumnType: columnType, TableId: tableId}
					//trainDataColumn := generateTrainDataColumn(columnId, columnType, tableIndex, gv.PLI[tableId][columnId])
					symbolType := enum.Equal
					predicate := rds.Predicate{
						PredicateStr:  predicateStr,
						LeftColumn:    column,
						RightColumn:   column,
						ConstantValue: nil,
						SymbolType:    symbolType,
						PredicateType: 1,
						Support:       1,
						Intersection:  nil,
					}
					predicate.LeftColumn.ColumnIndex = tableIndex * 2
					predicate.RightColumn.ColumnIndex = tableIndex*2 + 1
					predicateSupportLock.Lock()
					gv.EidPredicates = append(gv.EidPredicates, predicate)
					predicateSupportLock.Unlock()
					return
				}

				// 当一个谓词的support高于某个阈值的时候,则认为该列对于规则发现没有意义(几乎全都相同)
				if columnType != rds_config.IndexType && support > gv.PredicateSupportLimit {
					//predicateSupportLock.Lock()
					//delete(gv.ColumnsType[tableId], columnId)
					//predicateSupportLock.Unlock()
					//logger.Infof("table:%v, column:%v, 的值基本都相同,不参与规则发现", tableId, columnId)
					//return

					// 陈胜林修改
					logger.Infof("table:%v, column:%v, 不进行筛选条件判断，直接处理", tableId, columnId)
					// 直接跳过条件判断，不再执行列的删除逻辑，或者其他筛选相关逻辑
					return
				}

				// 每一行的值都不相同
				if support <= 0 {
					//if columnType != rds_config.IntType && columnType != rds_config.FloatType && !gv.PrimaryKeys[columnId] {
					//	predicateSupportLock.Lock()
					//	delete(gv.ColumnsType[tableId], columnId)
					//	predicateSupportLock.Unlock()
					//	logger.Infof("table:%v, column:%v, 每行的值都不同, 且不为数值类型,不参与规则发现", tableId, columnId)
					//} else {
					//	logger.Infof("table:%v, column:%v, 每行的值都不同,不作为y出现", tableId, columnId)
					//}
					//return

					// 陈胜林修改
					logger.Infof("table:%v, column:%v, 不进行筛选条件判断，直接处理", tableId, columnId)
					// 直接跳过条件判断，不再执行列的删除逻辑，或者其他筛选相关逻辑
					return
				}

				column := rds.Column{ColumnId: columnId, ColumnType: columnType, TableId: tableId}
				var udfId int64
				var err error
				if columnType == rds_config.IndexType {
					if columnId[0] != udfColPrefix[0] {
						panic("代码有问题 " + columnId)
					}
					udfId, err = strconv.ParseInt(strings.Split(column.ColumnId[1:], udfColConn)[0], 10, 64)
					if err != nil {
						logger.Error(err)
					}
					column.ColumnId = strings.Split(columnId, rds_config.UdfColumnConn)[1]
				} else { // 相似度ML不做单行
					//生成决策树单行规则的Y列, 有些列不作为Y出现
					if (columnType == rds_config.IntType || columnType == rds_config.FloatType || columnType == rds_config.BoolType ||
						columnType == rds_config.EnumType) && cardinality > 1 {
						predicateSupportLock.Lock()
						if _, exist := gv.Table2DecisionY[tableId]; !exist {
							gv.Table2DecisionY[tableId] = make([]string, 0)
						}
						gv.Table2DecisionY[tableId] = append(gv.Table2DecisionY[tableId], column.ColumnId)
						predicateSupportLock.Unlock()
					}
				}

				// 生成表内的结构性谓词
				predicateStr := utils.GeneratePredicateStr(columnId, columnId, tableIndex*2, tableIndex*2+1)
				if columnType == rds_config.IndexType || support < gv.PredicateSupportLimit {
					trainDataColumn := generateTrainDataColumn(columnId, columnType, tableIndex, gv.PLI[tableId][columnId])
					symbolType := enum.Equal
					predicate := rds.Predicate{
						PredicateStr:  predicateStr,
						LeftColumn:    column,
						RightColumn:   column,
						ConstantValue: nil,
						SymbolType:    symbolType,
						PredicateType: 1,
						Support:       support,
						Intersection:  nil,
					}
					predicate.LeftColumn.ColumnIndex = tableIndex * 2
					predicate.RightColumn.ColumnIndex = tableIndex*2 + 1
					if columnType == rds_config.IndexType {
						if gv.UDFInfos[udfId].LeftTableId != gv.UDFInfos[udfId].RightTableId {
							return
						}
						predicate.SymbolType = gv.UDFInfos[udfId].Type
						predicate.UDFName = gv.UDFInfos[udfId].Name
						predicate.Threshold = gv.UDFInfos[udfId].Threshold
						predicate.PredicateStr = utils.GeneratePredicateStrNew(&predicate)
						predicate.LeftColumnVectorFilePath = gv.UDFInfos[udfId].LeftColumnVectorFilePath
						predicate.RightColumnVectorFilePath = gv.UDFInfos[udfId].RightColumnVectorFilePath
						predicate.UdfIndex = int(udfId)
					}
					predicateSupportLock.Lock()
					predicateSupport[predicateStr] = support
					gv.Predicates = append(gv.Predicates, predicate)
					for _, c := range trainDataColumn {
						gv.TrainDataColumnsType[tableId][c] = columnType
					}
					predicateSupportLock.Unlock()
				}
			}(columnId, columnType, tableId, rowSize, tableIndex)
		}
	}
	wg.Wait()
	if len(gv.Predicates) == 0 {
		return
	}
	logger.Infof("all predicates size:%v", len(gv.Predicates))
	logger.Infof("support: %v, confidence: %v", gv.Support, gv.Confidence)
}

func generateTrainDataColumn(columnName, columnType string, tableIndex int, values map[interface{}][]int32) []string {
	var result []string
	// 相似度ML不参与
	//if strings.HasPrefix(columnName, rds_config.UdfColumnPrefix) {
	if columnType == rds_config.IndexType {
		return result
	}
	// 时间类型暂时跳过
	if columnType == rds_config.TimeType {
		return result
	}
	// 文本类型暂时跳过
	if columnType == rds_config.TextType {
		return result
	}
	// 数值类型
	if columnType == rds_config.FloatType || columnType == rds_config.IntType {
		columnStr := fmt.Sprintf("t%d.%s", tableIndex*2, columnName)
		result = append(result, columnStr)
		columnStr = fmt.Sprintf("t%d.%s", tableIndex*2+1, columnName)
		result = append(result, columnStr)
	}
	// 枚举类型的列,抽样前获取枚举值
	if columnType == rds_config.EnumType || columnType == rds_config.BoolType {
		for value := range values {
			// null值不生成
			if value == nil {
				continue
			}
			valStr := ""
			if columnType == rds_config.BoolType {
				valStr = strconv.FormatBool(value.(bool))
			} else {
				valStr = value.(string)
			}
			columnStr := fmt.Sprintf("t%d.%s%s%s", tableIndex*2, columnName, rds_config.Equal, valStr)
			result = append(result, columnStr)
			columnStr = fmt.Sprintf("t%d.%s%s%s", tableIndex*2+1, columnName, rds_config.Equal, valStr)
			result = append(result, columnStr)
		}
	}
	return result
}
