package train_data_util

import (
	"fmt"
	"rds-shenglin/rock-share/global/model/rds"
	"rds-shenglin/rds_config"
	"math"
	"sort"
	"strconv"
	"strings"
)

type CacheValue[E any] struct {
	Val E
	_   [64]byte
}

func GenerateTrainData(satisfyData []map[string]map[string][]interface{}, tableNames []string, index2table map[string]string, dataType map[string]map[string]string, enableEnum, enableNum, enableText bool, rhs rds.Predicate, filterRatio map[string]float64, decisionTreeMaxRowSize int) ([]string, [][]float64, map[interface{}]float64, map[string]string) {
	var header []string
	var data [][]float64
	columnType := make(map[string]string)
	rhsValue2index := make(map[interface{}]float64)
	columnUniqueValue := make(map[string]map[string][]interface{})
	//t := time.Now().UnixMilli()
	if enableEnum {
		columnUniqueValue = getColumnUnique(satisfyData, dataType, index2table)
	}
	//logger.Infof("spent time:%vms, get column unique", time.Now().UnixMilli()-t)
	//eachBatchSize := rds_config.DecisionTreeMaxRowSize / len(satisfyData)

	dataSize := decisionTreeMaxRowSize - len(data)

	for i, idPairs := range satisfyData {
		var headerT []string
		var dataT [][]float64
		columnTypeT := make(map[string]string)
		rhsValue2indexT := make(map[interface{}]float64)
		// 单行规则
		if len(idPairs) == 1 {
			if rhs.SymbolType == rds_config.Poly { // 多项式规则
				polyStr, rhsRelatedColumns := generatePolyExpression(rhs.PredicateStr)
				headerT, dataT, rhsValue2indexT, columnTypeT = generatePolySingleRowTrainData(idPairs["t0"], dataType[tableNames[0]], polyStr, filterRatio["t0"], rhsRelatedColumns, columnUniqueValue)
			} else { // 普通单行规则
				headerT, dataT, rhsValue2indexT, columnTypeT = generateNormalSingleRowTrainData(idPairs["t0"], dataType[tableNames[0]], enableEnum, enableNum, enableText, rhs.LeftColumn.ColumnId, rhs.ConstantValue, filterRatio["t0"], columnUniqueValue)
			}
		} else { //多行规则
			headerT, dataT, rhsValue2indexT, columnTypeT = generateMultiRowTrainData(idPairs, tableNames, index2table, dataType, enableEnum, enableText, rhs, filterRatio, columnUniqueValue, dataSize)
		}
		//addSize := eachBatchSize
		//if len(dataT) < addSize {
		//	addSize = len(dataT)
		//}
		if i == 0 {
			header = headerT
			columnType = columnTypeT
			rhsValue2index = rhsValue2indexT
		}
		//data = append(data, dataT[:addSize]...)
		data = append(data, dataT...)
		dataSize = decisionTreeMaxRowSize - len(data)
	}
	return header, data, rhsValue2index, columnType
}

func generatePolySingleRowTrainData(satisfyData map[string][]interface{}, dataType map[string]string, rhsStr string, filterRatio float64, rhsColumns []string, columnUniqueValue map[string]map[string][]interface{}) ([]string, [][]float64, map[interface{}]float64, map[string]string) {
	if len(rhsColumns) > 1 {
		satisfyData = generatePolyData(satisfyData, rhsStr, rhsColumns)
		for _, columnName := range rhsColumns {
			delete(dataType, columnName)
		}
		dataType[rhsStr] = rds_config.FloatType
	}
	return generateNormalSingleRowTrainData(satisfyData, dataType, false, true, false, rhsStr, nil, filterRatio, columnUniqueValue)
}

func generateNormalSingleRowTrainData(satisfyData map[string][]interface{}, dataType map[string]string, enableEnum, enableNum, enableText bool, rhsColumn string, constantValue interface{}, filterRatio float64, columnUniqueValue map[string]map[string][]interface{}) ([]string, [][]float64, map[interface{}]float64, map[string]string) {
	resultColumns, resultType := generateSingleRowResultColumns(satisfyData, dataType, enableEnum, enableNum, enableText, rhsColumn, columnUniqueValue)
	rhsStr := fmt.Sprintf("t0.%s", rhsColumn)
	rhsValue2index := make(map[interface{}]float64)
	//if dataType[rhsColumn] == rds_config.EnumType {
	//	rhsStr = fmt.Sprintf("t0.%s%s%s", rhsColumn, rds_config.Equal, constantValue)
	//} else if dataType[rhsColumn] == rds_config.BoolType {
	//	rhsStr = fmt.Sprintf("t0.%s%s%s", rhsColumn, rds_config.Equal, strconv.FormatBool(constantValue.(bool)))
	//}
	resultColumns = append(resultColumns, rhsStr)
	resultType[rhsStr] = dataType[rhsColumn]
	rowSize := int(float64(len(satisfyData[rhsColumn])) * filterRatio)
	resultData := make([][]float64, rowSize)
	for i := 0; i < rowSize; i++ {
		resultData[i] = make([]float64, len(resultColumns))
	}
	for j := 0; j < len(resultColumns); {
		column := resultColumns[j]
		columnType := resultType[column]
		columnData := satisfyData[strings.Split(column, "t0.")[1]]
		if columnType == rds_config.FloatType && enableNum {
			for i := 0; i < rowSize; i++ {
				value := columnData[i]
				if value == nil || math.IsNaN(value.(float64)) || value == math.Inf(0) || value == math.Inf(1) {
					if column == rhsStr {
						resultData[i][j] = rds_config.DecisionTreeYNilReplace
					} else {
						resultData[i][j] = math.NaN()
					}
				} else {
					resultData[i][j] = value.(float64)
				}
			}
			j++
		} else if columnType == rds_config.IntType && enableNum {
			for i := 0; i < rowSize; i++ {
				if len(columnData) == 0 {
					println()
				}
				value := columnData[i]
				if value == nil || float64(value.(int64)) == math.Inf(0) || float64(value.(int64)) == math.Inf(1) {
					if column == rhsStr {
						resultData[i][j] = rds_config.DecisionTreeYNilReplace
					} else {
						resultData[i][j] = math.NaN()
					}
				} else {
					resultData[i][j] = float64(value.(int64))
				}
			}
			j++
		} else if (columnType == rds_config.EnumType || columnType == rds_config.BoolType) && enableEnum {
			// 作为rhs的时候只用考虑列名
			if column == rhsStr {
				uniqueData := columnUniqueValue["t0"][strings.Split(column, "t0.")[1]]
				uniqueData = sortEnumColumnValues(columnType, uniqueData)
				for i, datum := range uniqueData {
					rhsValue2index[datum] = float64(i)
				}
				for i := 0; i < rowSize; i++ {
					resultData[i][j] = rhsValue2index[columnData[i]]
				}
				j++
				continue
			}
			split := strings.Split(column, rds_config.Equal)
			baseColumn := split[0]
			columnData = satisfyData[strings.Split(baseColumn, "t0.")[1]]
			column = split[0]
			columnValue := split[1]
			k := 0
			for i := 0; i < rowSize; i++ {
				value := columnData[i]
				k = j
				split = strings.Split(resultColumns[k], rds_config.Equal)
				column = split[0]
				if column != baseColumn {
					break
				}
				columnValue = split[1]
				var valueStr string
				switch columnType {
				case rds_config.EnumType:
					if value == nil {
						value = ""
					}
					valueStr = value.(string)
				case rds_config.BoolType:
					if value == nil {
						value = false
					}
					valueStr = strconv.FormatBool(value.(bool))
				}
				for {
					if columnValue == valueStr {
						resultData[i][k] = float64(1)
					} else {
						resultData[i][k] = float64(0)
					}
					k++
					if k >= len(resultColumns) {
						break
					}
					split = strings.Split(resultColumns[k], rds_config.Equal)
					column = split[0]
					if column != baseColumn {
						break
					}
					columnValue = split[1]
				}
			}
			j = k
		} else if enableText {

		}
	}
	return resultColumns, resultData, rhsValue2index, resultType
}

func generateMultiRowTrainData(satisfyData map[string]map[string][]interface{}, tableNames []string, index2table map[string]string, allDataType map[string]map[string]string, enableEnum, enableText bool, rhs rds.Predicate, filterRatios map[string]float64, columnUniqueValue map[string]map[string][]interface{}, batchSize int) ([]string, [][]float64, map[interface{}]float64, map[string]string) {
	// 获取rhs对应的列和表索引
	leftIndex, leftColumn, rightIndex, rightColumn := getPredicateColumnInfoNew(rhs)

	// 生成训练集的表头
	resultColumns, resultType := generateHeader(satisfyData, index2table, allDataType, enableEnum, enableText, leftIndex, leftColumn, rightIndex, rightColumn, rhs, columnUniqueValue)

	// 把索引相关信息生成一个数组
	indexInfoArr, index2pos, resultRowSize := generateIndexInfoArr(satisfyData, filterRatios)
	if batchSize < resultRowSize {
		resultRowSize = batchSize
	}

	resultData := make([][]float64, resultRowSize)
	leftColumnData := satisfyData[leftIndex][leftColumn]
	rightColumnData := satisfyData[rightIndex][rightColumn]
	rhsType := allDataType[rhs.LeftColumn.TableId][rhs.LeftColumn.ColumnId]
	for i := 0; i < resultRowSize; i++ {
		resultData[i] = make([]float64, len(resultColumns))
		for j := 0; j < len(resultColumns)-1; j++ {
			columnType := resultType[resultColumns[j]]
			columnIndex, columnName, constantValue := getPredicateInfo(resultColumns[j])
			columnData := satisfyData[columnIndex][columnName][indexInfoArr[index2pos[columnIndex]].beginRow]
			switch columnType {
			case rds_config.IntType:
				if columnData == nil {
					resultData[i][j] = math.NaN()
				} else {
					resultData[i][j] = float64(columnData.(int64))
				}
			case rds_config.FloatType:
				if columnData == nil {
					resultData[i][j] = math.NaN()
				} else {
					resultData[i][j] = columnData.(float64)
				}

			case rds_config.EnumType:
				if enableEnum {
					if columnData == nil {
						columnData = ""
					}
					if columnData.(string) == constantValue {
						resultData[i][j] = float64(1)
					} else {
						resultData[i][j] = float64(0)
					}
				}
			case rds_config.BoolType:
				if enableEnum {
					if columnData == nil {
						columnData = false
					}
					if strconv.FormatBool(columnData.(bool)) == constantValue {
						resultData[i][j] = float64(1)
					} else {
						resultData[i][j] = float64(0)
					}
				}
			case rds_config.TextType:
				if enableText {

				}
			}
		}

		// rhs的值
		leftValue := leftColumnData[indexInfoArr[index2pos[leftIndex]].beginRow]
		rightValue := rightColumnData[indexInfoArr[index2pos[rightIndex]].beginRow]
		if rhsType == rds_config.FloatType {
			if leftValue == nil {
				resultData[i][len(resultColumns)-1] = 0
			} else {
				resultData[i][len(resultColumns)-1] = leftValue.(float64)
			}
		} else if rhsType == rds_config.IntType {
			if leftValue == nil {
				resultData[i][len(resultColumns)-1] = 0
			} else {
				resultData[i][len(resultColumns)-1] = float64(leftValue.(int64))
			}
		} else if leftValue == rightValue {
			resultData[i][len(resultColumns)-1] = float64(1)
		} else {
			resultData[i][len(resultColumns)-1] = float64(0)
		}

		indexInfoArr[0].beginRow++
		for j := 0; j < len(indexInfoArr)-1; j++ {
			indexInfoArr[j+1].beginRow += indexInfoArr[j].beginRow / indexInfoArr[j].endRow
			indexInfoArr[j].beginRow = indexInfoArr[j].beginRow % indexInfoArr[j].endRow
		}
	}

	return resultColumns, resultData, nil, resultType
}

func generateMultiRowTrainDataSingleTable(satisfyData map[string]map[string][]interface{}, allDataType map[string]map[string]string, enableEnum, enableText bool, rhs rds.Predicate, filterRatios map[string]float64, leftIndex, leftColumn, rightIndex, rightColumn string, resultColumns []string, resultType map[string]string) [][]float64 {

	// 把索引相关信息生成一个数组
	_, _, resultRowSize := generateIndexInfoArr(satisfyData, filterRatios)

	resultData := make([][]float64, resultRowSize)
	leftColumnData := satisfyData[leftIndex][leftColumn]
	rightColumnData := satisfyData[rightIndex][rightColumn]
	rhsType := allDataType[rhs.LeftColumn.TableId][rhs.LeftColumn.ColumnId]
	i := 0
	l0 := 0
	l1 := 0
	for _, tmp := range satisfyData["t0"] {
		l0 = len(tmp)
		break
	}
	for _, tmp := range satisfyData["t1"] {
		l1 = len(tmp)
		break
	}
	for p := 0; p < l0; p++ {
		for k := 0; k < l1; k++ {
			for j := 0; j < len(resultColumns)-1; j++ {
				columnType := resultType[resultColumns[j]]
				columnIndex, columnName, constantValue := getPredicateInfo(resultColumns[j])
				var columnData interface{}
				if columnIndex == "t0" {
					columnData = satisfyData[columnIndex][columnName][p]
				} else if columnIndex == "t1" {
					columnData = satisfyData[columnIndex][columnName][k]
				}
				switch columnType {
				case rds_config.IntType:
					if columnData == nil {
						resultData[i][j] = math.NaN()
					} else {
						resultData[i][j] = float64(columnData.(int64))
					}
				case rds_config.FloatType:
					if columnData == nil {
						resultData[i][j] = math.NaN()
					} else {
						resultData[i][j] = columnData.(float64)
					}

				case rds_config.EnumType:
					if enableEnum {
						if columnData == nil {
							columnData = ""
						}
						if columnData.(string) == constantValue {
							resultData[i][j] = float64(1)
						} else {
							resultData[i][j] = float64(0)
						}
					}
				case rds_config.BoolType:
					if enableEnum {
						if columnData == nil {
							columnData = false
						}
						if strconv.FormatBool(columnData.(bool)) == constantValue {
							resultData[i][j] = float64(1)
						} else {
							resultData[i][j] = float64(0)
						}
					}
				case rds_config.TextType:
					if enableText {

					}
				}
			}
			// rhs的值
			leftValue := leftColumnData[p]
			rightValue := rightColumnData[k]
			if rhsType == rds_config.FloatType {
				if leftValue == nil {
					resultData[i][len(resultColumns)-1] = 0
				} else {
					resultData[i][len(resultColumns)-1] = leftValue.(float64)
				}
			} else if rhsType == rds_config.IntType {
				if leftValue == nil {
					resultData[i][len(resultColumns)-1] = 0
				} else {
					resultData[i][len(resultColumns)-1] = float64(leftValue.(int64))
				}
			} else if leftValue == rightValue {
				resultData[i][len(resultColumns)-1] = float64(1)
			} else {
				resultData[i][len(resultColumns)-1] = float64(0)
			}
		}
		i++
	}

	return resultData
}

func GenerateTrainDataFromIntersection(intersection [][][]int32, index2table map[string]string, dataType map[string]map[string]string, enableEnum, enableText bool, rhs rds.Predicate, filterRatio map[string]float64, tablesValue map[string]map[string][]interface{}) ([]string, [][]float64, map[interface{}]float64, map[string]string) {
	var data [][]float64
	// 获取rhs对应的列和表索引
	leftIndex, leftColumn, rightIndex, rightColumn := getPredicateColumnInfoNew(rhs)
	header, columnsType := generateHeaderFromIntersection(intersection, dataType, index2table, tablesValue, enableEnum,
		enableText, rhs, leftIndex, leftColumn, rightIndex, rightColumn)

	leftColumnData := tablesValue[index2table[leftIndex]][leftColumn]
	rightColumnData := tablesValue[index2table[rightIndex]][rightColumn]
	rhsType := dataType[rhs.LeftColumn.TableId][rhs.LeftColumn.ColumnId]
	yIndex := len(header) - 1

	dataSize := rds_config.DecisionTreeMaxRowSize - len(data)

	for _, idPairs := range intersection {
		indexInfoArr, index2pos, resultRowSize := generateIndexInfoArrFromIdPairs(idPairs, filterRatio)
		if dataSize < resultRowSize {
			resultRowSize = dataSize
		}

		for i := 0; i < resultRowSize; i++ {
			rowData := make([]float64, len(header))
			for j := 0; j < yIndex; j++ {
				columnType := columnsType[header[j]]
				columnIndex, columnName, constantValue := getPredicateInfo(header[j])
				tableId := index2table[columnIndex]
				columnData := tablesValue[tableId][columnName][indexInfoArr[index2pos[columnIndex]].beginRow]
				switch columnType {
				case rds_config.IntType:
					if columnData == nil {
						rowData[j] = math.NaN()
					} else {
						rowData[j] = float64(columnData.(int64))
					}
				case rds_config.FloatType:
					if columnData == nil {
						rowData[j] = math.NaN()
					} else {
						rowData[j] = columnData.(float64)
					}
				case rds_config.EnumType:
					if enableEnum {
						if columnData == nil {
							columnData = ""
						}
						if columnData.(string) == constantValue {
							rowData[j] = float64(1)
						} else {
							rowData[j] = float64(0)
						}
					}
				case rds_config.BoolType:
					if enableEnum {
						if columnData == nil {
							columnData = false
						}
						if strconv.FormatBool(columnData.(bool)) == constantValue {
							rowData[j] = float64(1)
						} else {
							rowData[j] = float64(0)
						}
					}
				case rds_config.TextType:
					if enableText {

					}
				}
			}

			// rhs的值
			leftValue := leftColumnData[indexInfoArr[index2pos[leftIndex]].beginRow]
			rightValue := rightColumnData[indexInfoArr[index2pos[rightIndex]].beginRow]
			if rhsType == rds_config.FloatType {
				if leftValue == nil {
					rowData[yIndex] = 0
				} else {
					rowData[yIndex] = leftValue.(float64)
				}
			} else if rhsType == rds_config.IntType {
				if leftValue == nil {
					rowData[yIndex] = 0
				} else {
					rowData[yIndex] = float64(leftValue.(int64))
				}
			} else if leftValue == rightValue {
				rowData[yIndex] = float64(1)
			} else {
				rowData[yIndex] = float64(0)
			}

			indexInfoArr[0].beginRow++
			for j := 0; j < len(indexInfoArr)-1; j++ {
				indexInfoArr[j+1].beginRow += indexInfoArr[j].beginRow / indexInfoArr[j].endRow
				indexInfoArr[j].beginRow = indexInfoArr[j].beginRow % indexInfoArr[j].endRow
			}
			data = append(data, rowData)
		}

		dataSize = rds_config.DecisionTreeMaxRowSize - len(data)
		if dataSize < 1 {
			break
		}
	}
	return header, data, nil, columnsType
}

func GenerateTrainDataFromIntersectionNew(intersection [][][]int32, index2table map[string]string, dataType map[string]map[string]string, enableEnum, enableNum, enableText bool, rhs rds.Predicate, filterRatio map[string]float64, tablesValue map[string]map[string][]interface{}, header []string, columnsType map[string]string, decisionTreeMaxRowSize int) [][]float64 {
	var data [][]float64
	// 获取rhs对应的列和表索引
	leftIndex, leftColumn, rightIndex, rightColumn := getPredicateColumnInfoNew(rhs)
	leftIndexI, _ := strconv.Atoi(leftIndex[1:])
	rightIndexI, _ := strconv.Atoi(rightIndex[1:])

	leftColumnData := tablesValue[index2table[leftIndex]][leftColumn]
	rightColumnData := tablesValue[index2table[rightIndex]][rightColumn]
	//rhsType := dataType[rhs.LeftColumn.TableId][rhs.LeftColumn.ColumnId]
	yIndex := len(header) - 1

	var indexArr []string
	for index := range index2table {
		indexArr = append(indexArr, index)
	}
	sort.Strings(indexArr)

	dataSize := decisionTreeMaxRowSize - len(data)

	for _, idPairs := range intersection {
		indexInfoMap, resultRowSize := generateIndexInfoArrFromIdPairsNew(idPairs, filterRatio, leftIndexI, rightIndexI, len(leftColumnData), len(rightColumnData))
		if len(indexInfoMap) < 1 {
			continue
		}
		if dataSize < resultRowSize {
			resultRowSize = dataSize
		}

		for i := 0; i < resultRowSize; i++ {
			rowData := make([]float64, len(header))
			for j := 0; j < yIndex; j++ {
				columnType := columnsType[header[j]]
				columnIndex, columnName, constantValue := getPredicateInfo(header[j])
				tableId := index2table[columnIndex]
				//columnData := tablesValue[tableId][columnName][indexInfoMap[columnIndex].beginRow]
				columnIndexI, _ := strconv.Atoi(columnIndex[1:])
				columnData := tablesValue[tableId][columnName][idPairs[columnIndexI][indexInfoMap[columnIndex].beginRow]]
				switch columnType {
				case rds_config.IntType:
					if enableNum {
						if columnData == nil {
							rowData[j] = math.NaN()
						} else {
							rowData[j] = float64(columnData.(int64))
						}
					}
				case rds_config.FloatType:
					if enableNum {
						if columnData == nil {
							rowData[j] = math.NaN()
						} else {
							rowData[j] = columnData.(float64)
						}
					}
				case rds_config.EnumType:
					if enableEnum {
						if columnData == nil {
							columnData = ""
						}
						if columnData.(string) == constantValue {
							rowData[j] = float64(1)
						} else {
							rowData[j] = float64(0)
						}
					}
				case rds_config.BoolType:
					if enableEnum {
						if columnData == nil {
							columnData = false
						}
						if strconv.FormatBool(columnData.(bool)) == constantValue {
							rowData[j] = float64(1)
						} else {
							rowData[j] = float64(0)
						}
					}
				case rds_config.TextType:
					if enableText {

					}
				}
			}

			// rhs的值
			//leftValue := leftColumnData[indexInfoMap[leftIndex].beginRow]
			//rightValue := rightColumnData[indexInfoMap[rightIndex].beginRow]
			var leftValue, rightValue interface{}
			if leftIndexI >= len(idPairs) || idPairs[leftIndexI] == nil {
				leftValue = leftColumnData[indexInfoMap[leftIndex].beginRow]
			} else {
				leftValue = leftColumnData[idPairs[leftIndexI][indexInfoMap[leftIndex].beginRow]]
			}
			if rightIndexI >= len(idPairs) || idPairs[rightIndexI] == nil {
				rightValue = rightColumnData[indexInfoMap[rightIndex].beginRow]
			} else {
				rightValue = rightColumnData[idPairs[rightIndexI][indexInfoMap[rightIndex].beginRow]]
			}
			//leftValue := leftColumnData[idPairs[leftIndexI][indexInfoMap[leftIndex].beginRow]]
			//rightValue := rightColumnData[idPairs[rightIndexI][indexInfoMap[rightIndex].beginRow]]
			//if rhsType == rds_config.FloatType {
			//	if leftValue == nil {
			//		rowData[yIndex] = 0
			//	} else {
			//		rowData[yIndex] = leftValue.(float64)
			//	}
			//} else if rhsType == rds_config.IntType {
			//	if leftValue == nil {
			//		rowData[yIndex] = 0
			//	} else {
			//		rowData[yIndex] = float64(leftValue.(int64))
			//	}
			//} else if leftValue == rightValue {
			//	rowData[yIndex] = float64(1)
			//} else {
			//	rowData[yIndex] = float64(0)
			//}
			if leftValue == rightValue {
				rowData[yIndex] = float64(1)
			} else {
				rowData[yIndex] = float64(0)
			}

			temp := indexInfoMap[indexArr[0]]
			temp.beginRow++
			for j := 0; j < len(indexArr)-1; j++ {
				indexInfoMap[indexArr[j+1]].beginRow += indexInfoMap[indexArr[j]].beginRow / indexInfoMap[indexArr[j]].endRow
				indexInfoMap[indexArr[j]].beginRow = indexInfoMap[indexArr[j]].beginRow % indexInfoMap[indexArr[j]].endRow
			}
			data = append(data, rowData)
		}

		dataSize = decisionTreeMaxRowSize - len(data)
		if dataSize < 1 {
			break
		}
	}
	return data
}
