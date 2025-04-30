package predicate

import (
	"fmt"
	"rds-shenglin/rock-share/base/logger"
	"rds-shenglin/rock-share/global/model/rds"
	"rds-shenglin/rds_config"
	"rds-shenglin/utils"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const (
	RealColumn = iota
	SingleDeriveColumn
	MultiDeriveColumn
)

type IDecisionTreePredicatesUtil interface {
	GeneratePredicates(index2Table map[string]string, colName string, columnType string, constantValue interface{}, lhs []interface{}, yFlag bool, transferColNameFlag bool) (rds.Predicate, error)
	ParseColumnName(index2Table map[string]string, colName string, columnType string) (columns []string, symbol string, constantValue interface{}, tableIds []string, tableIndexes []string, columnIndexes []int, err error)
}

type RealColumnPredicatesUtil struct {
}

type SingleDeriveColumnPredicatesUtil struct {
}

type MultiDeriveColumnPredicatesUtil struct {
}

func GetDecisionTreePredicatesUtil(deriveColumnType int) IDecisionTreePredicatesUtil {
	switch deriveColumnType {
	case RealColumn:
		return RealColumnPredicatesUtil{}
	case SingleDeriveColumn:
		return SingleDeriveColumnPredicatesUtil{}
	case MultiDeriveColumn:
		return MultiDeriveColumnPredicatesUtil{}
	default:
		return nil
	}
}

// 只有数值类型会
func (util RealColumnPredicatesUtil) GeneratePredicates(index2Table map[string]string, colName string, columnType string, constantValue interface{}, lhs []interface{}, yFlag bool, transferColNameFlag bool) (rds.Predicate, error) {
	index := colName[:strings.Index(colName, ".")]
	columnIndexStr := index[1:]
	columnIndex, err := strconv.Atoi(columnIndexStr)
	if err != nil {
		logger.Errorf("解析%s中的ColumnIndex失败", columnIndexStr, err)
		return rds.Predicate{}, err
	}
	columnName := colName[strings.Index(colName, ".")+1:]
	tableId := index2Table[index]
	relValue := constantValue
	if columnType == rds_config.IntType {
		var err error
		relValue, err = strconv.ParseInt(utils.GetInterfaceToString(constantValue), 10, 64)
		if err != nil {
			logger.Error("column " + colName + ", parse constantValue " + fmt.Sprint(constantValue) + " to " + columnType + " type error")
			return rds.Predicate{}, err
		}
	}

	predicate := rds.Predicate{
		PredicateStr:  fmt.Sprintf("%s%s%s", lhs[0], lhs[1], utils.GetInterfaceToString(lhs[2])),
		LeftColumn:    rds.Column{ColumnIndex: columnIndex, ColumnId: columnName, ColumnType: columnType, TableId: tableId},
		RightColumn:   rds.Column{},
		ConstantValue: relValue,
		SymbolType:    utils.GetInterfaceToString(lhs[1]),
		PredicateType: 0,
	}
	return predicate, nil
}

func (util SingleDeriveColumnPredicatesUtil) GeneratePredicates(index2Table map[string]string, colName string, columnType string, constantValue interface{}, lhs []interface{}, yFlag bool, transferColNameFlag bool) (rds.Predicate, error) {
	realColumnNames, symbol, value, tableIds, tableIndexes, columnIndexes, err := util.ParseColumnName(index2Table, colName, columnType)

	predicateStr := colName
	if transferColNameFlag {
		if columnType == rds_config.EnumType || columnType == rds_config.TextType || columnType == rds_config.BoolType {
			symbol = rds_config.NotEqual
			predicateStr = fmt.Sprintf("%s.%s%s%v", tableIndexes[0], realColumnNames[0], symbol, value)
		} else {
			logger.Errorf("generate predicate, error columnType %s of column %s", columnType, colName)
			return rds.Predicate{}, nil
		}
		//constant := value.(bool)
		//value = !constant
		//predicateStr = fmt.Sprintf("%s.%s%s%v", tableIndexes[0], realColumnNames[0], symbol, value)
	}

	if err != nil || symbol == "" {
		return rds.Predicate{}, err
	}
	predicate := rds.Predicate{
		PredicateStr:  predicateStr,
		LeftColumn:    rds.Column{ColumnIndex: columnIndexes[0], ColumnId: realColumnNames[0], ColumnType: columnType, TableId: tableIds[0]},
		RightColumn:   rds.Column{},
		ConstantValue: value,
		SymbolType:    symbol,
		PredicateType: 0,
	}
	return predicate, nil
}

func (util MultiDeriveColumnPredicatesUtil) GeneratePredicates(index2Table map[string]string, colName string, columnType string, constantValue interface{}, lhs []interface{}, yFlag bool, transferColNameFlag bool) (rds.Predicate, error) {
	realColumnNames, symbol, _, tableIds, _, columnIndexes, err := util.ParseColumnName(index2Table, colName, columnType)
	if err != nil {
		return rds.Predicate{}, err
	}
	predicateType := 1
	if tableIds[0] != tableIds[1] {
		if yFlag {
			predicateType = 3
		} else {
			predicateType = 2
		}
	}
	predicate := rds.Predicate{
		PredicateStr:  colName,
		LeftColumn:    rds.Column{ColumnIndex: columnIndexes[0], ColumnId: realColumnNames[0], ColumnType: columnType, TableId: tableIds[0]},
		RightColumn:   rds.Column{ColumnIndex: columnIndexes[1], ColumnId: realColumnNames[1], ColumnType: columnType, TableId: tableIds[1]},
		SymbolType:    symbol,
		PredicateType: predicateType,
	}
	return predicate, nil
}

func (util RealColumnPredicatesUtil) ParseColumnName(index2Table map[string]string, colName string, columnType string) (columns []string, symbol string, constantValue interface{}, tableIds []string, tableIndexes []string, columnIndexes []int, err error) {
	return nil, "", nil, nil, nil, nil, err
}

func (util SingleDeriveColumnPredicatesUtil) ParseColumnName(index2Table map[string]string, colName string, columnType string) (columns []string, symbol string, constantValue interface{}, tableIds []string, tableIndexes []string, columnIndexes []int, err error) {
	columns = make([]string, 1)
	tableIds = make([]string, 1)
	tableIndexes = make([]string, 1)
	columnIndexes = make([]int, 1)

	pattern := `=|>=|<=|<|>`
	// 转义模式中的特殊字符
	regex := regexp.MustCompile(pattern)
	matches := regex.FindAllString(colName, -1)
	if len(matches) < 1 {
		logger.Errorf("解析%s符号失败", colName)
		return columns, "", nil, tableIds, tableIndexes, columnIndexes, err
	}
	symbol = matches[0]

	index := strings.Index(colName, ".")
	if index == -1 {
		logger.Errorf("解析%s失败", colName)
		return columns, "", nil, tableIds, tableIndexes, columnIndexes, nil
	}

	tableIndex := strings.TrimSpace(colName[:index])
	columnIndexStr := strings.TrimSpace(tableIndex[1:])
	columnIndex, err := strconv.Atoi(columnIndexStr)
	if err != nil {
		logger.Errorf("解析%s中的ColumnIndex失败", columnIndexStr, err)
		return columns, "", nil, tableIds, tableIndexes, columnIndexes, err
	}
	tableId := index2Table[tableIndex]
	columnAndValueStr := strings.Split(colName[index+1:], symbol)
	column, realValueStr := strings.TrimSpace(columnAndValueStr[0]), strings.TrimSpace(columnAndValueStr[1])
	logger.Debugf("parse colName: %s, tableIndex: %s, columnIndex: %v, tableId: %s, realColumnName: %s, value: %s", colName, tableIndex, columnIndex, tableId, column, realValueStr)

	if columnType == rds_config.EnumType || columnType == rds_config.TextType {
		constantValue = realValueStr
	} else if columnType == rds_config.BoolType {
		constantValue, err = strconv.ParseBool(realValueStr)
		if err != nil {
			logger.Errorf("解析%s成为%s类型失败", realValueStr, columnType, err)
			return columns, "", nil, tableIds, tableIndexes, columnIndexes, err
		}
	} else if columnType == rds_config.TimeType { //todo
		// 解析时间字符串
		constantValue, err = time.Parse(time.RFC3339, realValueStr)
		if err != nil {
			logger.Errorf("解析%s成为%s类型失败", realValueStr, columnType, err)
			return columns, "", nil, tableIds, tableIndexes, columnIndexes, err
		}
	} else {
		logger.Errorf("类型错误：%s, 该类型不应产生这样的columnName：%s", columnType, colName)
		return columns, symbol, constantValue, tableIds, tableIndexes, columnIndexes, nil
	}
	columns[0] = column
	tableIds[0] = tableId
	tableIndexes[0] = tableIndex
	columnIndexes[0] = columnIndex
	return columns, symbol, constantValue, tableIds, tableIndexes, columnIndexes, nil
}

func (util MultiDeriveColumnPredicatesUtil) ParseColumnName(index2Table map[string]string, colName string, columnType string) (columns []string, symbol string, constantValue interface{}, tableIds []string, tableIndexes []string, columnIndexes []int, err error) {
	columns = make([]string, 2)
	tableIds = make([]string, 2)
	tableIndexes = make([]string, 2)
	columnIndexes = make([]int, 2)

	pattern := `=|>=|<=|<|>`
	// 转义模式中的特殊字符
	regex := regexp.MustCompile(pattern)
	matches := regex.FindAllString(colName, -1)
	symbol = matches[0]

	arr := strings.Split(colName, symbol)
	if len(arr) == 2 {
		left := arr[0]
		right := arr[1]
		leftArr := strings.Split(left, ".")
		rightArr := strings.Split(right, ".")
		if len(leftArr) != 2 || len(rightArr) != 2 {
			logger.Errorf("解析%s失败", colName)
			return columns, "", nil, tableIds, tableIndexes, columnIndexes, err
		}
		tableIndex1 := strings.TrimSpace(leftArr[0])
		columns[0] = strings.TrimSpace(leftArr[1])
		tableIndex2 := strings.TrimSpace(rightArr[0])
		columns[1] = strings.TrimSpace(rightArr[1])

		columnIndex1Str := strings.TrimSpace(tableIndex1[1:])
		columnIndex1, err := strconv.Atoi(columnIndex1Str)
		if err != nil {
			logger.Errorf("解析%s中的ColumnIndex失败", columnIndex1Str, err)
			return columns, "", nil, tableIds, tableIndexes, columnIndexes, err
		}
		columnIndex2Str := strings.TrimSpace(tableIndex2[1:])
		columnIndex2, err := strconv.Atoi(columnIndex2Str)
		if err != nil {
			logger.Errorf("解析%s中的ColumnIndex失败", columnIndex2Str, err)
			return columns, "", nil, tableIds, tableIndexes, columnIndexes, err
		}

		tableIds[0] = index2Table[tableIndex1]
		tableIds[1] = index2Table[tableIndex2]

		tableIndexes[0] = tableIndex1
		tableIndexes[1] = tableIndex2

		columnIndexes[0] = columnIndex1
		columnIndexes[1] = columnIndex2

		logger.Debugf("parse colName: %s, tableIndex1: %s, table1: %s, column1: %s, columnIndex1: %v, tableIndex2: %s, table2: %s, column2: %s, columnIndex2: %v",
			colName, tableIndex1, tableIds[0], columns[0], columnIndexes[0], tableIndex2, tableIds[1], columns[1], columnIndexes[1])

	} else {
		logger.Errorf("解析%s失败", colName)
		return columns, "", nil, tableIds, tableIndexes, columnIndexes, err
	}
	return columns, symbol, nil, tableIds, tableIndexes, columnIndexes, nil
}
