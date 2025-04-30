package storage_utils

import (
	"rds-shenglin/rds_config"
	"rds-shenglin/rock-share/base/logger"
	"sort"
)

// GetTableAllValueIndexes 计算一个表格的索引值->值、值->索引值映射
// 索引值是一列中的值排序之后的次序（不考虑nil）, nil的索引值是NilIndex。
func GetTableAllValueIndexes(tableValues map[string][]interface{}, columnTypeMap map[string]string) (map[string][]int32, map[string]map[int32]interface{}, map[string]map[interface{}]int32) {
	tableIndexes := map[string][]int32{}
	index2Value := map[string]map[int32]interface{}{}
	value2Index := map[string]map[interface{}]int32{}
	for colName, values := range tableValues {
		colIndex2Value := map[int32]interface{}{}
		colIndex2Value[rds_config.NilIndex] = nil
		colValue2Index := map[interface{}]int32{}
		colValue2Index[nil] = rds_config.NilIndex
		// 因为认为空字符串之间不相等,所以直接把空字符串当nil来索引
		colValue2Index[""] = rds_config.NilIndex
		//set := map[interface{}]int32{}
		//set[nil] = rds_config.NilIndex
		indexes := make([]int32, 0, len(values))
		if columnTypeMap[colName] == rds_config.FloatType || columnTypeMap[colName] == rds_config.IntType {
			// need sort
			valuesDistinctNotNil := UniqueArrayFilterNil(values)
			if columnTypeMap[colName] == rds_config.FloatType {
				sort.Slice(valuesDistinctNotNil, func(i, j int) bool {
					return floatVal(valuesDistinctNotNil[i]) < floatVal(valuesDistinctNotNil[j])
				})
			}
			if columnTypeMap[colName] == rds_config.IntType {
				sort.Slice(valuesDistinctNotNil, func(i, j int) bool {
					return valuesDistinctNotNil[i].(int64) < valuesDistinctNotNil[j].(int64)
				})
			}
			for orderIndex, value := range valuesDistinctNotNil {
				// 索引值由已排序的所有distinct value的位置决定
				colValue2Index[value] = int32(orderIndex)
				colIndex2Value[int32(orderIndex)] = value
			}
		}
		for _, value := range values {
			if id, ok := colValue2Index[value]; ok {
				indexes = append(indexes, id)
			} else {
				id = int32(len(colValue2Index))
				colValue2Index[value] = id
				colIndex2Value[id] = value
				indexes = append(indexes, id)
			}
		}
		tableIndexes[colName] = indexes
		index2Value[colName] = colIndex2Value
		value2Index[colName] = colValue2Index
	}
	return tableIndexes, index2Value, value2Index
}

func floatVal(i interface{}) float64 {
	if f, ok := i.(float64); ok {
		return f
	}
	logger.Warn("no float", i)
	return 0.0
}

func UniqueArrayFilterNil(arr []interface{}) []interface{} {
	var result []interface{}
	tmp := make(map[interface{}]bool)
	for _, value := range arr {
		tmp[value] = true
	}
	for key := range tmp {
		if key == nil {
			continue
		}
		result = append(result, key)
	}
	return result
}
