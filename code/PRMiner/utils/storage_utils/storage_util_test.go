package storage_utils

import (
	"fmt"
	"rds-shenglin/rock-share/global/utils/similarity"
	"rds-shenglin/rock-share/global/utils/similarity/similarity_cluster"
	"rds-shenglin/rock-share/global/utils/similarity/similarity_enum"
	"rds-shenglin/rds_config"
	"testing"
)

func TestGetTableAllValueIndexes(t *testing.T) {
	tableIndexes, index2Value, _ := GetTableAllValueIndexes(map[string][]interface{}{
		"aaa": {int64(1), int64(4), int64(3), int64(2)},
	}, map[string]string{
		"aaa": rds_config.IntType,
	})
	for col, indexes := range tableIndexes {
		fmt.Println(col, indexes)
	}
	for col, indexes := range index2Value {
		fmt.Println(col, indexes)
	}
}

func TestGetTableAllValueIndexes2(t *testing.T) {
	var values []interface{}
	values = append(values, nil)
	for _, v := range []int{5, 5, 6, 4, 4, 7, 3, 2} {
		values = append(values, int64(v))
	}
	values = append(values, nil)

	tableIndexes, index2Value, _ := GetTableAllValueIndexes(map[string][]interface{}{
		"aaa": values,
	}, map[string]string{
		"aaa": rds_config.IntType,
	})
	for col, indexes := range tableIndexes {
		fmt.Println(col, indexes)
	}
	for col, indexes := range index2Value {
		fmt.Println(col, indexes)
	}
}

func TestGetTableAllValueIndexes3(t *testing.T) {
	tableIndexes, index2Value, _ := GetTableAllValueIndexes(map[string][]interface{}{
		"aaa": {int64(1), int64(4), int64(3), int64(2)},
		"bbb": {"a", "b", "a", "b", "c", nil},
	}, map[string]string{
		"aaa": rds_config.IntType,
		"bbb": rds_config.StringType,
	})
	for col, indexes := range tableIndexes {
		fmt.Println(col, indexes)
	}
	for col, indexes := range index2Value {
		fmt.Println(col, indexes)
	}
}

func TestPairSimilar(t *testing.T) {
	s := similarity.AlgorithmMap[similarity_enum.JaroWinkler]

	f := s.Compare("apple", "apples")

	fmt.Println(f)
}

func TestClusterSimilar(t *testing.T) {
	var cluster [][]int32 = similarity_cluster.CreateCluster[int32]([]string{
		"aaaa",
		"bbb",
		"aaaa",
		"aaaa",
		"bbb",
		"aaaa",
	}, similarity.AlgorithmMap[similarity_enum.JaroWinkler], 0.9)
	for _, group := range cluster {
		fmt.Println(group)
	}
}
