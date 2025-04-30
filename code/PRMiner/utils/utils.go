package utils

import (
	"encoding/json"
	"errors"
	"fmt"
	"rds-shenglin/rds_config"
	"rds-shenglin/rock-share/base/logger"
	"rds-shenglin/rock-share/global/model/rds"
	"sort"
	"strconv"
	"strings"
	"time"
)

var lastTotalFreed uint64

func SortPredicates(data []rds.Predicate, isAscending bool) {
	if isAscending {
		sort.SliceStable(data, func(i, j int) bool {
			return data[i].Support < data[j].Support // 升序
		})
		sort.SliceStable(data, func(i, j int) bool {
			return data[i].PredicateStr < data[j].PredicateStr // 升序
		})
	} else {
		sort.SliceStable(data, func(i, j int) bool {
			return data[i].Support > data[j].Support // 降序
		})
		sort.SliceStable(data, func(i, j int) bool {
			return data[i].PredicateStr > data[j].PredicateStr // 降序
		})
	}
}

// FindDiffer 计算两个行号的集合的差集.
// 假设t0和t1是有序的。
func FindDiffer(t0, t1 []int32) []int32 {
	var ans []int32
	i, j := 0, 0
	for i < len(t0) && j < len(t1) {
		if t0[i] == t1[j] {
			i++
			j++
		} else {
			ans = append(ans, t1[j])
			j++
		}
	}
	for j < len(t1) {
		ans = append(ans, t1[j])
		j++
	}
	return ans
}

// 大于1 等于0 小于-1
func CompareTo(val1 interface{}, val2 interface{}, valType string) (int, error) {
	if rds_config.IntType == valType {
		if val1Trans, ok := val1.(int64); ok {
			if val2Trans, ok := val2.(int64); ok {
				if val1Trans > val2Trans {
					return 1, nil
				} else if val1Trans == val2Trans {
					return 0, nil
				} else {
					return -1, nil
				}
			} else {
				msg := fmt.Sprintf("值:%v 转换INT64数据类型失败", val2)
				logger.Errorf(msg)
				return 0, errors.New(msg)
			}
		} else {
			msg := fmt.Sprintf("值:%v 转换INT64数据类型失败", val1)
			logger.Errorf(msg)
			return 0, errors.New(msg)
		}
	} else if rds_config.FloatType == valType {
		if val1Trans, ok := val1.(float64); ok {
			if val2Trans, ok := val2.(float64); ok {
				if val1Trans > val2Trans {
					return 1, nil
				} else if val1Trans == val2Trans {
					return 0, nil
				} else {
					return -1, nil
				}
			} else {
				msg := fmt.Sprintf("值:%v 转换FLOAT64数据类型失败", val2)
				logger.Errorf(msg)
				return 0, errors.New(msg)
			}
		} else {
			msg := fmt.Sprintf("值:%v 转换FLOAT64数据类型失败", val1)
			logger.Errorf(msg)
			return 0, errors.New(msg)
		}
	} else if rds_config.TimeType == valType {
		msg := fmt.Sprintf("值:%v 转换TIME数据类型失败", val1)
		logger.Errorf(msg)
		return 0, errors.New(msg)
	} else {
		return 0, errors.New("不支持的数据类型")
	}
}

type Number interface {
	int | int64 | int32 | uint32 | uint64 | float64
}

func Max[N Number](a, b N) N {
	if a > b {
		return a
	} else {
		return b
	}
}

func Min[N Number](a, b N) N {
	if a < b {
		return a
	} else {
		return b
	}
}

func IsInteger(val interface{}) (int64, bool) {
	if val == nil {
		return -1, false
	}
	switch i := val.(type) {
	case int:
		return int64(i), true
	case int8:
		return int64(i), true
	case int16:
		return int64(i), true
	case int32:
		return int64(i), true
	case int64:
		return i, true
	case uint:
		return int64(i), true
	case uint8:
		return int64(i), true
	case uint16:
		return int64(i), true
	case uint32:
		return int64(i), true
	case uint64:
		return int64(i), true
	}
	return -1, false
}

type KV[K any, V any] struct {
	Key   K
	Value V
}

func MapKVs[M ~map[K]V, K comparable, V any](m M) []KV[K, V] {
	if m == nil {
		return nil
	}
	var kvs = make([]KV[K, V], 0, len(m))
	for k, v := range m {
		kvs = append(kvs, KV[K, V]{
			Key:   k,
			Value: v,
		})
	}
	return kvs
}

func MapIKVs[M ~map[any]V, V any](m M) []KV[any, V] {
	if m == nil {
		return nil
	}
	var kvs = make([]KV[any, V], 0, len(m))
	for k, v := range m {
		kvs = append(kvs, KV[any, V]{
			Key:   k,
			Value: v,
		})
	}
	return kvs
}

func Distinct[T comparable](s []T) []T {
	var r = make([]T, 0, len(s))
	set := map[T]struct{}{}
	for i := range s {
		if _, ok := set[s[i]]; !ok {
			r = append(r, s[i])
			set[s[i]] = struct{}{}
		}
	}
	return r
}

func GetInterfaceToString(value interface{}) string {
	// interface 转 string
	var key string
	if value == nil {
		return key
	}

	switch value.(type) {
	case float64:
		ft := value.(float64)
		key = strconv.FormatFloat(ft, 'f', -1, 64)
	case float32:
		ft := value.(float32)
		key = strconv.FormatFloat(float64(ft), 'f', -1, 64)
	case int:
		it := value.(int)
		key = strconv.Itoa(it)
	case uint:
		it := value.(uint)
		key = strconv.Itoa(int(it))
	case int8:
		it := value.(int8)
		key = strconv.Itoa(int(it))
	case uint8:
		it := value.(uint8)
		key = strconv.Itoa(int(it))
	case int16:
		it := value.(int16)
		key = strconv.Itoa(int(it))
	case uint16:
		it := value.(uint16)
		key = strconv.Itoa(int(it))
	case int32:
		it := value.(int32)
		key = strconv.Itoa(int(it))
	case uint32:
		it := value.(uint32)
		key = strconv.Itoa(int(it))
	case int64:
		it := value.(int64)
		key = strconv.FormatInt(it, 10)
	case uint64:
		it := value.(uint64)
		key = strconv.FormatUint(it, 10)
	case string:
		key = value.(string)
	case time.Time:
		t, _ := value.(time.Time)
		key = t.String()
		// 2022-11-23 11:29:07 +0800 CST  这类格式把尾巴去掉
		key = strings.Replace(key, " +0800 CST", "", 1)
		key = strings.Replace(key, " +0000 UTC", "", 1)
	case []byte:
		key = string(value.([]byte))
	default:
		newValue, _ := json.Marshal(value)
		key = string(newValue)
	}

	return key
}
