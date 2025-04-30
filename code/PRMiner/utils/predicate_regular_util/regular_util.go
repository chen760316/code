package regular_util

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/bovinae/common/collection"
	"github.com/bovinae/common/util"
	"rds-shenglin/rock-share/global/enum"
	"rds-shenglin/rock-share/global/model/rds"
)

const (
	ENUM_NUM_THRESHOLD = 16
)

type IRegularUtil interface {
	GetRegularPredict(ctx context.Context, columnId, columnType string, values []any, supp float64) ([]*rds.Predicate, map[string][]int32, map[string][]int32)
}

type regularUtil struct {
	reFieldCategory    map[FieldCategory]*regexp.Regexp
	reFieldType        map[FieldType]*regexp.Regexp
	fieldCategory2Type map[FieldCategory][]FieldType
}

func NewRegularUtil() IRegularUtil {
	reFieldCategory := make(map[FieldCategory]*regexp.Regexp, FC_TOTAL)
	for k, v := range fcPattern {
		reFieldCategory[k] = regexp.MustCompile(v)
	}
	reFieldType := make(map[FieldType]*regexp.Regexp, FT_TOTAL)
	for k, v := range ftPattern {
		reFieldType[k] = regexp.MustCompile(v)
	}
	return &regularUtil{
		reFieldCategory: reFieldCategory,
		reFieldType:     reFieldType,
		fieldCategory2Type: map[FieldCategory][]FieldType{
			FC_CHINESE: {
				FT_EMAIL,
				FT_SHOP_NAME,
				FT_CAR_NUMBER,
				FT_PROVINCE,
				FT_ADDRESS,
				FT_CHINESE_NAME,
			},
			FC_OTHERS: {
				FT_DATE_TIME,
				FT_RGB,
				FT_FIXED_TEL,
				FT_EMAIL,
				FT_WEB_ADDR,
				FT_IP,
				FT_MAC,
				FT_ENGLISH_NAME,
				FT_DOMAIN_NAME,
				FT_PHONE,
				FT_UNIFIED_SOCIAL_CREDIT_NUMBER,
			},
			FC_ENGLISH: {
				FT_CAR_NUMBER,
				FT_ID_NUMBER,
				FT_ENGLISH_NAME,
				FT_DOMAIN_NAME,
				FT_MAC,
			},
			FC_NUMBER: {
				FT_DATE_TIME,
				FT_ID_NUMBER,
				FT_PHONE,
				FT_ZIP_CODE,
				FT_UNIFIED_SOCIAL_CREDIT_NUMBER,
				FT_INTEGER,
				FT_FLOAT,
			},
		},
	}
}

func (r *regularUtil) GetRegularPredict(ctx context.Context, columnId, columnType string, values []any, supp float64) ([]*rds.Predicate, map[string][]int32, map[string][]int32) {
	if len(values) == 0 {
		return nil, nil, nil
	}

	ftMap := make(map[string]int)
	var nonEmptyRow int
	var hasChinese bool
	var predicates []*rds.Predicate
	rowNums := make(map[string][]int32, 0)
	for i := 0; i < len(values); i++ {
		if util.IsEmpty(values[i]) {
			continue
		}
		nonEmptyRow++

		fc := FC_OTHERS
		for j := FC_CHINESE; j < FC_OTHERS; j++ {
			if r.reFieldCategory[j].MatchString(fmt.Sprintf("%v", values[i])) {
				fc = j
				if fc == FC_CHINESE {
					hasChinese = true
				}
				break
			}
		}
		for _, ft := range r.fieldCategory2Type[fc] {
			if r.reFieldType[ft].MatchString(fmt.Sprintf("%v", values[i])) {
				ftMap[ftPattern[ft]]++
				if ftMap[ftPattern[ft]] == 1 {
					predicates = append(predicates, &rds.Predicate{
						PredicateStr: fmt.Sprintf("regular(t0.%v, '%v')", columnId, ftPattern[ft]),
						LeftColumn: rds.Column{
							ColumnId:   columnId,
							ColumnType: columnType,
						},
						ConstantValue: ftPattern[ft],
						SymbolType:    enum.Regular,
						PredicateType: 0,
					})
				}
				rowNums[ftPattern[ft]] = append(rowNums[ftPattern[ft]], int32(i))
				break
			}
		}
	}
	predicates = r.mergeIntegerFloat(predicates, rowNums)
	outputPredicates := make([]*rds.Predicate, 0, len(predicates))
	removePredicates := make([]*rds.Predicate, 0, len(predicates))
	for i := 0; i < len(predicates); i++ {
		if predicates[i].ConstantValue == nil {
			continue
		}
		currSupp := float64(len(rowNums[predicates[i].ConstantValue.(string)])) / float64(nonEmptyRow)
		if currSupp >= supp {
			predicates[i].Support = currSupp
			outputPredicates = append(outputPredicates, predicates[i])
		} else {
			removePredicates = append(removePredicates, predicates[i])
		}
	}

	if len(outputPredicates) > 0 {
		for _, p := range removePredicates {
			if len(rowNums[p.ConstantValue.(string)]) > 0 {
				delete(rowNums, p.ConstantValue.(string))
			}
		}
		if !r.onlyIntegerOrFloat(outputPredicates) {
			return r.getConflictRowNums(outputPredicates, rowNums, len(values))
		}
	}

	dynaPredicates, dynaRowNums := r.getDynamicRegularPredicate(ctx, columnId, columnType, values, hasChinese, supp)
	if len(dynaPredicates) == 0 {
		return r.getConflictRowNums(outputPredicates, rowNums, len(values))
	}
	for _, outputPredicate := range outputPredicates {
		dynaPredicates = append(dynaPredicates, outputPredicate)
		pred := outputPredicate.ConstantValue.(string)
		dynaRowNums[pred] = rowNums[pred]
	}
	return r.getConflictRowNums(dynaPredicates, dynaRowNums, len(values))
}

func (r *regularUtil) getConflictRowNums(predicates []*rds.Predicate, rowNum map[string][]int32, length int) ([]*rds.Predicate, map[string][]int32, map[string][]int32) {
	conflictRowNums := make(map[string][]int32, 0)
	for _, predicate := range predicates {
		positive := rowNum[predicate.ConstantValue.(string)]
		conflictRowNums[predicate.ConstantValue.(string)] = differenceSet(length, positive)
	}
	return predicates, rowNum, conflictRowNums
}

func differenceSet(length int, a []int32) (b []int32) {
	b = make([]int32, 0, length-len(a))
	var j int
	for i := int32(0); i < int32(length); i++ {
		if j >= len(a) {
			b = append(b, i)
			continue
		}
		if i == a[j] {
			j++
			continue
		}
		b = append(b, i)
	}
	return b
}

func (r *regularUtil) mergeIntegerFloat(predicates []*rds.Predicate, rowNums map[string][]int32) []*rds.Predicate {
	var integerPred, floatPred, zipcodePred *rds.Predicate
	for _, predicate := range predicates {
		if predicate.ConstantValue.(string) == ftPattern[FT_INTEGER] {
			integerPred = predicate
		} else if predicate.ConstantValue.(string) == ftPattern[FT_FLOAT] {
			floatPred = predicate
		} else if predicate.ConstantValue.(string) == ftPattern[FT_ZIP_CODE] {
			zipcodePred = predicate
		}
	}
	if integerPred != nil && zipcodePred != nil {
		if len(rowNums[integerPred.ConstantValue.(string)]) > len(rowNums[zipcodePred.ConstantValue.(string)]) {
			rowNums[integerPred.ConstantValue.(string)] = append(rowNums[integerPred.ConstantValue.(string)], rowNums[zipcodePred.ConstantValue.(string)]...)
			sort.Slice(rowNums[integerPred.ConstantValue.(string)], func(i, j int) bool {
				return rowNums[integerPred.ConstantValue.(string)][i] < rowNums[integerPred.ConstantValue.(string)][j]
			})
			delete(rowNums, zipcodePred.ConstantValue.(string))
			for i := 0; i < len(predicates); i++ {
				if predicates[i].ConstantValue.(string) == ftPattern[FT_ZIP_CODE] {
					predicates = append(predicates[:i], predicates[i+1:]...)
					break
				}
			}
		}
	}
	if integerPred != nil && floatPred != nil {
		rowNums[floatPred.ConstantValue.(string)] = append(rowNums[floatPred.ConstantValue.(string)], rowNums[integerPred.ConstantValue.(string)]...)
		sort.Slice(rowNums[floatPred.ConstantValue.(string)], func(i, j int) bool {
			return rowNums[floatPred.ConstantValue.(string)][i] < rowNums[floatPred.ConstantValue.(string)][j]
		})
		delete(rowNums, integerPred.ConstantValue.(string))
		for i := 0; i < len(predicates); i++ {
			if predicates[i].ConstantValue.(string) == ftPattern[FT_INTEGER] {
				predicates = append(predicates[:i], predicates[i+1:]...)
				break
			}
		}
	}
	return predicates
}

func (r *regularUtil) onlyIntegerOrFloat(outputPredicates []*rds.Predicate) bool {
	for _, outputPredicate := range outputPredicates {
		if outputPredicate.ConstantValue.(string) != ftPattern[FT_INTEGER] && outputPredicate.ConstantValue.(string) != ftPattern[FT_FLOAT] {
			return false
		}
	}
	return true
}

func (r *regularUtil) getDynamicRegularPredicate(ctx context.Context, columnId, columnType string, values []any, hasChinese bool, supp float64) (predicates []*rds.Predicate, rowNums map[string][]int32) {
	if len(values) == 0 {
		return nil, nil
	}

	lengthMap := make(map[int]int, 0)
	byteValues := make([][]rune, 0)
	reverseByteValues := make([][]rune, 0)
	enumMap := make(map[string]int, 0)
	var nonEmptyRow int
	for i := 0; i < len(values); i++ {
		if util.IsEmpty(values[i]) {
			continue
		}
		nonEmptyRow++

		value := fmt.Sprint(values[i])
		valueRune := []rune(value)
		lengthMap[len(valueRune)]++

		if len(enumMap) <= ENUM_NUM_THRESHOLD {
			enumMap[value]++
		}
		byteValues = append(byteValues, valueRune)
		valueRune1 := make([]rune, len(valueRune))
		copy(valueRune1, valueRune)
		reverseByteValues = append(reverseByteValues, util.ReverseSlice(valueRune1))
	}
	if nonEmptyRow == 0 {
		return nil, nil
	}

	predicateMap := make(map[FieldType]*rds.Predicate, 0)
	length := r.getLengthPredicate(columnId, columnType, lengthMap, nonEmptyRow, predicateMap, supp)
	r.getEnumPredicate(columnId, columnType, enumMap, nonEmptyRow, predicateMap)
	prefixRunes, suffixRunes := r.getPrefixSuffixPredicate(columnId, columnType, nonEmptyRow, byteValues, reverseByteValues, predicateMap, supp)
	if len(predicateMap) == 0 {
		return nil, nil
	}

	rowNums = r.getRowNumsAndSupport(values, length, prefixRunes, suffixRunes, predicateMap)
	for _, predicate := range predicateMap {
		predicates = append(predicates, predicate)
	}
	return predicates, rowNums
}

func (r *regularUtil) getLengthPredicate(columnId, columnType string, lengthMap map[int]int, nonEmptyRow int, predicateMap map[FieldType]*rds.Predicate, supp float64) int {
	var length, cnt int
	for k, v := range lengthMap {
		if v > cnt {
			cnt = v
			length = k
		}
	}
	threshold := int(float64(nonEmptyRow) * supp)
	if length > 0 && cnt >= threshold {
		predicateMap[FT_LENGTH] = r.getPredicate(length, FT_LENGTH, columnId, columnType, float64(cnt)/float64(nonEmptyRow))
		return length
	}
	return 0
}

func (r *regularUtil) getEnumPredicate(columnId, columnType string, enumMap map[string]int, nonEmptyRow int, predicateMap map[FieldType]*rds.Predicate) {
	if len(enumMap) > ENUM_NUM_THRESHOLD {
		return
	}
	enumSlice := make([]string, 0, len(enumMap))
	var cnt int
	for k, v := range enumMap {
		enumSlice = append(enumSlice, k)
		cnt += v
	}
	sort.Slice(enumSlice, func(i, j int) bool {
		return enumSlice[i] < enumSlice[j]
	})
	predicateMap[FT_ENUM] = r.getPredicate(strings.Join(enumSlice, "|"), FT_ENUM, columnId, columnType, float64(cnt)/float64(nonEmptyRow))
}

func (r *regularUtil) getPrefixSuffixPredicate(columnId, columnType string, nonEmptyRow int, byteValues, reverseByteValues [][]rune, predicateMap map[FieldType]*rds.Predicate, supp float64) ([]rune, []rune) {
	sort.Slice(byteValues, func(i, j int) bool {
		return util.RunesCompare(byteValues[i], byteValues[j]) <= 0
	})
	sort.Slice(reverseByteValues, func(i, j int) bool {
		return util.RunesCompare(reverseByteValues[i], reverseByteValues[j]) <= 0
	})
	threshold := int(float64(nonEmptyRow) * supp)
	prefixRunes := r.findPrefix(byteValues, threshold)
	suffixRunes := r.findPrefix(reverseByteValues, threshold)
	if len(prefixRunes) > 0 {
		predicateMap[FT_PREFIX] = r.getPredicate(string(prefixRunes), FT_PREFIX, columnId, columnType, 0)
	}
	if len(suffixRunes) > 0 {
		suffixRunes = util.ReverseSlice(suffixRunes)
		predicateMap[FT_SUFFIX] = r.getPredicate(string(suffixRunes), FT_SUFFIX, columnId, columnType, 0)
	}
	return prefixRunes, suffixRunes
}

func (r *regularUtil) getRowNumsAndSupport(values []any, length int, prefixRunes, suffixRunes []rune, predicateMap map[FieldType]*rds.Predicate) map[string][]int32 {
	rowNums := make(map[string][]int32, 0)
	var prefixRows, suffixRows int
	var nonEmptyRow int
	for i := 0; i < len(values); i++ {
		if util.IsEmpty(values[i]) {
			continue
		}
		nonEmptyRow++
		value := fmt.Sprint(values[i])
		valueRune := []rune(value)
		for ft, predicate := range predicateMap {
			var shouldAppend bool
			switch ft {
			case FT_LENGTH:
				if len(valueRune) == length {
					shouldAppend = true
				}
			case FT_ENUM:
				shouldAppend = true
			case FT_PREFIX:
				if len(util.GetPrefix(valueRune, prefixRunes)) == len(prefixRunes) {
					shouldAppend = true
					prefixRows++
				}
			case FT_SUFFIX:
				if len(util.GetSuffix(valueRune, suffixRunes)) == len(suffixRunes) {
					shouldAppend = true
					suffixRows++
				}
			}
			if shouldAppend {
				rowNums[predicate.ConstantValue.(string)] = append(rowNums[predicate.ConstantValue.(string)], int32(i))
			}
		}
	}
	if predicateMap[FT_PREFIX] != nil {
		predicateMap[FT_PREFIX].Support = float64(prefixRows) / float64(nonEmptyRow)
	}
	if predicateMap[FT_SUFFIX] != nil {
		predicateMap[FT_SUFFIX].Support = float64(suffixRows) / float64(nonEmptyRow)
	}
	return rowNums
}

func (r *regularUtil) getPredicate(argv any, ft FieldType, columnId, columnType string, supp float64) *rds.Predicate {
	regular := fmt.Sprintf(ftPattern[ft], argv)
	return &rds.Predicate{
		PredicateStr: fmt.Sprintf("regular(t0.%v, '%v')", columnId, regular),
		LeftColumn: rds.Column{
			ColumnId:   columnId,
			ColumnType: columnType,
		},
		ConstantValue: regular,
		SymbolType:    enum.Regular,
		PredicateType: 0,
		Support:       supp,
	}
}

func (r *regularUtil) findPrefix(values [][]rune, threshold int) []rune {
	if len(values) == 0 {
		return nil
	}

	var prefix, maxPrefix []rune
	var preRoot, currRoot *collection.SortedTrieNode[rune]
	for i := 1; i < len(values); i++ {
		prefix = util.GetPrefix(values[i-1], values[i])
		currRoot = preRoot.Add(prefix)
		if preRoot != currRoot && preRoot != nil {
			currMaxPrefix := preRoot.GetMaxPrefix(nil, nil, threshold)
			if len(currMaxPrefix) > len(maxPrefix) {
				maxPrefix = currMaxPrefix
			}
		}
		preRoot = currRoot
	}
	currMaxPrefix := currRoot.GetMaxPrefix(nil, nil, threshold)
	if len(currMaxPrefix) > len(maxPrefix) {
		maxPrefix = currMaxPrefix
	}

	return maxPrefix
}
