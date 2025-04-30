package utils

import (
	"bytes"
	"encoding/json"
	"fmt"
	"rds-shenglin/rock-share/global/enum"
	"rds-shenglin/rock-share/global/model/rds"
	"sort"
	"strings"
)

func SplitRule(rule string) (map[string]bool, string) {
	lhs := make(map[string]bool)
	tmp := strings.Split(rule, "->")
	rhs := strings.TrimSpace(tmp[1])
	for _, predicate := range strings.Split(tmp[0], "^") {
		lhs[strings.TrimSpace(predicate)] = true
	}
	return lhs, rhs
}

func GetRuleJson(rule rds.Rule) string {
	byteBuf := bytes.NewBuffer([]byte{})
	encoder := json.NewEncoder(byteBuf)
	encoder.SetEscapeHTML(false)
	err := encoder.Encode(rule)
	if err != nil {
		fmt.Println("error: ", err)
		return ""
	}
	return byteBuf.String()
	//jsonBytes, err := json.Marshal(rule)
	//if err != nil {
	//	fmt.Println("error: ", err)
	//	return ""
	//}
	//return string(jsonBytes)
}

func GetFormatRule(ruleJsonStr string) rds.Rule {
	rule := &rds.Rule{}
	err := json.Unmarshal([]byte(ruleJsonStr), &rule)
	if err != nil {
		fmt.Println("error: ", err)
		return rds.Rule{}
	}
	return *rule
}

func IsAbandoned(ree string, abandonedRules []string) bool {
	for _, rule := range abandonedRules {
		if ree == rule {
			return true
		}
	}
	return false
}

func GenerateTableIdStr(tableId string, isSingle bool) string {
	str := tableId + "(t0)^"
	if !isSingle {
		str += tableId
		str += "(t1)^"
	}
	return str
}

func GenerateMultiTableIdStr(tableId2index map[string]int) string {
	tableIdArr := make([]string, len(tableId2index))
	for s, i := range tableId2index {
		tableIdArr[i] = s
	}
	arr := make([]string, len(tableIdArr)*2)
	for i, tableId := range tableIdArr {
		arr[i*2] = fmt.Sprintf("%s(t%d)", tableId, i*2)
		arr[i*2+1] = fmt.Sprintf("%s(t%d)", tableId, i*2+1)
	}
	return strings.Join(arr, "^")
}

func GenerateMultiTableIdStrNew(tableId2index map[string]int) string {
	var tableIndexArr []int
	tableIndex2id := make(map[int]string)
	for tableId, i := range tableId2index {
		tableIndexArr = append(tableIndexArr, i)
		tableIndex2id[i] = tableId
	}
	sort.Ints(tableIndexArr)
	arr := make([]string, len(tableIndexArr)*2)
	for j, i := range tableIndexArr {
		tableId := tableIndex2id[i]
		arr[j*2] = fmt.Sprintf("%s(t%d)", tableId, i*2)
		arr[j*2+1] = fmt.Sprintf("%s(t%d)", tableId, i*2+1)
	}
	return strings.Join(arr, "^")
}

func GenerateMultiTableIdStrNew2(tableId2index map[string]int, lhs []rds.Predicate) string {
	var usedTableIndex []int
	for _, p := range lhs {
		usedTableIndex = append(usedTableIndex, p.LeftColumn.ColumnIndex)
		usedTableIndex = append(usedTableIndex, p.RightColumn.ColumnIndex)
	}
	usedTableIndex = Distinct(usedTableIndex)
	sort.Ints(usedTableIndex)

	var tableIndexArr []int
	tableIndex2id := make(map[int]string)
	for tableId, i := range tableId2index {
		tableIndexArr = append(tableIndexArr, i)
		tableIndex2id[i] = tableId
	}

	arr := make([]string, 0, len(tableIndexArr)*2)
	for _, tid := range usedTableIndex {
		tableId := tableIndex2id[tid/2]
		arr = append(arr, fmt.Sprintf("%s(t%d)", tableId, tid))
	}
	return strings.Join(arr, "^")
}

func GenerateTableIdStrByIndex2Table(index2Table map[string]string) string {
	indexes := make([]string, 0)
	for index := range index2Table {
		indexes = append(indexes, index)
	}
	sort.Strings(indexes)

	list := make([]string, len(indexes))
	for i, index := range indexes {
		list[i] = fmt.Sprintf("%s(%s)", index2Table[index], index)
	}
	return strings.Join(list, "^")
}

func GetRelatedTable(r rds.Rule) []string {
	m := make(map[string]bool)
	for _, pre := range r.LhsPredicates {
		m[pre.LeftColumn.TableId] = true
		rightTableId := pre.RightColumn.TableId
		if rightTableId != "" {
			m[rightTableId] = true
		}
	}
	m[r.Rhs.LeftColumn.TableId] = true

	rhsRightTid := r.Rhs.RightColumn.TableId
	if rhsRightTid != "" {
		m[rhsRightTid] = true
	}
	result := make([]string, len(m))
	i := 0
	for id := range m {
		result[i] = id
		i++
	}
	return result
}

func SaveRule(rule rds.Rule, ruleMap map[string]map[enum.RuleType][]rds.Rule) {
	leftTableId := rule.Rhs.LeftColumn.TableId
	//rightTableId := rule.Rhs.RightColumn.TableId
	ruleTypes := GetRuleType(rule)
	if _, ok := ruleMap[leftTableId]; !ok {
		ruleMap[leftTableId] = make(map[enum.RuleType][]rds.Rule)
	}
	for _, ruleType := range ruleTypes {
		ruleMap[leftTableId][ruleType] = append(ruleMap[leftTableId][ruleType], rule)
	}
	//if rightTableId != leftTableId {
	//	if _, ok := ruleMap[rightTableId]; !ok {
	//		ruleMap[rightTableId] = make(map[enum.RuleType][]rds.Rule)
	//	}
	//	for _, ruleType := range ruleTypes {
	//		ruleMap[rightTableId][ruleType] = append(ruleMap[rightTableId][ruleType], rule)
	//	}
	//}
}

func GetRuleType(rule rds.Rule) []enum.RuleType {
	var result []enum.RuleType
	if rule.RuleType == 0 {
		result = append(result, enum.SingleTableSingleRow)
	} else {
		relatedTable := make(map[string]bool)
		lhsRelatedIndex := make(map[int]bool)
		isML := false
		isSimilar := false
		hasConst := false
		isNormal := true
		for _, predicate := range rule.LhsPredicates {
			if predicate.SymbolType == enum.Similar {
				isNormal = false
				isSimilar = true
			}
			if predicate.SymbolType == enum.ML {
				isNormal = false
				isML = true
			}
			if predicate.PredicateType == 0 {
				isNormal = false
				hasConst = true
			}
			lhsRelatedIndex[predicate.LeftColumn.ColumnIndex] = true
			lhsRelatedIndex[predicate.RightColumn.ColumnIndex] = true
			relatedTable[predicate.LeftColumn.TableId] = true
			if predicate.RightColumn.TableId != "" {
				relatedTable[predicate.RightColumn.TableId] = true
			}
		}
		if len(relatedTable) == 1 {
			if hasConst {
				result = append(result, enum.SingleTableMultiRowWithConst)
			}
			if isSimilar {
				result = append(result, enum.SingleTableMultiRowWithSimilar)
			}
			if isML {
				result = append(result, enum.SingleTableMultiRowWithML)
			}
			if isNormal {
				result = append(result, enum.SingleTableMultiRow)
			}
		} else {
			if rule.Rhs.LeftColumn.ColumnId != rule.Rhs.RightColumn.ColumnId {
				result = append(result, enum.MultiTableMultiRowMultiY)
			}
			if hasConst {
				result = append(result, enum.MultiTableMultiRowWithConst)
			}
			if isSimilar {
				result = append(result, enum.MultiTableMultiRowWithSimilar)
			}
			if isML {
				result = append(result, enum.MultiTableMultiRowWithML)
			}
			if isNormal {
				result = append(result, enum.MultiTableMultiRow)
			}
		}
	}
	return result
}

func SortRules(rules []rds.Rule, specificColumns map[string]bool) []rds.Rule {
	var specialRules []rds.Rule
	var normalRules []rds.Rule
	for _, rule := range rules {
		isNorm := true
		for _, predicate := range rule.LhsPredicates {
			if specificColumns[predicate.LeftColumn.ColumnId] || specificColumns[predicate.RightColumn.ColumnId] {
				isNorm = false
				break
			}
		}
		if isNorm {
			normalRules = append(normalRules, rule)
		} else {
			specialRules = append(specialRules, rule)
		}
	}
	return append(specialRules, normalRules...)
}

func ReeToRule(ree string, dataType map[string]map[string]string, tableId2index map[string]int) rds.Rule {
	tableId := strings.Split(ree, " ^ ")[0]
	xStr := strings.Split(strings.Split(ree, " ^ ")[1], "->")[0]
	yStr := strings.Split(strings.Split(ree, " ^ ")[1], "->")[1]
	var lhs []rds.Predicate
	for _, pStr := range strings.Split(xStr, "^") {
		lhs = append(lhs, PreStrToPredicate(pStr, dataType, tableId, tableId2index))
	}
	rhs := PreStrToPredicate(yStr, dataType, tableId, tableId2index)
	px := GetLhsStr(lhs)
	ree = px + "->" + yStr
	ree = GenerateMultiTableIdStrNew2(tableId2index, lhs) + " ^ " + ree

	rule := rds.Rule{
		TableId:       "0",
		Ree:           ree,
		LhsPredicates: lhs,
		LhsColumns:    GetPredicatesColumn(lhs),
		Rhs:           rhs,
		RhsColumn:     rhs.LeftColumn,
		CR:            0,
		FTR:           0,
		RuleType:      1,
		XSupp:         0,
		XySupp:        0,
		XSatisfyCount: 0,
		XSatisfyRows:  nil,
		XIntersection: nil,
	}
	return rule
}

func FillRule(rule *rds.Rule, dataType map[string]map[string]string, tableId2index map[string]int) {
	for i := 0; i < len(rule.LhsPredicates); i++ {
		FillPredicate(&rule.LhsPredicates[i], dataType, tableId2index)
	}
	FillPredicate(&rule.Rhs, dataType, tableId2index)

	px := GetLhsStr(rule.LhsPredicates)
	ree := px + "->" + rule.Rhs.PredicateStr
	ree = GenerateMultiTableIdStrNew2(tableId2index, rule.LhsPredicates) + " ^ " + ree

	rule.Ree = ree
	rule.LhsColumns = GetPredicatesColumn(rule.LhsPredicates)
	rule.RhsColumn = rule.Rhs.LeftColumn
	if rule.Rhs.PredicateType == 0 {
		rule.RuleType = 0
	} else if rule.Rhs.PredicateType == 1 {
		rule.RuleType = 1
	}
}
