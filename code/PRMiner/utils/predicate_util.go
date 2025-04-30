package utils

import (
	"fmt"
	"golang.org/x/exp/slices"
	"rds-shenglin/rds_config"
	"rds-shenglin/rock-share/base/logger"
	"rds-shenglin/rock-share/global/enum"
	"rds-shenglin/rock-share/global/model/rds"
	"sort"
	"strconv"
	"strings"

	mapset "github.com/deckarep/golang-set"
)

// GetPredicatesStr1 将多个谓词排序并转换成str
func GetPredicatesStr1(predicates map[string]bool) string {
	s1 := mapset.NewSet() //lhs
	for kk := range predicates {
		s1.Add(kk)
	}
	lhsSorted := s1.ToSlice()
	sort.Slice(lhsSorted, func(i, j int) bool {
		return lhsSorted[i].(string) < lhsSorted[j].(string)
	})
	lhsStr := ""
	for _, vs11 := range lhsSorted {
		lhsStr += fmt.Sprintf("%v", vs11)
	}
	return lhsStr
}

func GetPredicatesStr(predicates []rds.Predicate) string {
	SortPredicates(predicates, false)
	lhsStr := ""
	for _, vs11 := range predicates {
		lhsStr += fmt.Sprintf("%v", vs11.PredicateStr)
	}
	return lhsStr
}

// GetConstantValue 获取常数谓词中的常数值
func GetConstantValue(predicate string) string {
	return strings.TrimSpace(strings.Split(predicate, "=")[1])
}

func GetSortedLhs(predicates map[string]bool) string {
	tmp := make([]string, 0, len(predicates))
	for k := range predicates {
		tmp = append(tmp, k)
	}
	sort.Slice(tmp, func(i, j int) bool {
		return tmp[i] < tmp[j]
	})
	return strings.Join(tmp, " ^ ")
}

func GetLhsStr(predicates []rds.Predicate) string {
	SortPredicates(predicates, false)
	var tmp []string
	for _, predicate := range predicates {
		tmp = append(tmp, predicate.PredicateStr)
	}
	return strings.Join(tmp, " ^ ")
}

func GetPredicateType(predicate string) string {
	if strings.Contains(predicate, enum.GreaterE) {
		return enum.GreaterE
	}
	if strings.Contains(predicate, enum.Less) {
		return enum.Less
	}
	return enum.Equal
}

func DeletePredicate(predicates []rds.Predicate, predicate rds.Predicate) []rds.Predicate {
	var result []rds.Predicate
	for _, p := range predicates {
		if p.PredicateStr != predicate.PredicateStr {
			result = append(result, p)
		}
	}
	return result
}

func GetPredicatesColumn(predicates []rds.Predicate) []rds.Column {
	columns := make([]rds.Column, len(predicates))
	for i, predicate := range predicates {
		columns[i] = predicate.LeftColumn
	}
	return columns
}

// SplitLhs 将谓词组合拆分成一个规则的形式
func SplitLhs(lhs []rds.Predicate) ([]rds.Predicate, rds.Predicate) {
	var lhsTmp []rds.Predicate
	var rhsTmp rds.Predicate
	flag := true
	lhsLen := len(lhs)
	index := 1
	for _, predicate := range lhs {
		// 存在常数谓词的话,将常数谓词作为rhs
		if flag && predicate.PredicateType == 0 {
			rhsTmp = predicate
			flag = false
			index++
			continue
		}
		// 不存在常数谓词的话,将最后一个谓词作为rhs
		if flag && index == lhsLen {
			rhsTmp = predicate
			continue
		}
		index++
		lhsTmp = append(lhsTmp, predicate)
	}
	return lhsTmp, rhsTmp
}

func FilterLhs(rhs rds.Predicate, lhs []rds.Predicate) []rds.Predicate {
	var result []rds.Predicate
	for _, k := range lhs {
		if k.LeftColumn != rhs.LeftColumn {
			result = append(result, k)
		}
	}
	return result
}

func ContainsColumn(predicate rds.Predicate, lhs []rds.Predicate) bool {
	// 如果待判断的谓词为相似度谓词需要特殊处理
	if predicate.PredicateType == 1 && predicate.SymbolType == enum.Similar {
		for _, predicate2 := range lhs {
			if predicate.LeftColumn.ColumnId == predicate2.LeftColumn.ColumnId && predicate.RightColumn.ColumnId == predicate2.RightColumn.ColumnId {
				return true
			}
			if predicate.RightColumn.ColumnId == predicate2.LeftColumn.ColumnId && predicate.LeftColumn.ColumnId == predicate2.RightColumn.ColumnId {
				return true
			}
		}
	}
	for _, predicate2 := range lhs {
		if predicate.LeftColumn.ColumnId == predicate2.LeftColumn.ColumnId {
			return true
		}
	}
	return false
}

func HasCrossColumnPredicate(predicates []rds.Predicate) bool {
	for _, predicate := range predicates {
		// 常数谓词的直接跳过
		if predicate.PredicateType == 0 {
			continue
		}
		if predicate.LeftColumn.ColumnId != predicate.RightColumn.ColumnId {
			return true
		}
	}
	return false
}

func GetSimilarTypeThreshold(predicate rds.Predicate) (string, float64) {
	predicateStr := predicate.PredicateStr
	tmpStr := strings.Split(predicateStr, "(")[1]
	values := strings.Split(tmpStr, ",")
	similarType := values[0]
	threshold, _ := strconv.ParseFloat(values[1], 64)
	return similarType, threshold
}

func MultiTableFilterLhs(rhs rds.Predicate, lhs []rds.Predicate, table2JoinTables map[string]map[string]struct{}) []rds.Predicate {
	var result []rds.Predicate
	rhsLeftTable := rhs.LeftColumn.TableId
	rhsRightTable := rhs.RightColumn.TableId
	for _, k := range lhs {
		if k.PredicateStr == rhs.PredicateStr {
			continue
		}
		leftTableId := k.LeftColumn.TableId
		rightTableId := k.RightColumn.TableId
		if IsTableRelated(rhsLeftTable, rhsRightTable, leftTableId, rightTableId, table2JoinTables) {
			result = append(result, k)
		}
	}
	return result
}

func GetRelatedTables(tableIds []string, table2JoinTables map[string]map[string]struct{}) []string {
	m := make(map[string]bool)
	for _, tid := range tableIds {
		m[tid] = true
	}

	var ans []string
	lastSize := 0
	for lastSize != len(m) {
		lastSize = len(m)
		for t := range m {
			for k := range table2JoinTables[t] {
				m[k] = true
			}
		}
	}
	for t := range m {
		ans = append(ans, t)
	}
	return ans
}

func SingleTableFilterLhs(rhs rds.Predicate, lhs []rds.Predicate) []rds.Predicate {
	var result []rds.Predicate
	for _, k := range lhs {
		if k.PredicateStr == rhs.PredicateStr {
			continue
		}
		// 避免 similar(a,a) -> a=a
		if rds_config.IndexType == k.LeftColumn.ColumnType {
			if k.LeftColumn.TableId == rhs.LeftColumn.TableId && k.LeftColumn.ColumnId == rhs.LeftColumn.ColumnId {
				continue
			}
		}
		if rds_config.IndexType == k.RightColumn.ColumnType {
			if k.RightColumn.TableId == rhs.RightColumn.TableId && k.RightColumn.ColumnId == rhs.RightColumn.ColumnId {
				continue
			}
		}
		result = append(result, k)
	}
	return result
}

func IsTableRelated(rhsLeftTable, rhsRightTable, leftTableId, rightTableId string, table2JoinTables map[string]map[string]struct{}) bool {
	if rhsLeftTable == leftTableId || rhsRightTable == rightTableId {
		return true
	} else if _, exist := table2JoinTables[rhsLeftTable][leftTableId]; exist {
		return true
	} else if _, exist := table2JoinTables[rhsLeftTable][rightTableId]; exist {
		return true
	} else if _, exist := table2JoinTables[rhsRightTable][leftTableId]; exist {
		return true
	} else if _, exist := table2JoinTables[rhsRightTable][rightTableId]; exist {
		return true
	}
	return false
}

func GetPredicateColumnIndex(predicate rds.Predicate) (int, int) {
	isConstant := predicate.PredicateType == 0
	if isConstant {
		return predicate.LeftColumn.ColumnIndex, -1
	}
	return predicate.LeftColumn.ColumnIndex, predicate.RightColumn.ColumnIndex
	// todo 现在临时方案,只有=号
	columnArr := strings.Split(predicate.PredicateStr, rds_config.Equal)
	leftIndex, err := strconv.Atoi(strings.Split(strings.Split(columnArr[0], ".")[0], "t")[1])
	if err != nil {
		logger.Errorf("convert string to int err:%v", err)
		return 0, 0
	}
	rightIndex, err := strconv.Atoi(strings.Split(strings.Split(columnArr[1], ".")[0], "t")[1])
	if err != nil {
		logger.Errorf("convert string to int err:%v", err)
		return 0, 0
	}
	return leftIndex, rightIndex
}

func GenerateConnectPredicateSingle(predicate rds.Predicate, tableId2index map[string]int) rds.Predicate {
	leftTableId := predicate.LeftColumn.TableId
	rightTableId := predicate.RightColumn.TableId
	leftColumnIndex, rightColumnIndex := GetPredicateColumnIndex(predicate)
	if _, ok := tableId2index[leftTableId]; !ok {
		tableId2index[leftTableId] = len(tableId2index)
	}
	if _, ok := tableId2index[rightTableId]; !ok {
		tableId2index[rightTableId] = len(tableId2index)
	}
	changedLeftColumnIndex := 2*tableId2index[leftTableId] + leftColumnIndex%2
	changedRightColumnIndex := 2*tableId2index[rightTableId] + rightColumnIndex%2
	predicate.PredicateStr = fmt.Sprintf("t%d.%s%st%d.%s", changedLeftColumnIndex, predicate.LeftColumn.ColumnId, predicate.SymbolType, changedRightColumnIndex, predicate.RightColumn.ColumnId)
	return predicate
}

func CheckPredicateIndex(predicate *rds.Predicate, tableId2index map[string]int) {
	return
	leftTableId := predicate.LeftColumn.TableId
	rightTableId := predicate.RightColumn.TableId
	leftColumnIndex, rightColumnIndex := GetPredicateColumnIndex(*predicate)
	if _, ok := tableId2index[leftTableId]; !ok {
		tableId2index[leftTableId] = len(tableId2index)
	}
	if _, ok := tableId2index[rightTableId]; !ok {
		tableId2index[rightTableId] = len(tableId2index)
	}
	changedLeftColumnIndex := 2*tableId2index[leftTableId] + leftColumnIndex%2
	changedRightColumnIndex := 2*tableId2index[rightTableId] + rightColumnIndex%2
	// 目前只有等于号,ml的有问题,只用用=连接
	//predicate.PredicateStr = fmt.Sprintf("t%d.%s%st%d.%s", changedLeftColumnIndex, predicate.LeftColumn.ColumnId, rds_config.Equal, changedRightColumnIndex, predicate.RightColumn.ColumnId)

	predicate.LeftColumn.ColumnIndex = changedLeftColumnIndex
	predicate.RightColumn.ColumnIndex = changedRightColumnIndex
	predicate.PredicateStr = GeneratePredicateStrNew(predicate)
}

func CheckPredicatesIsConnectedGraph(predicates []rds.Predicate) bool {
	edges := make([][2]int, len(predicates))
	for i, predicate := range predicates {
		//edges[i][0], edges[i][1] = GetPredicateColumnIndex(predicate)
		edges[i][0], edges[i][1] = predicate.LeftColumn.ColumnIndex, predicate.RightColumn.ColumnIndex
	}
	return isConnectedGraph(edges)
}

func isConnectedGraph(arr [][2]int) bool {
	// 构建邻接表
	graph := make(map[int][]int)
	nodes := make(map[int]bool)
	for _, edge := range arr {
		u, v := edge[0], edge[1]
		graph[u] = append(graph[u], v)
		graph[v] = append(graph[v], u)
		nodes[u] = true
		nodes[v] = true
	}

	visited := make(map[int]bool)
	dfs(arr[0][0], graph, visited)

	// 判断是否访问到所有节点
	for i := range nodes {
		if !visited[i] {
			return false // 存在未访问到的节点，说明不是连通图
		}
	}

	return true // 所有节点都被访问到，是连通图
}

func dfs(node int, graph map[int][]int, visited map[int]bool) {
	visited[node] = true

	for _, neighbor := range graph[node] {
		if !visited[neighbor] {
			dfs(neighbor, graph, visited)
		}
	}
}

func CheckSatisfyPredicateSupp(predicates []rds.Predicate, tableIndex map[string]int, supp float64) bool {
	if len(tableIndex) < 2 {
		return true
	}
	minSupp := float64(1)
	allCrossPredicates := true
	for _, predicate := range predicates {
		if predicate.PredicateType != 2 {
			allCrossPredicates = false
			if predicate.Support < minSupp {
				minSupp = predicate.Support
			}
		}
	}
	return allCrossPredicates || minSupp < supp
}

func CopyPredicate(predicate rds.Predicate) rds.Predicate {
	predicate_ := rds.Predicate{
		PredicateStr:              predicate.PredicateStr,
		LeftColumn:                predicate.LeftColumn,
		RightColumn:               predicate.RightColumn,
		ConstantValue:             predicate.ConstantValue,
		ConstantIndexValue:        predicate.ConstantIndexValue,
		SymbolType:                predicate.SymbolType,
		PredicateType:             predicate.PredicateType,
		UDFName:                   predicate.UDFName,
		Threshold:                 predicate.Threshold,
		Support:                   predicate.Support,
		Intersection:              predicate.Intersection,
		LeftColumnVectorFilePath:  predicate.LeftColumnVectorFilePath,
		RightColumnVectorFilePath: predicate.RightColumnVectorFilePath,
	}
	return predicate_
}

func GenerateConnectPredicate(predicate rds.Predicate, tableId2index map[string]int) []rds.Predicate {
	leftTableId := predicate.LeftColumn.TableId
	rightTableId := predicate.RightColumn.TableId
	leftColumnIndex, rightColumnIndex := GetPredicateColumnIndex(predicate)
	if _, ok := tableId2index[leftTableId]; !ok {
		tableId2index[leftTableId] = len(tableId2index)
	}
	if _, ok := tableId2index[rightTableId]; !ok {
		tableId2index[rightTableId] = len(tableId2index)
	}
	changedLeftColumnIndex := 2*tableId2index[leftTableId] + leftColumnIndex%2
	changedRightColumnIndex := 2*tableId2index[rightTableId] + rightColumnIndex%2
	predicate.PredicateStr = fmt.Sprintf("t%d.%s%st%d.%s", changedLeftColumnIndex, predicate.LeftColumn.ColumnId, predicate.SymbolType, changedRightColumnIndex, predicate.RightColumn.ColumnId)
	addedPredicate := CopyPredicate(predicate)
	addedPredicate.PredicateStr = fmt.Sprintf("t%d.%s%st%d.%s", changedLeftColumnIndex+1, predicate.LeftColumn.ColumnId, predicate.SymbolType, changedRightColumnIndex+1, predicate.RightColumn.ColumnId)
	return []rds.Predicate{predicate, addedPredicate}
}

func GenerateConnectPredicateNew(predicate rds.Predicate) []rds.Predicate {
	addedPredicate := CopyPredicate(predicate)
	addedPredicate.LeftColumn.ColumnIndex = predicate.LeftColumn.ColumnIndex + 1
	addedPredicate.RightColumn.ColumnIndex = predicate.RightColumn.ColumnIndex + 1
	//addedPredicate.PredicateStr = GeneratePredicateStr(addedPredicate.LeftColumn.ColumnId, addedPredicate.RightColumn.ColumnId, addedPredicate.LeftColumn.ColumnIndex, addedPredicate.RightColumn.ColumnIndex)
	addedPredicate.PredicateStr = GeneratePredicateStrNew(&addedPredicate)
	return []rds.Predicate{addedPredicate, predicate}
}

func DeletePredicateAndGini(predicates []rds.Predicate, giniIndexes []float64, predicate rds.Predicate) ([]rds.Predicate, []float64) {
	var result []rds.Predicate
	var resultGini []float64
	for i, p := range predicates {
		if p.PredicateStr != predicate.PredicateStr {
			result = append(result, p)
			resultGini = append(resultGini, giniIndexes[i])
		}
	}
	return result, resultGini
}

func SortPredicatesRelated(lhs []rds.Predicate) []rds.Predicate {
	lhs = slices.Clone(lhs)
	if len(lhs) < 2 {
		return lhs
	}

	var sortedLhs []rds.Predicate
	var visitedTableIndexes = map[int]bool{}
	var first rds.Predicate
	first = lhs[0]
	if first.PredicateType == 1 {
		lhs = lhs[1:]
	} else {
		for i, predicate := range lhs {
			if predicate.PredicateType == 1 {
				first = predicate
				lhs = append(lhs[:i], lhs[i+1:]...)
				break
			}
		}
	}

	leftTableIndex, rightTableIndex := GetPredicateColumnIndex(first)
	visitedTableIndexes[leftTableIndex] = true
	visitedTableIndexes[rightTableIndex] = true

	sortedLhs = append(sortedLhs, first)

	for len(lhs) > 0 {
		var i int
		for i = range lhs {
			leftTableIndex, rightTableIndex := GetPredicateColumnIndex(lhs[i])
			if visitedTableIndexes[leftTableIndex] || visitedTableIndexes[rightTableIndex] {
				break
			}
		}

		sortedLhs = append(sortedLhs, lhs[i])
		lhs = append(lhs[:i], lhs[i+1:]...)
	}

	return sortedLhs
}

func SortPredicatesCrossF(lhs []rds.Predicate) []rds.Predicate {
	var singlePre, crossPre []rds.Predicate
	for _, pre := range lhs {
		if pre.PredicateType == 1 {
			singlePre = append(singlePre, pre)
		} else if pre.PredicateType == 2 {
			crossPre = append(crossPre, pre)
		}
	}
	return append(crossPre, singlePre...)
}

func CheckPredicatesIndex(predicates []rds.Predicate, tableId2index map[string]int) {
	for i := range predicates {
		CheckPredicateIndex(&predicates[i], tableId2index)
	}
}

func sameLeftRight(p1 rds.Predicate, p2 rds.Predicate) bool {
	return p1.LeftColumn.ColumnId == p2.LeftColumn.ColumnId && p1.RightColumn.ColumnId == p2.RightColumn.ColumnId
}

func ForeignKeyPredicatesUnique(foreignKeyPredicates []rds.Predicate) ([]rds.Predicate, bool) {
	sz := len(foreignKeyPredicates)
	if sz == 0 {
		return nil, false
	}
	if sz%2 != 0 {
		return foreignKeyPredicates, false
	}

	var r []rds.Predicate
	for i := 0; i < sz-1; i++ {
		cur := foreignKeyPredicates[i]
		nex := foreignKeyPredicates[i+1]
		if sameLeftRight(cur, nex) {
			r = append(r, cur)
			i += 2
		} else {
			return foreignKeyPredicates, false
		}
	}
	return r, true
}

func GetMultiTablePredicatesColumn(predicates []rds.Predicate) []rds.Column { //去重了
	existColumn := make(map[string]struct{})
	columns := make([]rds.Column, 0)
	for _, predicate := range predicates {
		key := predicate.LeftColumn.TableId + "_" + predicate.LeftColumn.ColumnId
		if key == "_" { //groupby谓词的暂时处理
			continue
		}
		if _, exist := existColumn[key]; !exist {
			columns = append(columns, predicate.LeftColumn)
			existColumn[key] = struct{}{}
		}
		if predicate.PredicateType == 0 {
			continue
		} else if predicate.RightColumn.TableId != predicate.LeftColumn.TableId || predicate.RightColumn.ColumnId != predicate.LeftColumn.ColumnId {
			key := predicate.RightColumn.TableId + "_" + predicate.RightColumn.ColumnId
			if _, exist := existColumn[key]; !exist {
				columns = append(columns, predicate.RightColumn)
				existColumn[key] = struct{}{}
			}
		}
	}
	return columns
}

func GeneratePredicateStr(leftColumnName, rightColumnName string, leftColumnIndex, rightColumnIndex int) string {
	return fmt.Sprintf("t%d.%s=t%d.%s", leftColumnIndex, leftColumnName, rightColumnIndex, rightColumnName)
}

func GenerateConstPredicateStr(leftColumnName, leftColumnIndex, constantValue interface{}) string {
	return fmt.Sprintf("t%d.%s=%v", leftColumnIndex, leftColumnName, constantValue)
}

func GeneratePredicateStrNew(predicate *rds.Predicate) string {
	leftColName := predicate.LeftColumn.ColumnId
	rightColName := predicate.RightColumn.ColumnId
	//if leftColName[0] == rds_config.UdfColumnPrefix[0] {
	//	leftColName = strings.Split(leftColName, rds_config.UdfColumnConn)[1]
	//}
	//if rightColName[0] == rds_config.UdfColumnPrefix[0] {
	//	rightColName = strings.Split(rightColName, rds_config.UdfColumnConn)[1]
	//}
	if predicate.SymbolType == enum.ML {
		return fmt.Sprintf("%s('%s', t%d.%s, t%d.%s)", predicate.SymbolType, predicate.UDFName, predicate.LeftColumn.ColumnIndex, leftColName, predicate.RightColumn.ColumnIndex, rightColName)
	} else if predicate.SymbolType == enum.Similar {
		return fmt.Sprintf("%s('%s', t%d.%s, t%d.%s, %0.2f)", predicate.SymbolType, predicate.UDFName, predicate.LeftColumn.ColumnIndex, leftColName, predicate.RightColumn.ColumnIndex, rightColName, predicate.Threshold)
	} else {
		return fmt.Sprintf("t%d.%s=t%d.%s", predicate.LeftColumn.ColumnIndex, leftColName, predicate.RightColumn.ColumnIndex, rightColName)
	}
}

func GetMLPredicate(predicates []rds.Predicate) []rds.Predicate {
	var mlPredicates []rds.Predicate
	for _, predicate := range predicates {
		// 把ML谓词加入到ml谓词列表中
		if predicate.SymbolType == enum.Similar || predicate.SymbolType == enum.ML {
			mlPredicates = append(mlPredicates, predicate)
		}
	}
	return mlPredicates
}

func FindConnectedPre(rhs rds.Predicate, crossPres []rds.Predicate, tableId2index map[string]int) ([]rds.Predicate, []rds.Predicate) {
	var lhs, lhsCrossCandidate []rds.Predicate
	rhsLeftTableId := rhs.LeftColumn.TableId
	rhsRightTableId := rhs.RightColumn.TableId
	joinedMap := make(map[string]map[string]bool)
	for i, predicate := range crossPres {
		leftTableId := predicate.LeftColumn.TableId
		rightTableId := predicate.RightColumn.TableId
		if _, ok := joinedMap[leftTableId]; !ok {
			joinedMap[leftTableId] = make(map[string]bool)
		}
		if _, ok := joinedMap[rightTableId]; !ok {
			joinedMap[rightTableId] = make(map[string]bool)
		}
		joinedMap[leftTableId][rightTableId] = true
		joinedMap[rightTableId][leftTableId] = true
		if (rhsLeftTableId == leftTableId && rhsRightTableId == rightTableId) || (rhsLeftTableId == rightTableId && rhsRightTableId == leftTableId) {
			lhs = append(lhs, GenerateConnectPredicateSingle(predicate, tableId2index))
			lhsCrossCandidate = append(crossPres[:i], crossPres[i+1:]...)
			break
		}
	}

	// 当没有找到主外键可以直接连接满足rhs的时候需要继续匹配
	// 最多只有三张表,先按照三张表去实现吧
	// todo 大于三张表的情况
	if len(lhs) < 1 {
		midTableId := ""
		for midTableId = range joinedMap[rhsLeftTableId] {
			b := false
			for endTableId := range joinedMap[midTableId] {
				if endTableId == rhsRightTableId {
					b = true
					break
				}
			}
			if b {
				break
			}
		}
		for i, predicate := range crossPres {
			leftTableId := predicate.LeftColumn.TableId
			rightTableId := predicate.RightColumn.TableId
			if (rhsLeftTableId == leftTableId && midTableId == rightTableId) || (rhsLeftTableId == rightTableId && midTableId == leftTableId) {
				CheckPredicateIndex(&predicate, tableId2index)
				lhs = append(lhs, GenerateConnectPredicateSingle(predicate, tableId2index))
				lhsCrossCandidate = append(crossPres[:i], crossPres[i+1:]...)
				break
			}
		}
		crossPres = lhsCrossCandidate
		for i, predicate := range crossPres {
			leftTableId := predicate.LeftColumn.TableId
			rightTableId := predicate.RightColumn.TableId
			if (midTableId == leftTableId && rhsRightTableId == rightTableId) || (midTableId == rightTableId && rhsRightTableId == leftTableId) {
				lhs = append(lhs, GenerateConnectPredicateSingle(predicate, tableId2index))
				lhsCrossCandidate = append(crossPres[:i], crossPres[i+1:]...)
				break
			}
		}
	}
	return lhs, lhsCrossCandidate
}

func GetPredicateColumnIndexNew(predicate rds.Predicate) (int, int) {
	isConstant := predicate.PredicateType == 0
	if isConstant {
		return predicate.LeftColumn.ColumnIndex, -1
	}
	return predicate.LeftColumn.ColumnIndex, predicate.RightColumn.ColumnIndex
}

func PreStrToPredicate(pStr string, dataType map[string]map[string]string, tableId string, tableId2index map[string]int) rds.Predicate {
	leftColumnStr := strings.Split(strings.Split(pStr, "=")[0], ".")[1]
	rightColumnStr := strings.Split(strings.Split(pStr, "=")[1], ".")[1]
	tableIndex := tableId2index[tableId]
	predicateStr := GeneratePredicateStr(leftColumnStr, rightColumnStr, tableIndex*2, tableIndex*2+1)
	leftColumn := rds.Column{
		ColumnIndex: tableIndex * 2,
		TableId:     tableId,
		ColumnId:    leftColumnStr,
		ColumnType:  dataType[tableId][leftColumnStr],
		IsML:        false,
		JoinTableId: "",
	}
	rightColumn := rds.Column{
		ColumnIndex: tableIndex*2 + 1,
		TableId:     tableId,
		ColumnId:    rightColumnStr,
		ColumnType:  dataType[tableId][rightColumnStr],
		IsML:        false,
		JoinTableId: "",
	}
	predicate := rds.Predicate{
		PredicateStr:  predicateStr,
		LeftColumn:    leftColumn,
		RightColumn:   rightColumn,
		ConstantValue: nil,
		SymbolType:    enum.Equal,
		PredicateType: 1,
		Support:       0,
		Intersection:  nil,
	}
	return predicate
}

func FillPredicate(predicate *rds.Predicate, dataType map[string]map[string]string, tableId2index map[string]int) {
	leftIndex := tableId2index[predicate.LeftColumn.TableId] * 2
	predicate.LeftColumn.ColumnIndex = leftIndex
	predicate.LeftColumn.ColumnType = dataType[predicate.LeftColumn.TableId][predicate.LeftColumn.ColumnId]

	if predicate.ConstantValue != nil {
		constValueStr := fmt.Sprintf("%v", predicate.ConstantValue)
		switch predicate.LeftColumn.ColumnType {
		//const (
		//	StringType = "string"
		//	IntType    = "int64"
		//	FloatType  = "float64"
		//	BoolType   = "bool"
		//	TimeType   = "time"
		//	TextType   = "text"
		//	IndexType  = "index" // 索引类型
		//)
		case "int64", "index":
			predicate.ConstantValue, _ = strconv.ParseInt(constValueStr, 10, 64)
		case "float64":
			predicate.ConstantValue, _ = strconv.ParseFloat(constValueStr, 64)
		case "bool":
			predicate.ConstantValue, _ = strconv.ParseBool(constValueStr)
		}

		predicate.PredicateType = 0
		predicate.PredicateStr = GenerateConstPredicateStr(predicate.LeftColumn.ColumnId, leftIndex, predicate.ConstantValue)
		return
	} else {
		rightIndex := tableId2index[predicate.RightColumn.TableId]*2 + 1
		predicate.RightColumn.ColumnIndex = rightIndex
		predicate.RightColumn.ColumnType = dataType[predicate.RightColumn.TableId][predicate.RightColumn.ColumnId]
		predicate.PredicateType = 1
		predicate.PredicateStr = GeneratePredicateStr(predicate.LeftColumn.ColumnId, predicate.RightColumn.ColumnId, leftIndex, rightIndex)
	}
}
