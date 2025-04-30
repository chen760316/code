package main

import (
	"github.com/yourbasic/bit"
	"rds-shenglin/rock-share/base/logger"
	"rds-shenglin/rock-share/global/enum"
	"rds-shenglin/rock-share/global/model/rds"
	"rds-shenglin/utils"
	"rds-shenglin/utils/train_data_util"
	"runtime/debug"
	"sync"
	"time"
)

type TaskTree struct {
	Rhs                          rds.Predicate                //根节点
	GiniIndex                    float64                      //该节点的gini系数
	Lhs                          []rds.Predicate              //0层为空、1层1个、2层2个
	RelatedRows                  map[string]*bit.Set          //lhs涉及到的行
	LhsCandidate                 []rds.Predicate              //候选lhs不一定能用上
	LhsCrossCandidate            []rds.Predicate              //跨表候选lhs
	LhsCandidateGiniIndex        []float64                    //普通谓词候选集的gini系数
	LhsCrossCandidateGiniIndex   []float64                    //跨表谓词候选集的gini系数
	TableId2index                map[string]int               //表id和其对应的索引，比如tableA的index是0，那他对应的索引就是t0和t1
	LhsCandidateRelatedRows      []map[string]*bit.Set        //普通谓词候选集涉及到的行
	LhsCrossCandidateRelatedRows []map[string]*bit.Set        //跨表谓词候选集涉及到的行
	TrainDataColumnsType         map[string]map[string]string //xy训练集的列类型
	TrainDataColumns             map[string][]string          //xy训练集的列,有序
	Confidence                   float64                      // 节点对应规则的置信度
}

func createNode(rhs rds.Predicate, lhs []rds.Predicate, lhsCandidate []rds.Predicate, LhsCrossCandidate []rds.Predicate, trainDataColumnsType map[string]map[string]string) *TaskTree {
	root := &TaskTree{Rhs: rhs, Lhs: lhs, LhsCandidate: lhsCandidate, LhsCrossCandidate: LhsCrossCandidate}
	root.LhsCandidateGiniIndex = make([]float64, len(root.LhsCandidate))
	root.LhsCrossCandidateGiniIndex = make([]float64, len(root.LhsCrossCandidate))
	root.LhsCandidateRelatedRows = make([]map[string]*bit.Set, len(root.LhsCandidate))
	root.LhsCrossCandidateRelatedRows = make([]map[string]*bit.Set, len(root.LhsCrossCandidate))
	root.TableId2index = generateTableId2tableIndexNew(append(root.Lhs, root.Rhs))
	root.TrainDataColumns, root.TrainDataColumnsType = train_data_util.GenerateRootNodeHeader(rhs, trainDataColumnsType)
	return root
}

func generateTableId2tableIndexNew(predicates []rds.Predicate) map[string]int {
	result := make(map[string]int)
	for _, predicate := range predicates {
		result[predicate.LeftColumn.TableId] = predicate.LeftColumn.ColumnIndex / 2
		result[predicate.RightColumn.TableId] = predicate.RightColumn.ColumnIndex / 2
	}
	return result
}

func selectParNum(gv *GlobalV) int {
	tableRows := gv.RowSizes
	maxRow := 0
	parNum := len(TaskCh)
	for _, rows := range tableRows {
		if rows > maxRow {
			maxRow = rows
		}
	}
	for ; parNum > 1 && maxRow > 100000; parNum = parNum / 2 {
		maxRow = maxRow / 10
	}
	//return parNum
	// 陈胜林修改
	return 1
}

func generateCalNodes(rootPredicates []rds.Predicate, gv *GlobalV) []*TaskTree {
	// 把单表谓词按照表进行划分
	preMap := make(map[string][]rds.Predicate)
	for _, pre := range gv.Predicates {
		tableId := pre.LeftColumn.TableId
		if _, ok := preMap[tableId]; !ok {
			preMap[tableId] = []rds.Predicate{}
		}
		preMap[tableId] = append(preMap[tableId], pre)
	}

	// 主外键表列收集，不出现在 Y
	keyTableColumnMap := make(map[string]map[string]bool)
	for _, keyPre := range gv.CrossTablePredicates[0] {
		if _, ok := keyTableColumnMap[keyPre.LeftColumn.TableId]; !ok {
			keyTableColumnMap[keyPre.LeftColumn.TableId] = map[string]bool{}
		}
		keyTableColumnMap[keyPre.LeftColumn.TableId][keyPre.LeftColumn.ColumnId] = true
		if _, ok := keyTableColumnMap[keyPre.RightColumn.TableId]; !ok {
			keyTableColumnMap[keyPre.RightColumn.TableId] = map[string]bool{}
		}
		keyTableColumnMap[keyPre.RightColumn.TableId][keyPre.RightColumn.ColumnId] = true
	}

	// 构建树的根节点,一个谓词会构建出两棵树,一棵计算单表规则,一棵计算多表规则
	// 先计算完所有的单表规则,再计算多表规则
	// 这里的rhs都是单表谓词
	var singleRoots []*TaskTree
	var crossRoots []*TaskTree
	for _, rhs := range rootPredicates {
		// 相似度ML谓词不能作为y出现
		if rhs.SymbolType == enum.Similar || rhs.SymbolType == enum.ML {
			continue
		}
		connectPres := gv.CrossTablePredicates[0]
		if rhs.PredicateType == 3 { // y为跨表谓词

			// 跨表单行规则
			singleRoot := createNode(rhs, []rds.Predicate{}, connectPres, []rds.Predicate{}, gv.TrainDataColumnsType)
			singleRoots = append(singleRoots, singleRoot)

		} else { // y为单表谓词
			rhsTableId := rhs.LeftColumn.TableId
			curTablePre := utils.SingleTableFilterLhs(rhs, preMap[rhsTableId])

			singleRoot := createNode(rhs, []rds.Predicate{}, curTablePre, []rds.Predicate{}, gv.TrainDataColumnsType)
			singleRoots = append(singleRoots, singleRoot)
		}
	}
	roots := append(singleRoots, crossRoots...)

	logger.Infof("总共有%v棵树需要计算", len(roots))
	return roots
}

func BuildTrees(rootPredicates []rds.Predicate, gv *GlobalV) {

	t := time.Now().UnixMilli()
	logger.Infof("task id:%v, 多行规则发现开始", gv.TaskId)

	tolCalCnt := 0
	roots := generateCalNodes(rootPredicates, gv)
	parNum := selectParNum(gv)
	logger.Infof("task id:%v, 并发度:%v", gv.TaskId, parNum)
	ch := make(chan struct{}, parNum)
	for i := 0; i < parNum; i++ {
		ch <- struct{}{}
	}
	l := sync.Mutex{}
	var wg sync.WaitGroup
	for _, root := range roots {
		rhs := root.Rhs
		if gv.Stop() {
			logger.Infof("taskId:%v, 收到停止信号，不再跑新的规则发现任务树", gv.TaskId)
			return
		}
		//<-TaskCh
		<-ch
		wg.Add(1)
		root := root
		go func(rhs rds.Predicate, allPredicatesFilter *TaskTree) {
			defer func() {
				wg.Done()
				//TaskCh <- struct{}{}
				ch <- struct{}{}
				if err := recover(); err != nil {
					gv.HasError = true
					s := string(debug.Stack())
					logger.Errorf("recover.err:%v, stack:\n%v", err, s)
				}
			}()
			logger.Infof("taskId:%v,计算根节点为%v的树", gv.TaskId, rhs.PredicateStr)
			calCnt := calTree(rhs, root, gv)
			logger.Infof("taskId:%v,根节点为%v的树计算结构化规则数:%v", gv.TaskId, rhs.PredicateStr, calCnt)
			l.Lock()
			tolCalCnt += calCnt
			l.Unlock()
		}(rhs, root)
	}
	wg.Wait()
	logger.Infof("多行规则发现完成,耗时:%vms, 总共计算结构化规则数:%v", time.Now().UnixMilli()-t, tolCalCnt)
	logger.Infof("taskId:%v多行规则:%v", gv.TaskId, gv.MultiRuleSize)
}

func calTree(rhs rds.Predicate, root *TaskTree, gv *GlobalV) int {
	tolCalCnt := 0
	var decisionTreeNode []*TaskTree
	layer := 0
	if len(root.Lhs) > 0 {
		layer = len(root.Lhs) / 2
	}
	var finish bool
	currLayer := []*TaskTree{root}
	var nextLayer []*TaskTree

	// 新流程，将一整层的节点汇总起来然后给查错那边处理计算出supp和confidence
	// 对于需要继续拓展的节点，先执行决策树和计算对应子节点的giniIndex
	// 选出giniIndex最高的作为下一层的top1
	// 下层的所有节点也都是需要计算的
	// 对下层仍需要继续拓展的节点，迭代和top1节点求overlap，选出最小的topK个节点，然后进行拓展
	for len(currLayer) > 0 {
		layer++
		children := make([]*TaskTree, 0)
		for _, node := range currLayer {
			utils.SortPredicates(node.Lhs, false)
			utils.SortPredicates(node.LhsCandidate, false)
			utils.SortPredicates(node.LhsCrossCandidate, false)

			//检查，记录当前lhs有关的
			related, inRelated, crossRelated, crossInRelated := checkCandidatesRelated(node)
			// 可以在lhs中新添加的谓词集合
			// todo 这里跨表谓词的候选集需要加入一些限制，根据测试造的规则进行限制，主外键的t0和t1成对出现，t0跨表倒t2之后，不再出现t0相关的谓词
			addPredicateIndex := make([]int, 0)
			addPredicateIndex = append(addPredicateIndex, related...)
			addPredicateIndex = append(addPredicateIndex, crossRelated...)
			tmpGini := make([]float64, len(addPredicateIndex))
			tmpRelatedRows := make([]map[string]*bit.Set, len(addPredicateIndex))
			// 这里需要一个方法要从父节点中把子节点的gini系数拿到
			for i, index := range related {
				tmpGini[i] = node.LhsCandidateGiniIndex[index]
				tmpRelatedRows[i] = node.LhsCandidateRelatedRows[index]
			}
			for i, index := range crossRelated {
				tmpGini[len(related)+i] = node.LhsCrossCandidateGiniIndex[index]
				tmpRelatedRows[len(related)+i] = node.LhsCrossCandidateRelatedRows[index]
			}

			hasCrossPredicate := len(node.TableId2index) > 1

			for i, index := range addPredicateIndex {
				var lhsP rds.Predicate
				if i < len(related) { //还没到联表的谓词
					lhsP = node.LhsCandidate[index]
				} else {
					lhsP = node.LhsCrossCandidate[index]
				}
				// 当规则已经是跨表的时候,表内谓词的support必须小于某个值
				if hasCrossPredicate && lhsP.Support > gv.CrossTablePredicateSupp {
					continue
				}
				// TODO node.Lhs 和 lhsP 可能要判断是否同一列例如 ac=ac 和 similar("xx",ac,ac)

				// 需要保持谓词的编号是统一，当碰到跨表谓词的时候需要对谓词进行一定的变形
				// 比如当前谓词中有t0.a=t1.a，表示A表，新添加谓词t0.b=t2.a，t0.b来自B表，这时候需要将谓词转换成t0.a=t2.b，并且添加谓词t1.a=t3.b
				tableId2index := make(map[string]int, len(node.TableId2index))
				for tableId, indexId := range node.TableId2index {
					tableId2index[tableId] = indexId
				}
				if len(tableId2index) > 3 { //最多三张表的规则
					continue
				}
				childLhs := make([]rds.Predicate, len(node.Lhs))
				copy(childLhs, node.Lhs)
				if i >= len(related) { //表示该谓词是跨表谓词
					if len(tableId2index) > 2 { // 最多三张表相关联
						continue
					}
					//childLhs = append(childLhs, utils.GenerateConnectPredicate(lhsP, tableId2index)...)
					childLhs = append(childLhs, utils.GenerateConnectPredicateNew(lhsP)...)
				} else {
					utils.CheckPredicateIndex(&lhsP, tableId2index)
					childLhs = append(childLhs, lhsP)
				}
				tableId2index[lhsP.LeftColumn.TableId] = lhsP.LeftColumn.ColumnIndex / 2
				tableId2index[lhsP.RightColumn.TableId] = lhsP.RightColumn.ColumnIndex / 2

				child := &TaskTree{
					Rhs:                  rhs,
					Lhs:                  childLhs,
					TableId2index:        tableId2index,
					TrainDataColumns:     root.TrainDataColumns,
					TrainDataColumnsType: root.TrainDataColumnsType,
				}
				if i < len(related) { //还没到联表的谓词
					child.GiniIndex = node.LhsCandidateGiniIndex[index]
					child.RelatedRows = node.LhsCandidateRelatedRows[index]
					child.LhsCandidate = make([]rds.Predicate, 0)
					for _, tmpIndex := range related[i+1:] {
						child.LhsCandidate = append(child.LhsCandidate, node.LhsCandidate[tmpIndex])
					}
					for _, tmpIndex := range inRelated {
						child.LhsCandidate = append(child.LhsCandidate, node.LhsCandidate[tmpIndex])
					}
					child.LhsCrossCandidate = make([]rds.Predicate, len(node.LhsCrossCandidate))
					copy(child.LhsCrossCandidate, node.LhsCrossCandidate)
				} else { //当前为联表谓词
					child.GiniIndex = node.LhsCrossCandidateGiniIndex[index]
					child.RelatedRows = node.LhsCrossCandidateRelatedRows[index]
					child.LhsCandidate = make([]rds.Predicate, 0)
					for _, tmpIndex := range inRelated {
						child.LhsCandidate = append(child.LhsCandidate, node.LhsCandidate[tmpIndex])
					}
					child.LhsCrossCandidate = make([]rds.Predicate, 0)
					for _, tmpIndex := range addPredicateIndex[i+1:] {
						child.LhsCrossCandidate = append(child.LhsCrossCandidate, node.LhsCrossCandidate[tmpIndex])
					}
					for _, tmpIndex := range crossInRelated {
						child.LhsCrossCandidate = append(child.LhsCrossCandidate, node.LhsCrossCandidate[tmpIndex])
					}
				}
				child.LhsCandidateGiniIndex = make([]float64, len(child.LhsCandidate))
				child.LhsCrossCandidateGiniIndex = make([]float64, len(child.LhsCrossCandidate))
				child.LhsCandidateRelatedRows = make([]map[string]*bit.Set, len(child.LhsCandidate))
				child.LhsCrossCandidateRelatedRows = make([]map[string]*bit.Set, len(child.LhsCrossCandidate))
				children = append(children, child)
				continue
			}
		}

		if len(children) < 1 {
			logger.Infof("需要计算的节点数为0")
			return tolCalCnt
		}
		// 汇总一层需要计算的节点，调用刘鹏提供的方法，计算出这一组节点各自的supp和confidence
		hasRules, prunes, isDeletes, calCnt := calNodes(children, gv)
		tolCalCnt += calCnt
		// 根据返回结果，处理那些节点可以生成规则，哪些节点需要走决策树和计算gini系数
		for i := range hasRules { //和上面循环的i顺序应该一致
			child := children[i]
			hasRule := hasRules[i]
			prune := prunes[i]
			isDelete := isDeletes[i]
			if isDelete {
				for _, nextNode := range nextLayer {
					nextNode.LhsCandidate, nextNode.LhsCandidateGiniIndex = utils.DeletePredicateAndGini(nextNode.LhsCandidate, nextNode.LhsCandidateGiniIndex, child.Lhs[len(child.Lhs)-1])
					nextNode.LhsCrossCandidate, nextNode.LhsCrossCandidateGiniIndex = utils.DeletePredicateAndGini(nextNode.LhsCrossCandidate, nextNode.LhsCrossCandidateGiniIndex, child.Lhs[len(child.Lhs)-1])
				}
			}
			if prune {
				if hasRule {
					gv.RuleSizeLock.Lock()
					gv.MultiRuleSize++
					gv.RuleSizeLock.Unlock()
				}
				continue
			}

			nextLayer = append(nextLayer, child)
			decisionTreeNode = append(decisionTreeNode, child)

			if gv.Stop() {
				logger.Infof("taskId:%v, 收到停止信号，终止规则发现任务树:%v", gv.TaskId, rhs.PredicateStr)
				finish = true
			}
			if finish {
				break
			}
		}

		// 从nextLayer中筛选出topK个节点进行后续流程
		if layer >= gv.TopKLayer && len(nextLayer) > gv.TopKSize {
			nextLayer = nextLayer[:gv.TopKSize]
		}

		currLayer = nextLayer
		nextLayer = make([]*TaskTree, 0)

		if finish || layer >= gv.TreeLevel {
			break
		}
	}
	if gv.enableDecisionTree {
		decisionTreeRuleSize := CalDecisionNodes(gv, decisionTreeNode)
		gv.RuleSizeLock.Lock()
		gv.MultiRuleSize += decisionTreeRuleSize
		gv.RuleSizeLock.Unlock()
	}
	return tolCalCnt
}

func checkMutex(lhs []rds.Predicate, rhs rds.Predicate, mutexGroup map[string][]int, groupSize int) bool {
	tmpP := make([]rds.Predicate, len(lhs)+1)
	copy(tmpP, lhs)
	tmpP[len(lhs)] = rhs
	relatedColumns := make(map[string]bool)
	for _, predicate := range tmpP {
		relatedColumns[predicate.LeftColumn.ColumnId] = true
		relatedColumns[predicate.RightColumn.ColumnId] = true
	}
	count := make([]int, groupSize)
	for column := range relatedColumns {
		if arr, ok := mutexGroup[column]; ok {
			for _, i := range arr {
				count[i]++
			}
		}
	}
	for _, i := range count {
		if i > 1 {
			return false
		}
	}
	return true
}

func getTreeTables(node *TaskTree) map[string]struct{} {
	currentTables := make(map[string]struct{})
	currentTables[node.Rhs.LeftColumn.TableId] = struct{}{}
	if node.Rhs.RightColumn.TableId != "" {
		currentTables[node.Rhs.RightColumn.TableId] = struct{}{}
	}
	for _, lhs := range node.Lhs {
		currentTables[lhs.LeftColumn.TableId] = struct{}{}
		if lhs.RightColumn.TableId != "" {
			currentTables[lhs.RightColumn.TableId] = struct{}{}
		}
	}
	return currentTables
}

func checkCandidatesRelated(node *TaskTree) (related []int, inRelated []int, crossRelated []int, crossInRelated []int) {
	currentTables := getTreeTables(node)
	related = make([]int, 0)
	inRelated = make([]int, 0)
	crossRelated = make([]int, 0)
	crossInRelated = make([]int, 0)

	for i := range node.LhsCandidate {
		candidate := node.LhsCandidate[i]
		if _, exist := currentTables[candidate.LeftColumn.TableId]; exist {
			related = append(related, i)
			continue
		} else if _, exist := currentTables[candidate.RightColumn.TableId]; exist {
			related = append(related, i)
		} else {
			inRelated = append(inRelated, i)
		}
	}
	for i := range node.LhsCrossCandidate {
		candidate := node.LhsCrossCandidate[i]
		if _, exist := currentTables[candidate.LeftColumn.TableId]; exist {
			crossRelated = append(crossRelated, i)
			continue
		} else if _, exist := currentTables[candidate.RightColumn.TableId]; exist {
			crossRelated = append(crossRelated, i)
		} else {
			crossInRelated = append(crossInRelated, i)
		}
	}
	return related, inRelated, crossRelated, crossInRelated
}
